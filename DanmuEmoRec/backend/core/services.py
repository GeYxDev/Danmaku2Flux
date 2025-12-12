import math
import os
import jieba
import numpy as np
import requests
import re
import torch
from scipy.ndimage import gaussian_filter1d
from snownlp import SnowNLP
from torch import nn


class DanmakuCrawlerService:
    """
    Danmaku crawler service
    """
    @staticmethod
    def fetch_danmaku(bv):
        """
        Retrieve CID through BV number and crawl barrage data
        :param bv: Need to crawl the video BV number of the danmaku
        :return: Split and processed danmaku data
        """
        print(f"[Crawler] Crawling bullet comments from {bv}.")

        # Set request headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.bilibili.com/'
        }

        try:
            # Convert BV number to cid
            pagelist_url = f"https://api.bilibili.com/x/player/pagelist?bvid={bv}"
            pagelist_respond = requests.get(pagelist_url, headers=headers)
            pagelist_respond_data = pagelist_respond.json()

            # If there are errors in the request
            if pagelist_respond_data['code'] != 0:
                print(f"Failed to obtain cid: {pagelist_respond_data.get('message')}")
                return []

            # Extract cid
            cid = pagelist_respond_data['data'][0]['cid']

            # Using cid to request barrage XML files
            xml_url = f"https://comment.bilibili.com/{cid}.xml"
            xml_respond = requests.get(xml_url, headers=headers)
            xml_respond.encoding = 'utf-8'

            # Parse the obtained barrage XML data
            danmu_list = []
            pattern = re.compile(r'<d p="(.*?)">(.*?)</d>')
            matches = pattern.findall(xml_respond.text)

            # Carefully analyze every piece of bullet comment data
            for params, content in matches:
                # split attribute
                attributes = params.split(',')

                # Extract the time of barrage appearance
                video_time = float(attributes[0])

                # Extract bullet screen mode
                danmu_mode = int(attributes[1])
                # Determine whether the barrage is advanced based on its type
                danmu_type = "Advanced" if danmu_mode >= 4 else "Basic"

                # Put the processed barrage into the barrage list
                danmu_list.append({
                    "time": video_time,
                    "type": danmu_type,
                    "mode": danmu_mode,
                    "content": content
                })

            # Sort bullet comments in a video by time
            danmu_list.sort(key=lambda x: x['time'])

            print(f"Successfully obtained {len(danmu_list)} bullet comments.")

            return danmu_list

        except Exception as e:
            print(f"Error in bullet screen crawling process: {e}")
            return []


# noinspection PyAttributeOutsideInit
class TransformerDataPreprocessor:
    """
    Pre data processor for vector transformation model
    """
    # Store a reference to a unique instance
    _instance = None

    def __new__(cls):
        """
        Convert raw barrage data into types suitable for BERT and Transformer models
        """
        if cls._instance is None:
            cls._instance = super(TransformerDataPreprocessor, cls).__new__(cls)
            # Initialize default configuration state
            cls._instance.stopwords = set()
            cls._instance.num_bins = 100
            cls._instance.min_danmu_count = 100
            cls._instance.min_fill_ratio = 0.2
            # Reserved emotional symbols
            cls._instance.protected_symbols = {'!', '！', '?', '？', '~', '～', '6', '2', '3', '5', 'h', 'H', 'w', 'W'}
            # Remove invisible characters or completely garbled characters
            cls._instance.noise_pattern = re.compile(r'\s+')
            # Has the resource been loaded
            cls._instance.loaded = False
        return cls._instance

    def load_resources(self, stopwords_path='stopwords.txt', num_bins=100, min_danmu_count=100, min_fill_ratio=0.2):
        """
        Load stop words and initialize configuration parameters
        :param stopwords_path: Stop word list path
        :param num_bins: Number of video clips
        :param min_danmu_count: Minimum number of individual barrages
        :param min_fill_ratio: Minimum density of individual video shards
        :return: None
        """
        # If resources have already been loaded, do not reload again
        if self.loaded:
            return

        print(f"[Preprocessor] Loading resources and configuration.")

        # Update configuration
        self.num_bins = num_bins
        self.min_danmu_count = min_danmu_count
        self.min_fill_ratio = min_fill_ratio

        # Load stop words
        self.load_stopwords(stopwords_path)

        # Marking resource loading completed
        self.loaded = True
        print(f"[Preprocessor] Initialization completed.")

    @staticmethod
    def load_stopwords(stopwords_path):
        """
        Load stop word list
        :param stopwords_path: Stop Word List Path
        :return: Stop word set
        """
        if not os.path.exists(stopwords_path):
            print(f"Warning: Stopwords file {stopwords_path} not found. Running without stopwords filter.")
            return set()
        with open(stopwords_path, 'r', encoding='utf-8') as stopwords:
            return set([stopword.strip() for stopword in stopwords])

    def is_protected(self, word):
        """
        Protect emotional characters in bullet comments
        :param word: Characters under inspection
        :return: Is the character protected
        """
        for char in word:
            if char in self.protected_symbols:
                return True
        return False

    def clean_and_tokenize(self, text):
        """
        Clean the barrage and segment it into words
        :param text: Bullet comments that need to be handled
        :return: Processed barrage
        """
        if not text:
            return []

        # Remove the first and last blank spaces from the text
        text = text.strip()

        # Segment the text into words
        tokens = jieba.lcut(text)

        # Record valid characters
        valid_tokens = []
        for token in tokens:
            # Remove blank tokens
            if not token or self.noise_pattern.fullmatch(token):
                continue

            # Retain characters that are not in the stop word list or in protected words
            if (token not in self.stopwords) or self.is_protected(token):
                valid_tokens.append(token)

        # Return the processed barrage
        return valid_tokens

    def check_danmaku_quality(self, danmu_list, duration):
        """
        Check if the density and quality of the barrage meet the standards
        :param danmu_list: List of bullet comments for a single video
        :param duration: The duration of the corresponding video
        :return: Is the density and quality of the barrage qualified
        """
        # Check if the number of barrages meets the requirements
        if len(danmu_list) < self.min_danmu_count:
            return False, "Low count"

        # Check if the video duration is correct
        if duration <= 0:
            return False, "Zero duration"

        # Check if the barrage density within each segment meets the standard
        filled_bins = set()
        for d in danmu_list:
            # Calculate which video fragment the barrage belongs to
            b_idx = min(int((d['time'] / duration) * self.num_bins), self.num_bins - 1)
            # No empty shards will be recorded
            filled_bins.add(b_idx)

        # Check the empty bucket rate
        if (len(filled_bins) / self.num_bins) < self.min_fill_ratio:
            return False, "Sparse distribution"

        # The quality of the above inspection is recognized
        return True, "OK"

    def process(self, danmu_list, bv=None, title=None):
        """
        Batch processing of barrage raw data
        :param danmu_list: List of bullet comments that need to be processed
        :param bv: The BV number of the video to which the barrage belongs
        :param title: The title of the video to which the barrage belongs
        :return: Processed barrage list
        """
        # If resources and initialization are not completed, complete them first
        if not self.loaded:
            self.load_resources()

        # If the barrage list is empty, do not proceed with the following steps
        if not danmu_list:
            print(f"[Preprocessor] No danmaku data provided.")
            return None

        # Ensure that the barrage is sorted by time
        danmu_list.sort(key=lambda x: x['time'])

        # Obtain video duration
        duration = danmu_list[-1]['time'] if danmu_list else 0.0

        # Filter out videos with low barrage quality
        is_valid, reason = self.check_danmaku_quality(danmu_list, duration)
        # Videos that do not meet the standards will be skipped
        if not is_valid:
            print(f"[Preprocessor] Quality check failed for {bv}: {reason}")
            return None

        # Initialize N buckets for storing bullet comments
        bins = [[] for _ in range(self.num_bins)]

        # Sort the barrage into corresponding buckets one by one
        for item in danmu_list:
            # Extract barrage information
            content = item.get('content', '')
            timestamp = item.get('time', 0.0)

            # Clear and segmented to the barrage
            tokens = self.clean_and_tokenize(content)

            # Invalid barrage will be skipped
            if not tokens:
                continue

            # Calculate which bucket the barrage belongs to
            bin_idx = min(int((timestamp / duration) * self.num_bins), self.num_bins - 1)

            # Place the barrage in the corresponding bucket
            bins[bin_idx].append(tokens)

        print(f"[Preprocessor] Successfully processed {bv}.")

        # The result of processing the raw barrage data
        return {"title": title, "bv": bv, "tokenized_bins": bins}


# noinspection PyMethodMayBeStatic
class SentimentAnalyzerService:
    """
    SnowNLP sentiment analysis service
    """
    # Store a reference to a unique instance
    _instance = None

    def __new__(cls):
        """
        Initialize the sentiment analysis system
        """
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzerService, cls).__new__(cls)
            cls._instance.sigma = 1.5
            cls._instance.loaded = True
            print("[Sentiment] Service instance created.")
        return cls._instance

    def get_danmu_sentiment_score(self, bucket_danmus):
        """
        Calculate the sentiment score for a single bucket
        :param bucket_danmus: Need to predict emotional barrel barrage
        :return: The emotional score of the barrage in the current bucket
        """
        # If there are no bullet comments in the bucket,
        # it is considered that the emotions of the entire bucket are neutral
        if not bucket_danmus:
            return 0.0

        # List for storing barrage emotions
        sentiment_scores = []
        for tokens in bucket_danmus:
            # Skip empty danmu
            if not tokens:
                continue

            # Combine word segmentation into text for a single danmu
            text = "".join(tokens)

            # Attempt to conduct sentiment analysis
            # noinspection PyBroadException
            try:
                s = SnowNLP(text)
                # Map the original ratings from 0 to 1 to -1 to 1
                score = (s.sentiments - 0.5) * 2
                sentiment_scores.append(score)
            except:
                # Skip characters that cannot be processed
                continue

        # If no valid scores were generated return 0.0
        if not sentiment_scores:
            return 0.0

        # Return the mean sentiment of all danmus in this bucket
        return float(np.mean(sentiment_scores))

    def compute_danmu_density(self, raw_bins):
        """
        Calculate the barrage density of each bucket
        :param raw_bins: Bucket list in a single video
        :return: Return bucket density vector
        """
        # Count the number of danmus for each bucket
        counts = np.array([len(bucket) for bucket in raw_bins], dtype=float)

        # Normalize the mapping density to between 0 and 1
        max_val = np.max(counts)
        if max_val > 0:
            normalized_counts = counts / max_val
        else:
            # Dealing with situations where all densities are 0
            normalized_counts = counts

        # Return bucket density vector
        return normalized_counts

    def process_video_by_danmu(self, video_data):
        """
        Generate emotion vectors for each video based on bullet comments
        :param video_data: The data and barrage data of this video
        :return: Return emotional rating related data of the video
        """
        # Read buckets containing bullet comments
        raw_bins = video_data.get('tokenized_bins', [])

        # Calculate the sentiment generation vector for each bucket
        sentiment_raw = []
        for bucket in raw_bins:
            score = self.get_danmu_sentiment_score(bucket)
            sentiment_raw.append(score)
        sentiment_raw = np.array(sentiment_raw)

        # Calculate the density of each bucket to obtain the density vector
        density_raw = self.compute_danmu_density(raw_bins)

        # Apply smoothing to emotion vectors and density vectors
        sentiment_smooth = gaussian_filter1d(sentiment_raw, sigma=self.sigma)
        density_smooth = gaussian_filter1d(density_raw, sigma=self.sigma)

        # Constructing a composite vector of emotions and density
        final_vector = np.concatenate([sentiment_smooth, density_smooth]).tolist()

        # Return processed video information and vectors
        return {
            "title": video_data.get('title'),
            "bv": video_data.get('bv'),
            "duration": video_data.get('duration'),
            "feature_vector": final_vector
        }


# noinspection PyAttributeOutsideInit
class TransformerRecommendModelService:
    """
    Transformer emotion similarity video recommender
    """
    # Store a reference to a unique instance
    _instance = None

    class PositionalEncoding(nn.Module):
        """
        Encode the embedding location for data
        """
        def __init__(self, d_model, max_len=5000):
            """
            Position code generation
            :param d_model: Dimension of word embedding
            :param max_len: Maximum sequence length
            """
            super().__init__()
            # Store all location codes
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            # Generate location index
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            # Calculate scaling factor
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # Register the position encoding as a buffer instead of a trainable parameter
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            # Add corresponding positional encoding for each word embedding
            return x + self.pe[:, :x.size(1), :]

    class VideoSentimentTransformer(nn.Module):
        """
        Specialized Transformer for video emotion recognition
        """
        def __init__(self, parent_cls, n_features=2, d_model=128, n_head=4, num_layers=2, seq_len=100):
            """
            Initialize model structure
            :param parent_cls: External class reference
            :param n_features: The characteristic dimension of each time step
            :param d_model: Transformer internal representation dimension
            :param n_head: Multi head attention head count
            :param num_layers: Transformer encoder layers
            :param seq_len: Input sequence length
            """
            super().__init__()

            # Input Projection Layer: [batch, 100, 2] -> [batch, 100, 128]
            self.input_embedding = nn.Sequential(
                nn.Linear(n_features, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model)
            )
            # Learnable classification markers: [1, 1, 128]
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            # Mask Layer: [1, 1, 128]
            self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            # Positional encoding: 100 input tokens + 1 CLS token
            self.pos_encoder = parent_cls.PositionalEncoding(d_model, max_len=seq_len + 1)

            # Transformer encoder: Capture long-range dependencies
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, norm_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Decode head: Restore 128 dimensional features back to 2D
            self.decoder_head = nn.Linear(d_model, n_features)

            # CLS head: Responsible for overall emotional judgment
            self.cls_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, 8)
            )

        def forward(self, x, mask=None):
            """
            Forward propagation process
            :param x: Batch of video emotional feature sequences
            :param mask: Masks used during the training process
            :return: Reconstructed sequence and video sentiment labels
            """
            # x: [batch, 100, 2]
            batch_size = x.shape[0]

            # Embedding: [batch, 100, 2] -> [batch, 100, 128]
            x = self.input_embedding(x)

            # Masking: [batch, 100, 128] -> [batch, 101, 128]
            if mask is not None:
                mask_bool = mask.bool()
                if mask_bool.any():
                    # Expand mask_token to [batch, 100, 128] and overwrite positions
                    mask_token_expanded = self.mask_token.expand(batch_size, x.size(1), -1)
                    # Using Boolean index assignment
                    x[mask_bool] = mask_token_expanded[mask_bool]

            # Add CLS Token: [batch, 100, 128] -> [batch, 101, 128]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # Positional Encoding
            x = self.pos_encoder(x)

            # Transformer Logic: [batch, 101, 128] -> [batch, 101, 128]
            transformer_output = self.transformer_encoder(x)

            # Output Splitting
            # Take the first token, CLS as the full video fingerprint -> [batch, 128]
            video_vector = transformer_output[:, 0, :]

            # Reconstruction
            # Retrieve the last 100 tokens to restore the original data -> [batch, 100, 2]
            reconstructed_seq = self.decoder_head(transformer_output[:, 1:, :])

            # CLS regression prediction
            cls_pred = self.cls_head(video_vector)

            # Return the reconstructed sequence and video sentiment labels
            return reconstructed_seq, video_vector, cls_pred

    def __new__(cls):
        """
        Initialize Transformer recommendation system
        """
        if cls._instance is None:
            cls._instance = super(TransformerRecommendModelService, cls).__new__(cls)
            # Basic attribute initialization
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance.model = None
            cls._instance.loaded = False
            print(f"[RecommendService] Service instance created. Device: {cls._instance.device}.")
        return cls._instance

    def load_model(self, model_path="danmu_transformer_best.pth"):
        """
        Load model weights and configurations
        :param model_path: Model weight storage path
        :return: None
        """
        # Prevent duplicate loading
        if self.loaded:
            return

        print(f"[RecommendService] Loading model from {model_path}.")

        # 实例化内部模型类 (注意传入 self.__class__ 以便内部访问 PositionalEncoding)
        self.model = self.VideoSentimentTransformer(parent_cls=self.__class__, n_features=2, d_model=128, seq_len=100)

        # Load model weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print("[RecommendService] Model loaded successfully.")
        except FileNotFoundError:
            print(f"[RecommendService] Warning: Model file {model_path} not found. Using random weights.")
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True

    @staticmethod
    def mask_modeling(x, mask_ratio=0.15, mean_span_length=5):
        """
        Mask modeling enables models to learn how to mask data
        :param x: Input data
        :param mask_ratio: Mask coverage rate
        :param mean_span_length: Average continuous occlusion time step
        :return: Partially obscured mask
        """
        batch, seq_len, feats = x.shape

        # Generate cover
        mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=x.device)

        # At least the number of tokens that need to be masked for each sample
        n_mask = max(1, int(seq_len * mask_ratio))

        # Generate masks for each batch
        for b in range(batch):
            # Current number of masks
            current_masked = 0
            # Attempt count
            attempts = 0

            # Generate a mask until it meets the requirements
            while current_masked < n_mask and attempts < seq_len * 3:
                # Randomly generate mask length
                span_len = max(1, int(np.round(np.random.normal(mean_span_length, 2))))

                # Randomly select the starting position
                if seq_len - span_len <= 0:
                    start = 0
                else:
                    start = np.random.randint(0, seq_len - span_len + 1)

                # Only count the positions within the span that have not been obscured yet
                span_idx = torch.arange(start, start + span_len, device=mask.device)
                # noinspection PyUnresolvedReferences
                new_positions = (~mask[b, span_idx]).sum().item()
                mask[b, span_idx] = True

                # Update to mask quantity
                current_masked += new_positions
                attempts += 1

        # Return partially obscured mask
        return mask

    def _preprocess(self, sentiment_vector):
        """
        Transformer model input adaptation processing
        :param sentiment_vector: Video barrage emotion and density vector
        :return: Adapt to the vector input of Transformer
        """
        # Abnormal emotion vector processing
        if not sentiment_vector:
            return torch.zeros(1, 100, 2).to(self.device)

        # Obtain emotional and density vectors
        combined_vector = np.array(sentiment_vector, dtype=np.float32)

        # Split emotion vector and density vector
        sentiment_vector = combined_vector[:100]
        # The density vector should maintain the same range as the emotion vector
        density_vector = combined_vector[100:] * 2.0 - 1.0

        # Vector stacking
        sequence = np.stack([sentiment_vector, density_vector], axis=1)

        # Return the prepared data
        return torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, sentiment_vector):
        """
        Video sentiment vectorization based on Transformer
        :param sentiment_vector: Video barrage emotion and density vector
        :return: Converted video sentiment vector
        """
        # Automatically load when the model is not loaded
        if not self.loaded:
            self.load_model()

        # Return all 0 vectors when video vector does not exist
        if not sentiment_vector:
            return np.zeros(128).tolist()

        # Vector preprocessing
        sentiment_vector = self._preprocess(sentiment_vector)

        # Multi mask smooth inference
        ema = 0.6
        cls_ema = None

        # Loop multiple mask inference and exponential smoothing to obtain stable video feature vectors
        with torch.no_grad():
            for _ in range(8):
                mask = self.mask_modeling(sentiment_vector, mask_ratio=0.30, mean_span_length=12)
                _, video_vectors, _ = self.model(sentiment_vector, mask=mask)
                if cls_ema is None:
                    cls_ema = video_vectors
                else:
                    cls_ema = ema * cls_ema + (1 - ema) * video_vectors

        # Transfer inference results to CPU
        cls_out = cls_ema.cpu().numpy()

        # Return the converted video sentiment vector
        return cls_out.flatten().tolist()


# Initialize a global single pre data processor for vector transformation model
transformerDataPreprocessor = TransformerDataPreprocessor()

# Initialize a global single SnowNLP sentiment analysis service
sentimentAnalyzerService = SentimentAnalyzerService()

# Initialize a global single transformer emotion similarity video recommender
transformerRecommendModelService = TransformerRecommendModelService()
