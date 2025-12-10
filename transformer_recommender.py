import math
import json
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class DanmuSequenceDataset(Dataset):
    """
    Video emotion data loading
    """
    def __init__(self, dataset_path, seq_len=100):
        """
        Load a dataset containing video emotions and density
        :param dataset_path: The path where the dataset file is located
        :param seq_len: Emotion and density vector length
        """
        self.seq_len = seq_len

        # Load vector data from the dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, index):
        """
        Return the data adapted by Dataloader
        :param index: Data item index
        :return: Single prepared data
        """
        item = self.data[index]

        # Obtain emotional and density vectors
        combined_vector = np.array(item['vector'], dtype=np.float32)

        # Split emotion vector and density vector
        sentiment_vector = combined_vector[:self.seq_len]
        # The density vector should maintain the same range as the emotion vector
        density_vector = combined_vector[self.seq_len:] * 2.0 - 1.0

        # Vector stacking
        sequence = np.stack([sentiment_vector, density_vector], axis=1)

        # Return the prepared single data
        return torch.tensor(sequence, dtype=torch.float32), item.get('title', '')


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
        super(PositionalEncoding, self).__init__()
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
    def __init__(self, n_features=2, d_model=128, n_head=4, num_layers=2, seq_len=100):
        """
        Initialize model structure
        :param n_features: The characteristic dimension of each time step
        :param d_model: Transformer internal representation dimension
        :param n_head: Multi head attention head count
        :param num_layers: Transformer encoder layers
        :param seq_len: Input sequence length
        """
        super(VideoSentimentTransformer, self).__init__()

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
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 1)

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
            nn.Linear(64, n_features)
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


def train():
    """
    Model training process
    :return: None
    """
    # Configuration item
    DATA_DIR = "simplified_vector_danmu.json"
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    CLS_LOSS_WEIGHT = 1.0
    GRAD_CLIP_NORM = 1.0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    dataset = DanmuSequenceDataset(DATA_DIR, seq_len=100)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} videos for training.")

    # Model initialization
    model = VideoSentimentTransformer(n_features=2, d_model=128, seq_len=100).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    recon_criterion = nn.MSELoss()
    cls_criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_data, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            # Batch shape: [batch, 100, 2]
            original_seq = batch_data.to(DEVICE)

            # Generate mask
            mask = mask_modeling(original_seq, mask_ratio=0.15, mean_span_length=5)

            # Forward propagation
            reconstructed_seq, _, cls_pred = model(original_seq, mask=mask)

            # Only calculate the loss of the obscured position
            float_mask = mask.unsqueeze(-1).float()
            if float_mask.sum() > 0:
                recon_loss = recon_criterion(reconstructed_seq * float_mask, original_seq * float_mask)
            else:
                recon_loss = recon_criterion(reconstructed_seq, original_seq)

            # Calculate the cls loss
            cls_loss = cls_criterion(cls_pred, original_seq.mean(dim=1))

            # Calculate the total loss
            loss = recon_loss + CLS_LOSS_WEIGHT * cls_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()

        # Calculate the average printing loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.6f}")

    # Save the final model
    torch.save(model.state_dict(), "danmu_transformer.pth")
    print("Training Complete! Model saved.")


def inference(dataset_path, output_path, batch_size=64):
    """
    Inference conversion of video emotion and density vector data
    :param dataset_path: Need to convert dataset address
    :param output_path: Output the address of the video emotional label file
    :param batch_size: Batch size for single processing
    :return: None
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = VideoSentimentTransformer(n_features=2, d_model=128, seq_len=100)
    model.load_state_dict(torch.load("danmu_transformer.pth", map_location=device))
    model.to(device)
    model.eval()

    # Load data
    dataset = DanmuSequenceDataset(dataset_path, seq_len=100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Store the converted video sentiment tags and video titles
    all_convert_results = []

    # Conversion processing
    with torch.no_grad():
        for batch_data, batch_titles in tqdm(dataloader, desc="Processing"):
            original_seq = batch_data.to(device)
            current_batch_size = batch_data.size(0)

            # Initialize accumulator
            cls_accum = torch.zeros(current_batch_size, model.cls_token.shape[-1], device=device)

            # Create a CLS vector for accumulating all zero tensors 8 times for inference
            for _ in range(8):
                mask = mask_modeling(original_seq, mask_ratio=0.15, mean_span_length=5)
                _, video_vectors, _ = model(original_seq, mask=mask)
                cls_accum += video_vectors

            # Transfer inference results to CPU
            cls_avg = (cls_accum / float(8)).cpu().numpy()

            # Save conversion results
            for i in range(len(batch_titles)):
                all_convert_results.append({
                    "title": batch_titles[i],
                    "embedding": cls_avg[i].tolist()
                })

    # Save results to local file
    print(f"Saving embeddings to {output_path}.")
    with open(output_path, 'w', encoding='utf-8') as f:
        # noinspection PyTypeChecker
        json.dump(all_convert_results, f, ensure_ascii=False, indent=None)


if __name__ == "__main__":
    train()
    # inference("simplified_vector_danmu.json", "transformer_vector_danmu.json")
