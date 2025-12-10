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
        density_vector = combined_vector[self.seq_len:]

        # Vector stacking
        sequence = np.stack([sentiment_vector, density_vector], axis=1)

        # Return the prepared single data
        return torch.tensor(sequence), item.get('title', '')


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

        # Linear embedding layer: [batch, 100, 2] -> [batch, 100, 128]
        self.input_embedding = nn.Linear(n_features, d_model)
        # Learnable classification markers: [1, 1, 128]
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Positional encoding: 100 input tokens + 1 CLS token
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 1)

        # Transformer encoder: Capture long-range dependencies
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decode head: Restore 128 dimensional features back to 2D
        self.decoder_head = nn.Linear(d_model, n_features)

    def forward(self, x):
        """
        Forward propagation process
        :param x: Batch of video emotional feature sequences
        :return: Reconstructed sequence and video sentiment labels
        """
        # x: [batch, 100, 2]
        batch_size = x.shape[0]

        # Embedding: [batch, 100, 2] -> [batch, 100, 128]
        x = self.input_embedding(x)

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

        # Return the reconstructed sequence and video sentiment labels
        return reconstructed_seq, video_vector


def mask_modeling(x, mask_ratio=0.15):
    """
    Mask modeling enables models to learn how to mask data
    :param x: Input data
    :param mask_ratio: Mask coverage rate
    :return: Partially covered data
    """
    # Generate cover
    mask = torch.rand(x.shape[:2]) < mask_ratio
    # Match feature dimensions
    mask = mask.unsqueeze(-1).to(x.device)

    # Covering the data
    x_masked = x * (1 - mask.float())

    # Return processed data
    return x_masked


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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    dataset = DanmuSequenceDataset(DATA_DIR, seq_len=100)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} videos for training.")

    # Model initialization
    model = VideoSentimentTransformer(n_features=2, d_model=128, seq_len=100).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_data, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            # Batch shape: [batch, 100, 2]
            original_seq = batch_data.to(DEVICE)

            # Manufacturing obscured data
            masked_data = mask_modeling(original_seq, mask_ratio=0.15)

            # Forward propagation
            reconstructed_seq, _ = model(masked_data)

            # Calculate the loss
            loss = criterion(reconstructed_seq, original_seq)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
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

            # Model inference
            _, video_vectors = model(original_seq)

            # Transfer inference results to CPU
            video_vectors_cpu = video_vectors.cpu().numpy()

            # Save conversion results
            for i in range(len(batch_titles)):
                all_convert_results.append({
                    "title": batch_titles[i],
                    "reconstructed_vector": video_vectors_cpu[i].tolist()
                })

    # Save results to local file
    print(f"Saving embeddings to {output_path}.")
    with open(output_path, 'w', encoding='utf-8') as f:
        # noinspection PyTypeChecker
        json.dump(all_convert_results, f, ensure_ascii=False, indent=None)


if __name__ == "__main__":
    train()
    # inference("simplified_vector_danmu.json", "transformer_vector_danmu.json")
