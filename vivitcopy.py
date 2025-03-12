import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# Setting seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

data_path = "sst.npz"

# Constants
BATCH_SIZE = 32
INPUT_SHAPE = (24, 89, 180)
NUM_CLASSES = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
PATCH_SIZE = (8, 8, 8)
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 2
LAYER_NORM_EPS = 1e-6

# Load and preprocess dataset
def download_and_prepare_dataset(data_path):
    with np.load(data_path, allow_pickle=True) as data:
        train_videos = np.nan_to_num(data["train_images"]).astype(np.float32)
        valid_videos = np.nan_to_num(data["val_images"]).astype(np.float32)
        test_videos = np.nan_to_num(data["test_images"]).astype(np.float32)
        train_labels = data["train_labels"].astype(np.int64).flatten()
        valid_labels = data["val_labels"].astype(np.int64).flatten()
        test_labels = data["test_labels"].astype(np.int64).flatten()
    return (train_videos, train_labels), (valid_videos, valid_labels), (test_videos, test_labels)

# PyTorch Dataset class
class SSTDataset(data.Dataset):
    def __init__(self, videos, labels):
        self.videos = torch.tensor(videos, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.videos[idx], self.labels[idx]

# ViViT model
class TubeletEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.projection = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        return self.projection(x).flatten(2).transpose(1, 2)

class PositionalEncoder(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
    
    def forward(self, x):
        return x + self.position_embedding

class ViViT(nn.Module):
    def __init__(self, input_shape, embed_dim, num_heads, num_layers, num_classes):
        super().__init__()
        num_patches = (input_shape[0] // PATCH_SIZE[0]) * (input_shape[1] // PATCH_SIZE[1])
        self.tubelet_embedding = TubeletEmbedding(embed_dim, PATCH_SIZE)
        self.positional_encoder = PositionalEncoder(num_patches, embed_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.tubelet_embedding(x)
        x = self.positional_encoder(x)
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.fc(x)

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, device):
    model.to(device)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        print(f"Epoch {epoch+1}: Loss {total_loss / len(train_loader):.4f}, Accuracy {correct / len(train_loader.dataset):.4f}")

# Load dataset
(train_videos, train_labels), (valid_videos, valid_labels), (test_videos, test_labels) = download_and_prepare_dataset(data_path)
train_dataset = SSTDataset(train_videos, train_labels)
valid_dataset = SSTDataset(valid_videos, valid_labels)
test_dataset = SSTDataset(test_videos, test_labels)

train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize and train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViViT(INPUT_SHAPE, PROJECTION_DIM, NUM_HEADS, NUM_LAYERS, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train(model, train_loader, valid_loader, criterion, optimizer, device)