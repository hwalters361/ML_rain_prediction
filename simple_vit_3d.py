# Imports

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import torch.optim as optim
from tqdm import tqdm  # Progress bar for training

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = "sst_df"
BATCH_SIZE = 32
INPUT_SHAPE = (24, 89, 180, 1)
NUM_CLASSES = 4

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 15 # Originally 60

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2
# TODO : visualize patch sizes, reduce model size 

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 2 # Prof Gabe says this should be 2 - 4, but the original paper sets it at 8 

DIM = 1024
DEPTH = 6
HEADS = 8
MLP_DIM = 2048

DATA_PATH = "sst.npz"


class VideoDataset(Dataset):
    """Dataset for loading video data."""
    
    def __init__(self, videos, labels):
        self.videos = torch.tensor(videos, dtype=torch.float32)  # Convert to float32 for efficiency
        self.labels = torch.tensor(labels, dtype=torch.long)  # Assuming classification task
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.videos[idx], self.labels[idx]

def download_and_prepare_dataset(data_path):
    """Loads dataset from .npz file and returns PyTorch DataLoader objects."""
    with np.load(data_path, allow_pickle=True) as data:
        # Load and clean data
        train_videos = np.nan_to_num(data["train_images"])
        valid_videos = np.nan_to_num(data["val_images"])
        test_videos = np.nan_to_num(data["test_images"])

        train_labels = data["train_labels"].astype(np.int64)
        valid_labels = data["val_labels"].astype(np.int64)
        test_labels = data["test_labels"].astype(np.int64)

    return train_videos, train_labels, valid_videos, valid_labels, test_videos, test_labels

def prepare_dataloader(videos, labels, batch_size=BATCH_SIZE, shuffle=True):
    """Creates a PyTorch DataLoader."""
    dataset = VideoDataset(videos, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

## Everything below this comment is for the 3D Simple ViT Model

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_height, image_width, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        # image_height
        # image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by the frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, video):
        *_, h, w, dtype = *video.shape, video.dtype

        x = self.to_patch_embedding(video)
        pe = posemb_sincos_3d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
    

def run_experiment(trainloader, testloader, validloader):
    model = SimpleViT(
        image_height=89, 
        image_width=180, 
        image_patch_size=8, 
        frames=24, 
        frame_patch_size=8, 
        num_classes=4, 
        dim=1024, 
        depth=6, 
        heads=8, 
        mlp_dim=2048
    ).to(device)

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # Training function
    def train_model(model, train_loader, valid_loader, epochs=15):
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
                videos, labels = videos.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # Validation Phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for videos, labels in valid_loader:
                    videos, labels = videos.to(device), labels.to(device)
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(valid_loader)
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return train_losses, val_losses

    # Run training
    train_losses, val_losses = train_model(model, train_loader, valid_loader)


    # def run_experiment(trainloader, testloader, validloader):
    #     # Initialize model
    #     model = SimpleViT(image_height = INPUT_SHAPE[1], image_width = INPUT_SHAPE[2], image_patch_size = PATCH_SIZE, frames = INPUT_SHAPE[0], frame_patch_size = PATCH_SIZE[2], num_classes = NUM_CLASSES, dim = DIM, depth = DEPTH, heads = HEADS, mlp_dim = MLP_DIM, channels = 3, dim_head = 64)


    #     # Compile the model with the optimizer, loss function
    #     # and the metrics.
    #     # optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    #     model.

    #     # Train the model.
    #     _ = model.fit(trainloader, epochs=EPOCHS, validation_data=validloader)

    #     _, accuracy, top_5_accuracy = model.evaluate(testloader)
    #     print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    #     print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    #     return model


    # model = run_experiment()
