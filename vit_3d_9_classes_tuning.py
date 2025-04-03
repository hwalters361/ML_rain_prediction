# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_3d.py
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

# Define patch sizes for padding calculation
IMAGE_PATCH_SIZE = (8, 8)  # Example patch size, adjust accordingly
FRAME_PATCH_SIZE = 8

def pad_video(video, patch_height, patch_width):
    """Pads the video spatial dimensions to be divisible by the patch size."""
    _, _, h, w = video.shape  # (Frames, Channels, Height, Width)
    
    pad_h = (patch_height - h % patch_height) % patch_height
    pad_w = (patch_width - w % patch_width) % patch_width

    if pad_h > 0 or pad_w > 0:
        padding = (0, pad_w, 0, pad_h)  # (Left, Right, Top, Bottom)
        video = F.pad(video, padding, mode='constant', value=0)  # Pad with zeros

    return video

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def download_and_prepare_dataset_tuning(data_path, image_patch_size=IMAGE_PATCH_SIZE):
    """Loads and preprocesses the dataset, including padding."""
    with np.load(data_path, allow_pickle=True) as data:

        # randomly shuffle the data
        p = np.random.permutation(len(data["train_images"]))

        train_videos = np.nan_to_num(data["train_images"])
        valid_videos = np.nan_to_num(data["val_images"])
        test_videos = np.nan_to_num(data["test_images"])

        train_videos = np.concatenate(train_videos, valid_videos[:len(valid_videos)//2])
        test_videos = np.concatenate(test_videos, valid_videos[len(valid_videos)//2:])

        train_labels = data["train_labels"]
        valid_labels = data["val_labels"]
        test_labels = data["test_labels"]

        

        np.concatenate

    # Convert to PyTorch tensors
    train_videos = torch.tensor(train_videos, dtype=torch.float32)
    # valid_videos = torch.tensor(valid_videos, dtype=torch.float32)
    test_videos = torch.tensor(test_videos, dtype=torch.float32)

    # Add channel dimension: (B, C, F, H, W)
    train_videos = train_videos[:, None, :, :, :]
    test_videos = test_videos[:, None, :, :, :]

    # Pad videos to make spatial dimensions divisible by patch size
    train_videos = torch.stack([pad_video(v, *image_patch_size) for v in train_videos])
    test_videos = torch.stack([pad_video(v, *image_patch_size) for v in test_videos])

    # Convert labels to tensors
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    return (train_videos, train_labels), (valid_videos, valid_labels), (test_videos, test_labels)


def prepare_dataloader(videos, labels, loader_type="train", batch_size=32):
    """Creates a PyTorch DataLoader with preprocessed videos."""
    dataset = TensorDataset(videos, labels)

    if loader_type == "train":
        sampler = RandomSampler(dataset)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0  # Adjust based on system
    )
    
    return dataloader


# Load and preprocess dataset
# All 9 cluster data
# prepared_dataset = download_and_prepare_dataset("data/cluster_sst.npz")

# train_videos, train_labels = prepared_dataset[0]
# valid_videos, valid_labels = prepared_dataset[1]
# test_videos, test_labels = prepared_dataset[2]

# print(f'train_videos {train_videos.shape}, train_labels {train_labels.shape}')
# print(f'valid_videos {valid_videos.shape}, valid_labels {valid_labels.shape}')
# print(f'test_videos {test_videos.shape}, test_labels {test_labels.shape}')

# # Create DataLoaders
# trainloader = prepare_dataloader(train_videos, train_labels, "train", batch_size=32)
# validloader = prepare_dataloader(valid_videos, valid_labels, "valid", batch_size=32)
# testloader = prepare_dataloader(test_videos, test_labels, "test", batch_size=32)


# dataloader
def load_data(data_dir="./data/cluster_sst.npz"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    return trainset, testset


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

import torch.nn.functional as F
class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 9  # Number of independent classifications
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = image_patch_size[0], p2 = image_patch_size[1], pf = frame_patch_size),
            nn.LayerNorm(channels * image_patch_size[0] * image_patch_size[1] * frame_patch_size),
            nn.Linear(channels * image_patch_size[0] * image_patch_size[1] * frame_patch_size, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_outputs * num_classes)  # Output (batch_size, 9 * 4)
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x.view(b, self.num_outputs, self.num_classes)  # Reshape to (batch_size, 9, 4)


#### End of given code

class History:
    '''
    History class keeps track of our training loss, training accuracy, validation loss,
    and validation accuracy
    '''
    def __init__(self):
        self.training_loss = []
        self.training_accuracy = []
        
        self.validation_loss = []
        self.validation_accuracy = []
    
    def update(self, train_loss, train_acc, valid_loss, valid_acc):
        self.training_loss.append(train_loss)
        self.training_accuracy.append(train_acc)
        
        self.validation_loss.append(valid_loss)
        self.validation_accuracy.append(valid_acc)

from torch.utils.data import DataLoader
import torch.optim as optim
import torch

def run_experiment(trainloader, validloader, testloader):
    # batch_size = 8  # Adjust as needed
    # trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # validloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
    # testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    # PATCH_SIZE = (8, 8, 8)
    # INPUT_SHAPE = (24, 89, 180, 1)
    # Define model
    # def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
    model = ViT(
        image_size=(96,184),  
        image_patch_size=(8, 8),
        frames=24, 
        frame_patch_size=8,
        num_classes=4, 
        dim=128, # initially 512
        depth=4, # initially 6
        heads=4, # initially 8
        dim_head= 32, #initially 64
        mlp_dim=256, # initially 1024
        channels=1,
    )
    # .cuda()  # Move to GPU if available


    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def train(model, trainloader, validloader, criterion, optimizer, epochs=100):
        model.train()
        history = History()
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            
            for videos, labels in trainloader:
                optimizer.zero_grad()
                outputs = model(videos)  # (batch, 9, 4)
                # changed line
                loss = criterion(outputs.view(-1, 4), labels.view(-1))  # Flatten for CrossEntropyLoss
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=2)  # Get class predictions for each of the 9 outputs
                # another changed line
                correct += (predicted == labels).sum().item()
                total += labels.numel()  # Count all predictions (batch_size * 9)

            train_acc = correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {train_acc:.4f}")

            valid_loss, valid_acc = validate(model, validloader, criterion)
            history.update(total_loss, train_acc, valid_loss, valid_acc)
        
        return history


    def validate(model, validloader, criterion):
        
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for videos, labels in validloader:
                # videos, labels = videos.cuda(), labels.cuda()
                outputs = model(videos)
                # this is another changed line from the previous file
                loss = criterion(outputs.view(-1, 4), labels.view(-1))
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=2)  # Get class predictions for each of the 9 outputs
                # another changed line
                correct += (predicted == labels).sum().item()
                total += labels.numel()  # Count all predictions (batch_size * 9)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {total_loss}")
        return total_loss, val_acc

    def test(model, testloader):
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for videos, labels in testloader:
                # videos, labels = videos.cuda(), labels.cuda()
                outputs = model(videos)
                _, predicted = torch.max(outputs, dim=2)  # Get class predictions for each of the 9 outputs
                correct += (predicted == labels).sum().item()
                total += labels.numel()  # Count all predictions (batch_size * 9)
        
        print(f"Test Accuracy: {correct / total:.4f}")

    # Train the model
    history = train(model, trainloader, validloader, criterion, optimizer, epochs=100)

    # Run evaluation
    test(model, testloader)
    return model, history


def save_model(model, path="savedmodels/vit3d_model.pth"):
    """Saves the trained PyTorch model to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path="vit3d_model.pth"):
    """Loads the model's saved state_dict.
    Example usage:
    # Initialize the same model architecture
    model = ViT(
        image_size=224, image_patch_size=16, frames=32, frame_patch_size=4,
        num_classes=10, dim=512, depth=6, heads=8, mlp_dim=1024
    ).cuda()

    load_model(model, "trained_vit3d.pth")
    """
    model.load_state_dict(torch.load(path))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {path}")
