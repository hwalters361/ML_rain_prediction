import fiona
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from time import time
from tqdm import tqdm
import xarray as xr
import torch
import torch.nn as nn
import matplotlib.colors as colors
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree

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

def plotHistory(history):
    """
    Plot training and validation metrics over epochs
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.training_loss, label='Training Loss')
    if history.validation_loss[0] is not None:
        plt.plot(history.validation_loss, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.training_accuracy, label='Training Accuracy')
    if history.validation_accuracy[0] is not None:
        plt.plot(history.validation_accuracy, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

seed = 1234
np.random.seed(seed)


# Define patch sizes for padding calculation
IMAGE_PATCH_SIZE = (8, 8)  # Example patch size, adjust accordingly
FRAME_PATCH_SIZE = 8

# def prepare_dataloader(videos, labels, loader_type="train", batch_size=32):
#     """Creates a PyTorch DataLoader with preprocessed videos and starting month information."""
#     dataset = SSTDataset(videos, labels)
    
#     if loader_type == "train":
#         sampler = RandomSampler(dataset)
#     else:
#         sampler = None
    
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         shuffle=(sampler is None),
#         num_workers=0  # Adjust based on system
#     )
    
#     return dataloader

# def pad_video(video, patch_height, patch_width):
#     """Pads the video spatial dimensions to be divisible by the patch size."""
#     _, _, h, w = video.shape  # (Frames, Channels, Height, Width)
    
#     pad_h = (patch_height - h % patch_height) % patch_height
#     pad_w = (patch_width - w % patch_width) % patch_width

#     if pad_h > 0 or pad_w > 0:
#         padding = (0, pad_w, 0, pad_h)  # (Left, Right, Top, Bottom)
#         video = F.pad(video, padding, mode='constant', value=0)  # Pad with zeros

#     return video
    
# Define patch sizes for padding calculation
IMAGE_PATCH_SIZE = (8, 8)  # Example patch size, adjust accordingly
FRAME_PATCH_SIZE = 8

# def download_and_prepare_dataset(data_path, image_patch_size=IMAGE_PATCH_SIZE):
#     """Loads and preprocesses the dataset, including padding."""
#     with np.load(data_path, allow_pickle=True) as data:
#         train_videos = np.nan_to_num(data["train_images"])
#         test_videos = np.nan_to_num(data["test_images"])

#         train_labels = data["train_labels"]
#         test_labels = data["test_labels"]

#         train_start_months = data["train_start_months"]
#         test_start_months = data["test_start_months"]
    
#     # # Extract starting months from the data
    
#     # Convert to PyTorch tensors
#     train_videos = torch.tensor(train_videos, dtype=torch.float32)
#     test_videos = torch.tensor(test_videos, dtype=torch.float32)

#     # Add channel dimension: (B, C, F, H, W)
#     train_videos = train_videos[:, None, :, :, :]
#     test_videos = test_videos[:, None, :, :, :]

#     # Pad videos to make spatial dimensions divisible by patch size
#     train_videos = torch.stack([pad_video(v, *image_patch_size) for v in train_videos])
#     test_videos = torch.stack([pad_video(v, *image_patch_size) for v in test_videos])

#     # Convert labels to tensors
#     train_labels = torch.tensor(train_labels, dtype=torch.long)
#     test_labels = torch.tensor(test_labels, dtype=torch.long)

#     train_start_months = torch.tensor(train_start_months, dtype=torch.int)
#     test_start_months = torch.tensor(test_start_months, dtype=torch.int)

#     # return (train_videos, train_labels), (test_videos, test_labels)
#     return (train_videos, train_labels, train_start_months), (test_videos, test_labels, test_start_months)
#     # return (train_videos, train_labels), (test_videos, test_labels)



# change to built in methods
class RainPrediction(nn.Module):
    def __init__(self, num_regions, num_classes):
        super().__init__()
        self.num_outputs = 9  # Number of independent classifications
        
        # Enhanced architecture with more layers and batch normalization
        # self.model = nn.Sequential(
        #     nn.Dropout(0.05),
        #     nn.Linear(num_regions, 20),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Linear(20, num_classes * self.num_outputs)  # Multiply by num_outputs to get total output size
        # )
        self.model = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_regions, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes * self.num_outputs)  # Multiply by num_outputs to get total output size
        )
        

    def forward(self, x):
        if len(x.shape) == 5:
            B, C, F, H, W = x.shape
            x = x.view(B, -1)
        x = self.model(x)
        return x.view(-1, self.num_outputs, 9)  # Reshape to (batch_size, num_outputs, num_classes)

def run_experiment(trainloader, validloader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for videos, _ in trainloader:
        input_size = videos[0].numel()
        break
    
    model = RainPrediction(input_size, 9)  # Changed to 9 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    def train(model, trainloader, validloader, criterion, optimizer, epochs=100):
        model.train()
        history = History()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            num_batches = 0
            
            for videos, labels in trainloader:
                videos, labels = videos.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs.view(-1, 9), labels.view(-1))  # Changed to 9 classes
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=2)
                correct += (predicted == labels).sum().item()
                total += labels.numel()
                num_batches += 1

            avg_loss = total_loss / num_batches
            train_acc = correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

            # Always perform validation if validloader is provided
            valid_loss, valid_acc = validate(model, validloader, criterion)
            print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}")
            history.update(avg_loss, train_acc, valid_loss, valid_acc)
            
            scheduler.step(valid_loss)
            
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                torch.save(model.state_dict(), 'best_model_simple_nn.pth')
        
        return history

    def validate(model, validloader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        num_batches = 0
        
        with torch.no_grad():
            for videos, labels in validloader:
                videos, labels = videos.to(device), labels.to(device)
                
                outputs = model(videos)
                loss = criterion(outputs.view(-1, 9), labels.view(-1))  # Changed to 9 classes
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=2)
                correct += (predicted == labels).sum().item()
                total += labels.numel()
                num_batches += 1

        avg_val_loss = total_loss / num_batches
        val_acc = correct / total
        return avg_val_loss, val_acc

    # def test(model, testloader):
    #     model.eval()
    #     correct, total = 0, 0
        
    #     with torch.no_grad():
    #         for videos, labels in testloader:
    #             videos, labels = videos.to(device), labels.to(device)
                
    #             outputs = model(videos)
    #             _, predicted = torch.max(outputs, dim=2)
    #             correct += (predicted == labels).sum().item()
    #             total += labels.numel()
        
    #     print(f"Test Accuracy: {correct / total:.4f}")

    # Train the model
    history = train(model, trainloader, validloader, criterion, optimizer, epochs)

    # Run evaluation
    # test(model, testloader)
    return model, history

# Remove the old training loop and replace with:
if __name__ == "__main__":
    # Load and prepare dataset
    data_path = "data/cluster_sst.npz"  # Adjust path as needed
    (train_videos, train_labels, train_start_months), (test_videos, test_labels, test_start_months) = download_and_prepare_dataset(data_path)
    
    # Split train into train and validation
    train_videos, valid_videos, train_labels, valid_labels = train_test_split(
        train_videos, train_labels, test_size=0.2, random_state=seed
    )
    
    # Create dataloaders
    trainloader = prepare_dataloader(train_videos, train_labels, "train")
    validloader = prepare_dataloader(valid_videos, valid_labels, "valid")
    # testloader = prepare_dataloader(test_videos, test_labels, "test")
    
    # Run experiment
    model, history = run_experiment(trainloader, validloader, validloader, epochs=100)
    
    # Plot the training history
    plotHistory(history)


