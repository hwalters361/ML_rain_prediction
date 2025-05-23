# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_3d.py
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
from positional_encodings.torch_encodings import *
import pandas as pd
import numpy as np  # Added numpy import for positional encoding
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# works with 9 classes

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

    def forward(self, x, mask_patch=-1):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # batch x heads x number of patches x number of patches
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # set column of matrix to negative number
        if mask_patch >=0:
            attn[:, :, mask_patch, :] = -torch.inf
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

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'mean', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # because we are using sea surface temperature data, the number of channels is 1.
        self.num_classes = num_classes # there should be 4 for sst data, quantized
        self.num_outputs = 9  # Number of independent classifications. There are 9 since we have 9 clusters.
        
        self.dim = dim  # Store dimension for positional encoding
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_width % patch_width == 0, f"Image Width {image_width} is not divisible by Patch Width: {patch_width}"
        assert image_height % patch_height == 0, f"Image Height {image_height} is not divisible by Patch Height: {patch_height}"
        
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size
        print(f'Number of Patches {num_patches} Patch dim {patch_dim}')

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        # [b, f, d, h, w] = [32, 1, 24, 96, 184]
        self.to_patch_embedding = nn.Sequential(
            # changed rearrange from :
            # 'b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)'
            # to 'b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)'
            # encodes the frame, height, and width as separate dimensions.
            Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_outputs * num_classes)  # Output (batch_size, 9 * 4)
        )
    '''
    Mehrnaz's positional encoder
    '''
    def positional_encoding(self, position, d_model, base=10000):
        """
        Create sinusoidal positional encodings using PyTorch.
        Args:
            position: Position indices (temporal or spatial) as a PyTorch tensor
            d_model: Dimension of the model
            base: Base for the sinusoidal functions
        """
        # Ensure position is a PyTorch tensor
        if not torch.is_tensor(position):
            position = torch.tensor(position)
        
        # Get the length of the position tensor
        seq_len = position.shape[0] if len(position.shape) > 0 else 1
        
        # Create a tensor for the positional encodings
        pos_vector = torch.zeros((seq_len, d_model))
        
        # Compute the positional encodings using PyTorch
        for i in range(d_model):
            if i % 2 == 0:
                # For even indices: sin(position/10000^(2i/d_model))
                pos_vector[:, i] = torch.sin(position / (base ** (2 * i / d_model)))
            else:
                # For odd indices: cos(position/10000^(2(i-1)/d_model))
                pos_vector[:, i] = torch.cos(position / (base ** (2 * (i - 1) / d_model)))
        
        return pos_vector

    def forward(self, video, start_month=None):
        x = self.to_patch_embedding(video)  # Shape: (batch_size, f, h, w, dim)
        b, f, h, w, c = x.shape  # `b` = batch_size, `f` = frames, `h` = height patches, `w` = width patches, `c` = embedding dim
        # print(f"Input shape after patch embedding: {x.shape}")
        
        # Split the embedding dimension into 4 parts
        xall = x.split(x.shape[-1] // 4, -1)
        # print(f"Split shapes: {[x.shape for x in xall]}")
        
        # Create positional encodings for each dimension
        # For temporal dimension (frames)
        f_positions = torch.arange(f, device=x.device)
        f_enc = self.positional_encoding(f_positions, xall[0].shape[-1])
        # Reshape to [1, F, 1, 1, C // 4] and repeat for batch size
        f_enc = f_enc.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        # print(f"f_enc shape before repeat: {f_enc.shape}")
        f_enc = f_enc.repeat(b, 1, h, w, 1)
        # print(f"f_enc shape after repeat: {f_enc.shape}")
        # print(f"xall[0] shape: {xall[0].shape}")
        
        # For height dimension
        h_positions = torch.arange(h, device=x.device)
        h_enc = self.positional_encoding(h_positions, xall[1].shape[-1])
        # Reshape to [1, 1, H, 1, C // 4] and repeat for batch size
        h_enc = h_enc.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        # print(f"h_enc shape before repeat: {h_enc.shape}")
        h_enc = h_enc.repeat(b, f, 1, w, 1)
        # print(f"h_enc shape after repeat: {h_enc.shape}")
        # print(f"xall[1] shape: {xall[1].shape}")
        
        # For width dimension
        w_positions = torch.arange(w, device=x.device)
        w_enc = self.positional_encoding(w_positions, xall[2].shape[-1])
        # Reshape to [1, 1, 1, W, C // 4] and repeat for batch size
        w_enc = w_enc.unsqueeze(0).unsqueeze(1).unsqueeze(1)
        # print(f"w_enc shape before repeat: {w_enc.shape}")
        w_enc = w_enc.repeat(b, f, h, 1, 1)
        # print(f"w_enc shape after repeat: {w_enc.shape}")
        # print(f"xall[2] shape: {xall[2].shape}")
        
        # For temporal dimension with start_month (if provided)
        if start_month is not None:
            # Convert start_month to tensor if it's not already
            if not torch.is_tensor(start_month):
                start_month = torch.tensor(start_month, device=x.device)
            
            # Calculate (position + start_month) % 12 for each sample in the batch
            f_positions_with_month = (f_positions.unsqueeze(0) + start_month.unsqueeze(1)) % 12
            
            # Apply positional encoding with start_month
            f_month_enc = self.positional_encoding(f_positions_with_month, xall[3].shape[-1])
            # Reshape to [b, F, 1, 1, C // 4]
            f_month_enc = f_month_enc.unsqueeze(2).unsqueeze(2)
            # print(f"f_month_enc shape before repeat: {f_month_enc.shape}")
            f_month_enc = f_month_enc.repeat(1, 1, h, w, 1)
            # print(f"f_month_enc shape after repeat: {f_month_enc.shape}")
        else:
            # If no start_month provided, use regular temporal encoding
            f_month_enc = f_enc  # Reuse the same temporal encoding
        
        # print(f"xall[3] shape: {xall[3].shape}")
        
        # Apply positional encodings to each part
        # Make sure the dimensions match before adding
        part0 = xall[0] + f_enc
        part1 = xall[1] + h_enc
        part2 = xall[2] + w_enc
        part3 = xall[3] + f_month_enc
        
        # Concatenate the parts back together
        x = torch.cat([part0, part1, part2, part3], -1)
        
        # Reshape for transformer processing
        x = rearrange(x, 'b f h w c -> b (f h w) c')
        
        x = self.dropout(x)
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

class SSTDataset(torch.utils.data.Dataset):
    """Dataset for loading SST data with starting month information."""
    def __init__(self, videos, labels, start_months=None):
        self.videos = videos
        self.labels = labels
        self.start_months = start_months  # Starting month for each sequence (0-11)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.start_months is not None:
            return self.videos[idx], self.labels[idx], self.start_months[idx]
        else:
            return self.videos[idx], self.labels[idx]

# Define patch sizes for padding calculation
IMAGE_PATCH_SIZE = (8, 8)  # Example patch size, adjust accordingly
FRAME_PATCH_SIZE = 8

def prepare_dataloader(videos, labels, loader_type="train", batch_size=32):
    """Creates a PyTorch DataLoader with preprocessed videos and starting month information."""
    dataset = SSTDataset(videos, labels)
    
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

def pad_video(video, patch_height, patch_width):
    """Pads the video spatial dimensions to be divisible by the patch size."""
    _, _, h, w = video.shape  # (Frames, Channels, Height, Width)
    
    pad_h = (patch_height - h % patch_height) % patch_height
    pad_w = (patch_width - w % patch_width) % patch_width

    if pad_h > 0 or pad_w > 0:
        padding = (0, pad_w, 0, pad_h)  # (Left, Right, Top, Bottom)
        video = F.pad(video, padding, mode='constant', value=0)  # Pad with zeros

    return video
    
# Define patch sizes for padding calculation
IMAGE_PATCH_SIZE = (8, 8)  # Example patch size, adjust accordingly
FRAME_PATCH_SIZE = 8

def download_and_prepare_dataset(data_path, image_patch_size=IMAGE_PATCH_SIZE):
    """Loads and preprocesses the dataset, including padding."""
    with np.load(data_path, allow_pickle=True) as data:
        train_videos = np.nan_to_num(data["train_images"])
        test_videos = np.nan_to_num(data["test_images"])

        train_labels = data["train_labels"]
        test_labels = data["test_labels"]

        train_start_months = data["train_start_months"]
        test_start_months = data["test_start_months"]
    
    # # Extract starting months from the data
    
    # Convert to PyTorch tensors
    train_videos = torch.tensor(train_videos, dtype=torch.float32)
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

    train_start_months = torch.tensor(train_start_months, dtype=torch.int)
    test_start_months = torch.tensor(test_start_months, dtype=torch.int)

    # return (train_videos, train_labels), (test_videos, test_labels)
    return (train_videos, train_labels, train_start_months), (test_videos, test_labels, test_start_months)
    # return (train_videos, train_labels), (test_videos, test_labels)

def analyze_patch_importance(model, dataloader, criterion, device):
    """
    Analyzes the importance of each patch by masking it out and measuring performance impact.
    
    Args:
        model: Trained ViT model
        dataloader: DataLoader containing validation/test data
        criterion: Loss function
        device: Device to run computations on
    
    Returns:
        patch_importance: Dictionary containing importance scores for each patch
    """
    model.eval()
    original_loss, original_acc = 0, 0
    total_samples = 0
    correct_predictions = 0
    
    # First, get baseline performance
    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs.view(-1, 4), labels.view(-1))
            original_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=2)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.numel()
    
    original_loss /= len(dataloader)
    original_acc = correct_predictions / total_samples
    
    # Get patch dimensions
    b, c, f, h, w = next(iter(dataloader))[0].shape
    patch_h, patch_w = model.to_patch_embedding[0].p1, model.to_patch_embedding[0].p2
    patch_f = model.to_patch_embedding[0].pf
    
    num_patches_h = h // patch_h
    num_patches_w = w // patch_w
    num_patches_f = f // patch_f
    
    # Initialize importance scores
    patch_importance = {
        'loss_impact': torch.zeros(num_patches_f, num_patches_h, num_patches_w),
        'acc_impact': torch.zeros(num_patches_f, num_patches_h, num_patches_w)
    }
    
    # Test each patch position
    for f_idx in range(num_patches_f):
        for h_idx in range(num_patches_h):
            for w_idx in range(num_patches_w):
                total_loss = 0
                correct_predictions = 0
                total_samples = 0
                
                with torch.no_grad():
                    for videos, labels in dataloader:
                        videos, labels = videos.to(device), labels.to(device)
                        
                        # Create a copy of the input
                        masked_videos = videos.clone()
                        
                        # Mask out the current patch
                        f_start = f_idx * patch_f
                        f_end = (f_idx + 1) * patch_f
                        h_start = h_idx * patch_h
                        h_end = (h_idx + 1) * patch_h
                        w_start = w_idx * patch_w
                        w_end = (w_idx + 1) * patch_w
                        
                        masked_videos[:, :, f_start:f_end, h_start:h_end, w_start:w_end] = 0
                        
                        # Get predictions with masked patch
                        outputs = model(masked_videos)
                        loss = criterion(outputs.view(-1, 4), labels.view(-1))
                        total_loss += loss.item()
                        
                        _, predicted = torch.max(outputs, dim=2)
                        correct_predictions += (predicted == labels).sum().item()
                        total_samples += labels.numel()
                
                # Calculate impact
                masked_loss = total_loss / len(dataloader)
                masked_acc = correct_predictions / total_samples
                
                # Store importance scores (higher values mean more important)
                patch_importance['loss_impact'][f_idx, h_idx, w_idx] = masked_loss - original_loss
                patch_importance['acc_impact'][f_idx, h_idx, w_idx] = original_acc - masked_acc
    
    return patch_importance

def visualize_patch_importance(patch_importance, save_path=None):
    """
    Visualizes patch importance scores.
    
    Args:
        patch_importance: Dictionary containing importance scores
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss impact
    sns.heatmap(patch_importance['loss_impact'].mean(dim=0), 
                ax=ax1, cmap='YlOrRd')
    ax1.set_title('Patch Importance (Loss Impact)')
    ax1.set_xlabel('Width Patch Index')
    ax1.set_ylabel('Height Patch Index')
    
    # Plot accuracy impact
    sns.heatmap(patch_importance['acc_impact'].mean(dim=0), 
                ax=ax2, cmap='YlOrRd')
    ax2.set_title('Patch Importance (Accuracy Impact)')
    ax2.set_xlabel('Width Patch Index')
    ax2.set_ylabel('Height Patch Index')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    """
    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, value, model):
        if self.best_value is None:
            self.best_value = value
            self.best_model_state = model.state_dict().copy()
        elif self.mode == 'min':
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
                self.best_model_state = model.state_dict().copy()
            else:
                self.counter += 1
        else:  # mode == 'max'
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
                self.best_model_state = model.state_dict().copy()
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
    def load_best_model(self, model):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model

def run_experiment(trainloader, validloader, testloader=None, start_months=None, epochs=100):
    # Get device
    from utils import get_device, to_device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create and move model to device
    model = ViT(
        image_size=(96,184),  
        image_patch_size=(8, 8),
        frames=24, 
        frame_patch_size=8,
        num_classes=4, 
        dim=128,
        depth=4,
        heads=4,
        dim_head=32,
        mlp_dim=256,
        channels=1,
    )
    model = to_device(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='min')

    def train(model, trainloader, validloader, criterion, optimizer, epochs=100):
        model.train()
        history = History()
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            num_batches = 0
            for videos, labels in trainloader:
                # Move data to device
                videos, labels = to_device([videos, labels])
                
                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs.view(-1, 4), labels.view(-1))
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=2)
                correct += (predicted == labels).sum().item()
                total += labels.numel()
                num_batches += 1

            # Average the loss over number of batches
            avg_loss = total_loss / num_batches
            train_acc = correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

            valid_loss, valid_acc = validate(model, validloader, criterion)
            history.update(avg_loss, train_acc, valid_loss, valid_acc)
            
            # Early stopping check
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load the best model before returning
        model = early_stopping.load_best_model(model)
        return history

    def validate(model, validloader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        num_batches = 0
        with torch.no_grad():
            for videos, labels in validloader:
                # Move data to device
                videos, labels = to_device([videos, labels])
                
                outputs = model(videos)
                loss = criterion(outputs.view(-1, 4), labels.view(-1))
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=2)
                correct += (predicted == labels).sum().item()
                total += labels.numel()
                num_batches += 1

        # Average the loss over number of batches
        avg_val_loss = total_loss / num_batches
        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss, val_acc

    def test(model, testloader):
        if testloader is None:
            print("No testing dataset provided")
            return
            
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for videos, labels in testloader:
                # Move data to device
                videos, labels = to_device([videos, labels])
                
                outputs = model(videos)
                _, predicted = torch.max(outputs, dim=2)
                correct += (predicted == labels).sum().item()
                total += labels.numel()
        
        print(f"Test Accuracy: {correct / total:.4f}")

    # Train the model
    history = train(model, trainloader, validloader, criterion, optimizer, epochs)

    # Run evaluation
    test(model, testloader)
    
    # After training, analyze patch importance
    # print("\nAnalyzing patch importance...")
    # patch_importance = analyze_patch_importance(model, validloader, criterion, device)
    # visualize_patch_importance(patch_importance, save_path="patch_importance.png")
    
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
