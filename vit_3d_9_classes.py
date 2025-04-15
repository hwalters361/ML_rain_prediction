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
    def positional_encoding(self, position, d_model, base=10000, start_month=None):
        """
        Create sinusoidal positional encodings optimized for temporal data with starting month information.
        Args:
            position: Position indices (temporal or spatial)
            d_model: Dimension of the model
            base: Base for the sinusoidal functions
            start_month: Starting month of the sequence (0-11)
        """
        # Initialize a zero vector
        if isinstance(position, (np.ndarray, pd.Series, list)):
            pos_vector = np.zeros((len(position), d_model))
        else:
            pos_vector = np.zeros((1, d_model))
        
        # Convert start_month to numpy array if it's a tensor
        if torch.is_tensor(start_month):
            start_month = start_month.cpu().numpy()
        
        # Ensure start_month is a scalar
        if isinstance(start_month, (np.ndarray, list)):
            start_month = start_month[0]  # Take the first value if it's an array
        
        # Calculate phase if start_month is provided
        phase = (start_month / 12.0) * 2 * np.pi if start_month is not None else 0
        
        # Compute the positional encodings
        for i in range(d_model):
            if i % 2 == 0:
                # For even indices: sin(position/10000^(2i/d_model) + phase)
                pos_vector[:, i] = np.sin(position / (base ** (2 * i / d_model)) + phase)
            else:
                # For odd indices: cos(position/10000^(2(i-1)/d_model) + phase)
                pos_vector[:, i] = np.cos(position / (base ** (2 * (i - 1) / d_model)) + phase)
        
        return torch.from_numpy(pos_vector).float()

    def forward(self, video, start_month=None):
        x = self.to_patch_embedding(video)  # Shape: (batch_size, f, h, w, dim)
        b, f, h, w, c = x.shape  # `b` = batch_size, `f` = frames, `h` = height patches, `w` = width patches, `c` = embedding dim
        
        # Create position indices for temporal dimension (frames)
        f_positions = np.arange(f)
        
        # Handle start_month for each sample in the batch
        if start_month is not None:
            # Create a list to store encodings for each sample
            f_pos_enc_list = []
            for i in range(b):
                sample_start_month = start_month[i].item() if torch.is_tensor(start_month) else start_month[i]
                f_pos_enc = self.positional_encoding(f_positions, c, start_month=sample_start_month)
                f_pos_enc_list.append(f_pos_enc)
            f_pos_enc = torch.stack(f_pos_enc_list)  # Shape: (batch_size, f, c)
        else:
            f_pos_enc = self.positional_encoding(f_positions, c)
            f_pos_enc = f_pos_enc.unsqueeze(0).repeat(b, 1, 1)  # Shape: (batch_size, f, c)
        
        # Create position indices for spatial dimensions (height and width)
        h_positions = np.arange(h)
        w_positions = np.arange(w)
        h_pos_enc = self.positional_encoding(h_positions, c)  # Shape: (h, c)
        w_pos_enc = self.positional_encoding(w_positions, c)  # Shape: (w, c)
        
        # Move encodings to the correct device
        f_pos_enc = f_pos_enc.to(x.device)
        h_pos_enc = h_pos_enc.to(x.device)
        w_pos_enc = w_pos_enc.to(x.device)
        
        # Create a working copy to avoid modifying x in-place
        x_encoded = x.clone()
        
        # Apply temporal positional encoding
        for i in range(f):
            x_encoded[:, i, :, :, :] += f_pos_enc[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        # Apply spatial positional encodings
        for j in range(h):
            x_encoded[:, :, j, :, :] += h_pos_enc[j].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        for k in range(w):
            x_encoded[:, :, :, k, :] += w_pos_enc[k].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Reshape for transformer processing
        x = rearrange(x_encoded, 'b f h w c -> b (f h w) c')
        
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
    def __init__(self, videos, labels, start_months):
        self.videos = videos
        self.labels = labels
        self.start_months = start_months  # Starting month for each sequence (0-11)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.videos[idx], self.labels[idx], self.start_months[idx]
# Define patch sizes for padding calculation
IMAGE_PATCH_SIZE = (8, 8)  # Example patch size, adjust accordingly
FRAME_PATCH_SIZE = 8

def prepare_dataloader(videos, labels, start_months, loader_type="train", batch_size=32):
    """Creates a PyTorch DataLoader with preprocessed videos and starting month information."""
    dataset = SSTDataset(videos, labels, start_months)
    
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
    
    # Extract starting months from the data
    # Assuming the data is organized chronologically, we can calculate the starting month
    # from the index in the original time series
    train_start_months = np.array([(i % 12) for i in range(len(train_videos))])
    test_start_months = np.array([(i % 12) for i in range(len(test_videos))])
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

    # return (train_videos, train_labels), (valid_videos, valid_labels), (test_videos, test_labels)
    return (train_videos, train_labels, train_start_months), (test_videos, test_labels, test_start_months)


def run_experiment(trainloader, validloader, testloader=None):
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
        try:
            for epoch in range(epochs):
                total_loss, correct, total = 0, 0, 0
                num_batches = 0
                
                for videos, labels, start_months in trainloader:
                    optimizer.zero_grad()
                    outputs = model(videos, start_month=start_months)  # Pass start_month to model
                    loss = criterion(outputs.view(-1, 4), labels.view(-1))
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, dim=2)
                    correct += (predicted == labels).sum().item()
                    total += labels.numel()
                    num_batches += 1
                
                # Average the loss over number of batches
                avg_train_loss = total_loss / num_batches

                train_acc = correct / total
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f} {num_batches}, Accuracy: {train_acc:.4f}")

                valid_loss, valid_acc = validate(model, validloader, criterion)
                history.update(avg_train_loss, train_acc, valid_loss, valid_acc)
            
            return history
        except KeyboardInterrupt:
            print("Training interrupted. Saving the model and history...")
            return model, history
            


    def validate(model, validloader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        num_batches = 0
        
        with torch.no_grad():
            for videos, labels, _ in validloader:
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
        if testloader == None:
            print("No testing dataset provided")
            return
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for videos, labels, _ in testloader:
                outputs = model(videos)
                _, predicted = torch.max(outputs, dim=2)
                correct += (predicted == labels).sum().item()
                total += labels.numel()
        
        print(f"Test Accuracy: {correct / total:.4f}")

    # Train the model
    history = train(model, trainloader, validloader, criterion, optimizer, epochs=100)

    # Run evaluation
    # test(model, testloader)
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
