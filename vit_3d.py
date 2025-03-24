# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_3d.py
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    

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

def run_experiment(trainset, validset, testset):
    batch_size = 8  # Adjust as needed
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Define model
    model = ViT(
        image_size=224,  # Adjust based on input shape
        image_patch_size=16,
        frames=32,  # Adjust based on your dataset
        frame_patch_size=4,
        num_classes=10,  # Match your classification task
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024
    ).cuda()  # Move to GPU if available


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def train(model, trainloader, validloader, criterion, optimizer, epochs=10):
        '''
        training loop for the 3d vit, also returns history object with training history
        '''
        model.train()
        history = History()
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            
            for videos, labels in trainloader:
                videos, labels = videos.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

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
                videos, labels = videos.cuda(), labels.cuda()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {total_loss}")
        return total_loss, val_acc

    def test(model, testloader):
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for videos, labels in testloader:
                videos, labels = videos.cuda(), labels.cuda()
                outputs = model(videos)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        print(f"Test Accuracy: {correct / total:.4f}")

    # Train the model
    train(model, trainloader, validloader, criterion, optimizer, epochs=10)

    # Run evaluation
    test(model, testloader)


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
