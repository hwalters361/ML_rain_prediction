import torch
from torch import nn

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

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
        self.norm = nn.LayerNorm(dim)
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
        return self.norm(x)

class FactorizedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        # x.shape = torch.Size([32, 3, 276, 128])
        b, f, n, _ = x.shape
        for spatial_attn, temporal_attn, ff in self.layers:
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = spatial_attn(x) + x
            x = rearrange(x, '(b f) n d -> (b n) f d', b=b, f=f)
            x = temporal_attn(x) + x
            x = ff(x) + x
            x = rearrange(x, '(b n) f d -> b f n d', b=b, n=n)

        return self.norm(x)

class ViViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 1,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        variant = 'factorized_encoder',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 9

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_width % patch_width == 0, f"Image Width {image_width} is not divisible by Patch Width: {patch_width}"
        assert image_height % patch_height == 0, f"Image Height {image_height} is not divisible by Patch Height: {patch_height}"
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        assert variant in ('factorized_encoder', 'factorized_self_attention'), f'variant = {variant} is not implemented'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size
        print(f'Number of image patches {num_image_patches}\n Number of frame patches {num_frame_patches}\n Patch dim {patch_dim}')
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        
        # Should make this a fixed positional embedding.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # learned CLS token
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        if variant == 'factorized_encoder':
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
            self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
            self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)
        elif variant == 'factorized_self_attention':
            assert spatial_depth == temporal_depth, 'Spatial and temporal depth must be the same for factorized self-attention'
            self.factorized_transformer = FactorizedTransformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, self.num_outputs*num_classes)
        self.variant = variant

    def forward(self, video):
        x = self.to_patch_embedding(video)
        # print(x.shape)
        b, f, n, _ = x.shape

        x = x + self.pos_embedding[:, :f, :n]

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
            x = torch.cat((spatial_cls_tokens, x), dim = 2)

        x = self.dropout(x)

        if self.variant == 'factorized_encoder':
            x = rearrange(x, 'b f n d -> (b f) n d')

            # attend across space

            x = self.spatial_transformer(x)
            x = rearrange(x, '(b f) n d -> b f n d', b = b)

            # excise out the spatial cls tokens or average pool for temporal attention

            x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

            # append temporal CLS tokens

            if exists(self.temporal_cls_token):
                temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

                x = torch.cat((temporal_cls_tokens, x), dim = 1)
            

            # attend across time

            x = self.temporal_transformer(x)

            # excise out temporal cls token or average pool

            x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        elif self.variant == 'factorized_self_attention':
            x = self.factorized_transformer(x)
            x = x[:, 0, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b d', 'mean')

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x.view(b, self.num_outputs, self.num_classes)
    
#### End of given code. Start of training loop:

from torch.utils.data import DataLoader
import torch.optim as optim
import torch

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


def run_experiment(trainloader, validloader, testloader):
    # Define model
    model = ViViT(
        image_size=(96, 184),  
        image_patch_size=(8, 8),
        frames=24, 
        frame_patch_size=8,
        num_classes=4, 
        dim=128, 
        spatial_depth=4,  # Adjust as necessary
        temporal_depth=4, # Adjust as necessary
        heads=4, 
        dim_head=32, 
        mlp_dim=256, 
        channels=1,
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    best_model_wts = None
    save_path = "models/best_vivit_model.pth"  # Path to save the model

    def train(model, trainloader, validloader, criterion, optimizer, epochs=100):
        model.train()
        history = History()
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            
            for videos, labels in trainloader:
                optimizer.zero_grad()
                outputs = model(videos)  # (batch, 9, 4)
                
                # Reshape outputs and labels for CrossEntropyLoss
                # outputs =   # Flatten (batch_size * 9, 4)
                # labels =  # Flatten (batch_size * 9,)
                
                loss = criterion(outputs.view(-1, 4), labels.view(-1) )  # CrossEntropyLoss
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=2)#dim=1)  # Get class predictions
                correct += (predicted == labels).sum().item()
                total += labels.numel()  # Count all predictions (batch_size * 9)

            train_acc = correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {train_acc:.4f}")

            valid_loss, valid_acc = validate(model, validloader, criterion)
            
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                save_model(model, path=save_path)
                print(f"Epoch {epoch}: New best model saved with validation loss: {valid_loss}")

            history.update(total_loss, train_acc, valid_loss, valid_acc)
        
        return history


    def validate(model, validloader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for videos, labels in validloader:
                outputs = model(videos)
                
                # Reshape outputs and labels for CrossEntropyLoss
                # outputs = outputs.view(-1, 4)  # Flatten (batch_size * 9, 4)
                # labels = labels.view(-1)  # Flatten (batch_size * 9,)
                
                loss = criterion(outputs.view(-1, 4), labels.view(-1))
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=2)#dim=1)  # Get class predictions
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
                outputs = model(videos)
                _, predicted = torch.max(outputs, dim=2)#dim=1)  # Get class predictions
                correct += (predicted == labels).sum().item()
                total += labels.numel()  # Count all predictions (batch_size * 9)
        
        print(f"Test Accuracy: {correct / total:.4f}")

    # Train the model
    history = train(model, trainloader, validloader, criterion, optimizer, epochs=100)

    # Run evaluation
    test(model, testloader)
    return model, history


def save_model(model, path="savedmodels/vivit_model.pth"):
    """Saves the trained PyTorch model to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path="vit_model.pth"):
    """Loads the model's saved state_dict."""
    model.load_state_dict(torch.load(path))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {path}")
