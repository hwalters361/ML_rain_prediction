import torch

def get_device():
    """
    Returns the appropriate device (CUDA if available, CPU otherwise)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data, device=None):
    """
    Moves data to the specified device (or automatically determined device if none specified)
    Args:
        data: The data to move (tensor, model, or list/dict of tensors)
        device: Optional device to move to. If None, will use get_device()
    """
    if device is None:
        device = get_device()
    
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.nn.Module):
        return data.to(device)
    else:
        return data.to(device) 
    

import torch
import matplotlib.pyplot as plt
from PIL import Image
import io

def tensor_to_gif_save(tensor, gif_path="output.gif", duration=200):
    """
    Converts a [1, T, H, W] tensor to a grayscale GIF.
    - tensor: torch.Tensor of shape [1, T, H, W]
    - gif_path: output file path
    - duration: duration per frame in milliseconds
    """
    tensor = tensor.squeeze(0)  # Shape: [24, 96, 184]
    frames = []

    # Normalize to [0, 255] for image display
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = (tensor * 255).byte()

    for frame_tensor in tensor:
        plt.imshow(frame_tensor.numpy(), cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        frame = Image.open(buf).convert('L')
        frames.append(frame)

    # Save as GIF
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    print(f"GIF saved to {gif_path}")


import torch
from PIL import Image, ImageDraw, ImageFont
import io
from IPython.display import display, Image as IPyImage

def display_tensor_as_gif(tensor, duration=200, scale=2):
    """
    Displays a [1, T, H, W] tensor as a grayscale animated GIF in a Jupyter cell.
    - Flips each frame vertically for correct orientation.
    - Enlarges each frame by the given scale factor.
    - Adds frame number labels in the bottom-left corner.
    """
    tensor = tensor.squeeze(0)  # Shape: [T, H, W]
    frames = []

    # Normalize to [0, 255]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = (tensor * 255).byte()

    for i, frame_tensor in enumerate(tensor):
        # Convert to PIL Image and flip
        img = Image.fromarray(frame_tensor.numpy(), mode='L')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Resize (scale up)
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

        # Draw label
        draw = ImageDraw.Draw(img)
        label = f"Frame {i}"
        draw.text((10, img.height - 20), label, fill=255)  # Bottom-left corner

        frames.append(img)

    # Save GIF to an in-memory buffer
    buf = io.BytesIO()
    frames[0].save(buf, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    buf.seek(0)

    return IPyImage(data=buf.read(), format='gif')
