import matplotlib.pyplot as plt
import torch

def visualize_patches(video_sample, image_patch_size, frame_patch_size):
    """
    Visualizes how a video sample is split into patches.
    Args:
        video_sample: A single video sample of shape (C, F, H, W)
        image_patch_size: Tuple of (height, width) for spatial patches
        frame_patch_size: Number of frames per temporal patch
    """
    # Get dimensions
    c, f, h, w = video_sample.shape
    patch_h, patch_w = image_patch_size
    
    # Calculate number of patches in each dimension
    n_patches_h = h // patch_h
    n_patches_w = w // patch_w
    n_patches_f = f // frame_patch_size
    
    # Create a figure with subplots for each spatial patch in the first frame
    fig, axes = plt.subplots(n_patches_h, n_patches_w, figsize=(15, 15))
    fig.suptitle(f'Patches for first frame (Patch size: {patch_h}x{patch_w}, Frame patch size: {frame_patch_size})')
    
    # Plot each patch
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Extract patch
            patch = video_sample[0, 0, 
                               i*patch_h:(i+1)*patch_h, 
                               j*patch_w:(j+1)*patch_w]
            
            # Plot patch
            axes[i, j].imshow(patch, cmap='viridis')
            axes[i, j].set_title(f'({i},{j})')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print patch information
    print(f"\nPatch Information:")
    print(f"Total number of patches: {n_patches_h * n_patches_w * n_patches_f}")
    print(f"Spatial patches per frame: {n_patches_h * n_patches_w}")
    print(f"Temporal patches: {n_patches_f}")
    print(f"Patch dimensions: {patch_h}x{patch_w}x{frame_patch_size}")

def visualize_patches_from_dataloader(dataloader, image_patch_size, frame_patch_size):
    """
    Visualizes patches from the first sample in a dataloader.
    Args:
        dataloader: PyTorch DataLoader containing video samples
        image_patch_size: Tuple of (height, width) for spatial patches
        frame_patch_size: Number of frames per temporal patch
    """
    for videos, labels in dataloader:
        if len(videos) > 0:
            sample_video = videos[0]  # Get first video sample
            visualize_patches(sample_video, image_patch_size, frame_patch_size)
            break
