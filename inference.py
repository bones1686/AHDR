"""
HDR Batch Processing Script

Process folders with 5-bracket HDR images using trained AHDR-Net model.

Usage:
    # Process with global color matching (recommended)
    python inference.py --input "path/to/raw" --output "output" --checkpoint "best.pth"
    
    # Process at reduced resolution (faster, no tile artifacts)
    python inference.py --input "path/to/raw" --output "output" --checkpoint "best.pth" --mode resize --scale 0.5
    
    # Process with tiles (for very large images)
    python inference.py --input "path/to/raw" --output "output" --checkpoint "best.pth" --mode tile
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional
import math

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

from config import InferenceConfig
from model import AHDRNet, AHDRNetLite, create_model
from utils import get_device


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch process HDR images")
    
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input folder containing 5-bracket image sets"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output folder for processed images"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to trained model checkpoint"
    )
    
    # Model settings
    parser.add_argument(
        "--model", type=str, default="full",
        choices=["full", "lite"],
        help="Model type (should match training)"
    )
    
    # Processing mode
    parser.add_argument(
        "--mode", type=str, default="resize",
        choices=["resize", "tile", "full"],
        help="Processing mode: 'resize' (recommended), 'tile', or 'full'"
    )
    
    # Resize mode settings
    parser.add_argument(
        "--max-size", type=int, default=1920,
        help="Max dimension for resize mode (default: 1920)"
    )
    
    # Tile mode settings
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="Tile size for tile mode (default: 512)"
    )
    parser.add_argument(
        "--tile-overlap", type=int, default=128,
        help="Overlap between tiles (default: 128, increased for better blending)"
    )
    
    # Output settings
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality for output (default: 95)"
    )
    
    # Hardware
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    return parser.parse_args()


def extract_number(filename: str) -> int:
    """Extract the numeric part from a filename like 'ILN07640' -> 7640"""
    import re
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def find_image_sets(folder: Path, num_brackets: int = 5) -> List[Tuple[List[Path], str]]:
    """
    Find sets of bracketed images in a folder.
    Groups images by consecutive numbers (e.g., 07640-07644 is one set).
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = []
    
    for f in folder.iterdir():
        if f.suffix in image_extensions:
            images.append(f)
    
    # Sort by the numeric part of the filename
    images.sort(key=lambda x: extract_number(x.stem))
    
    sets = []
    i = 0
    while i <= len(images) - num_brackets:
        # Check if next num_brackets images are consecutive
        bracket_set = images[i:i + num_brackets]
        numbers = [extract_number(img.stem) for img in bracket_set]
        
        # Check if numbers are consecutive (each differs by 1)
        is_consecutive = all(
            numbers[j+1] - numbers[j] == 1 
            for j in range(len(numbers) - 1)
        )
        
        if is_consecutive:
            output_name = bracket_set[0].stem + ".jpg"
            sets.append((bracket_set, output_name))
            i += num_brackets  # Move to next set
        else:
            i += 1  # Try starting from next image
    
    return sets


def load_images(paths: List[Path]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load a set of images and stack them.
    Handles slight size differences by resizing all to match the first image.
    
    Returns:
        Tuple of (tensor [1, num_brackets*3, H, W], original_size (W, H))
    """
    to_tensor = transforms.ToTensor()
    
    images = []
    target_size = None
    
    for path in paths:
        img = Image.open(path).convert('RGB')
        if target_size is None:
            target_size = img.size  # (W, H)
        elif img.size != target_size:
            # Resize to match first image if sizes differ
            img = img.resize(target_size, Image.LANCZOS)
        tensor = to_tensor(img)
        images.append(tensor)
    
    stacked = torch.cat(images, dim=0)
    return stacked.unsqueeze(0), target_size


def process_resize_mode(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_size: Tuple[int, int],
    max_size: int = 1920,
    device: torch.device = None
) -> torch.Tensor:
    """
    Process by resizing to manageable size, then resize output back.
    This avoids tile artifacts entirely.
    """
    if device is None:
        device = next(model.parameters()).device
    
    _, C, H, W = input_tensor.shape
    orig_W, orig_H = original_size
    
    # Calculate resize scale
    max_dim = max(H, W)
    if max_dim > max_size:
        scale = max_size / max_dim
        new_H = int(H * scale)
        new_W = int(W * scale)
        # Make divisible by 8 for network
        new_H = (new_H // 8) * 8
        new_W = (new_W // 8) * 8
    else:
        new_H = (H // 8) * 8
        new_W = (W // 8) * 8
        scale = 1.0
    
    # Resize input
    input_resized = F.interpolate(
        input_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False
    )
    
    # Process
    input_resized = input_resized.to(device)
    with torch.no_grad():
        output = model(input_resized)
    output = output.cpu()
    
    # Resize back to original
    output = F.interpolate(
        output, size=(orig_H, orig_W), mode='bilinear', align_corners=False
    )
    
    return output


def process_full(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device = None
) -> torch.Tensor:
    """Process full image at once (requires lots of VRAM)"""
    if device is None:
        device = next(model.parameters()).device
    
    _, C, H, W = input_tensor.shape
    
    # Pad to be divisible by 8
    pad_H = (8 - H % 8) % 8
    pad_W = (8 - W % 8) % 8
    
    if pad_H > 0 or pad_W > 0:
        input_tensor = F.pad(input_tensor, (0, pad_W, 0, pad_H), mode='reflect')
    
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    output = output.cpu()
    
    # Remove padding
    if pad_H > 0 or pad_W > 0:
        output = output[:, :, :H, :W]
    
    return output


def create_blend_weights(tile_size: int, overlap: int) -> torch.Tensor:
    """Create smooth blending weights using raised cosine window"""
    weights = torch.ones(1, 1, tile_size, tile_size)
    
    if overlap > 0:
        # Create 1D raised cosine
        ramp = torch.linspace(0, 1, overlap)
        ramp = 0.5 - 0.5 * torch.cos(ramp * math.pi)
        
        # Apply to edges
        for i in range(overlap):
            weights[:, :, i, :] *= ramp[i]
            weights[:, :, tile_size - 1 - i, :] *= ramp[i]
            weights[:, :, :, i] *= ramp[i]
            weights[:, :, :, tile_size - 1 - i] *= ramp[i]
    
    return weights


def process_tiled_with_color_matching(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    tile_size: int = 512,
    tile_overlap: int = 128,
    device: torch.device = None
) -> torch.Tensor:
    """
    Process image using tiled inference with global color harmonization.
    """
    if device is None:
        device = next(model.parameters()).device
    
    _, C, H, W = input_tensor.shape
    
    # If image is smaller than tile size, process directly
    if H <= tile_size and W <= tile_size:
        return process_full(model, input_tensor, device)
    
    # First, get global reference by processing a resized version
    scale = min(tile_size / H, tile_size / W, 1.0)
    ref_H = max(int(H * scale), 64)
    ref_W = max(int(W * scale), 64)
    ref_H = (ref_H // 8) * 8
    ref_W = (ref_W // 8) * 8
    
    input_ref = F.interpolate(input_tensor, size=(ref_H, ref_W), mode='bilinear', align_corners=False)
    input_ref = input_ref.to(device)
    with torch.no_grad():
        global_ref = model(input_ref)
    global_ref = global_ref.cpu()
    
    # Get global color statistics
    global_mean = global_ref.mean(dim=(2, 3), keepdim=True)
    global_std = global_ref.std(dim=(2, 3), keepdim=True) + 1e-6
    
    # Calculate stride
    stride = tile_size - tile_overlap
    
    # Calculate number of tiles
    n_tiles_h = math.ceil((H - tile_overlap) / stride)
    n_tiles_w = math.ceil((W - tile_overlap) / stride)
    
    # Pad input if necessary
    pad_h = max(0, n_tiles_h * stride + tile_overlap - H)
    pad_w = max(0, n_tiles_w * stride + tile_overlap - W)
    
    if pad_h > 0 or pad_w > 0:
        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    # Initialize output and weight tensors
    _, _, H_pad, W_pad = input_tensor.shape
    output = torch.zeros(1, 3, H_pad, W_pad)
    weight = torch.zeros(1, 1, H_pad, W_pad)
    
    # Create blending weights
    blend_weights = create_blend_weights(tile_size, tile_overlap)
    
    # Process tiles
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            # Calculate tile coordinates
            y_start = i * stride
            x_start = j * stride
            y_end = min(y_start + tile_size, H_pad)
            x_end = min(x_start + tile_size, W_pad)
            
            # Adjust for boundary tiles
            y_start = y_end - tile_size
            x_start = x_end - tile_size
            
            # Extract and process tile
            tile = input_tensor[:, :, y_start:y_end, x_start:x_end].to(device)
            
            with torch.no_grad():
                tile_output = model(tile)
            tile_output = tile_output.cpu()
            
            # Apply color matching to tile
            tile_mean = tile_output.mean(dim=(2, 3), keepdim=True)
            tile_std = tile_output.std(dim=(2, 3), keepdim=True) + 1e-6
            
            # Normalize tile to global statistics (partial matching for smooth transition)
            blend_factor = 0.5  # How much to blend toward global color
            tile_output = (tile_output - tile_mean) / tile_std
            target_mean = tile_mean * (1 - blend_factor) + global_mean * blend_factor
            target_std = tile_std * (1 - blend_factor) + global_std * blend_factor
            tile_output = tile_output * target_std + target_mean
            tile_output = tile_output.clamp(0, 1)
            
            # Add to output with blending weights
            tile_weight = blend_weights
            output[:, :, y_start:y_end, x_start:x_end] += tile_output * tile_weight
            weight[:, :, y_start:y_end, x_start:x_end] += tile_weight
    
    # Normalize by weights
    output = output / (weight + 1e-8)
    
    # Remove padding
    output = output[:, :, :H, :W]
    
    return output


def save_image(tensor: torch.Tensor, path: Path, quality: int = 95, debug: bool = False):
    """Save tensor as JPEG image"""
    img = tensor.squeeze(0)
    
    if debug:
        print(f"  Output stats - min: {img.min():.4f}, max: {img.max():.4f}, mean: {img.mean():.4f}")
    
    img = img.clamp(0, 1)
    img = (img * 255).byte()
    img = img.permute(1, 2, 0).numpy()
    
    pil_img = Image.fromarray(img)
    pil_img.save(path, "JPEG", quality=quality)


def main():
    args = parse_args()
    
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_type = checkpoint.get('config', {}).get('model_type', args.model)
    
    model = create_model(model_type, num_brackets=5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model type: {model_type}")
    print(f"Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'config' in checkpoint:
        print(f"Training config: {checkpoint['config']}")
    print(f"Processing mode: {args.mode}")
    
    # Quick sanity check with random input
    test_input = torch.randn(1, 15, 256, 256).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"Model sanity check - input shape: {test_input.shape}, output shape: {test_output.shape}")
    print(f"Model output range: min={test_output.min():.4f}, max={test_output.max():.4f}, mean={test_output.mean():.4f}")
    
    # Find image sets
    print(f"\nScanning input folder: {input_folder}")
    image_sets = find_image_sets(input_folder, num_brackets=5)
    print(f"Found {len(image_sets)} image sets to process")
    
    if len(image_sets) == 0:
        print("No valid image sets found.")
        return
    
    # Process each set
    for image_paths, output_name in tqdm(image_sets, desc="Processing"):
        try:
            input_tensor, original_size = load_images(image_paths)
            
            if args.mode == "resize":
                output_tensor = process_resize_mode(
                    model, input_tensor, original_size,
                    max_size=args.max_size, device=device
                )
            elif args.mode == "tile":
                output_tensor = process_tiled_with_color_matching(
                    model, input_tensor,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                    device=device
                )
            else:  # full
                output_tensor = process_full(model, input_tensor, device)
            
            output_path = output_folder / output_name
            save_image(output_tensor, output_path, quality=args.quality, debug=True)
            
        except Exception as e:
            print(f"\nError processing {image_paths[0].stem}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessing complete! Output saved to: {output_folder}")


if __name__ == "__main__":
    main()
