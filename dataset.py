"""
HDR Dataset Loader
Loads 5-bracket HDR image sets from the dataset structure:
- Each folder has 'raw' and 'edited' subfolders
- Raw folder contains sets of 5 consecutive bracketed images
- Edited folder contains merged HDR results named after first source image
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class HDRDataset(Dataset):
    """
    Dataset for loading 5-bracket HDR image sets.
    
    Structure expected:
    dataset_root/
        folder1/
            raw/
                IMG001.JPG, IMG002.JPG, IMG003.JPG, IMG004.JPG, IMG005.JPG  (set 1)
                IMG006.JPG, IMG007.JPG, IMG008.JPG, IMG009.JPG, IMG010.JPG  (set 2)
                ...
            edited/
                IMG001.jpg  (merged result for set 1)
                IMG006.jpg  (merged result for set 2)
                ...
        folder2/
            ...
    """
    
    def __init__(
        self,
        dataset_path: str,
        num_brackets: int = 5,
        crop_size: int = 384,
        is_training: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the HDR dataset.
        
        Args:
            dataset_path: Path to the dataset root directory
            num_brackets: Number of bracketed exposures (default: 5)
            crop_size: Size for random crop during training
            is_training: Whether this is for training (enables augmentation)
            max_samples: Maximum number of samples to load (for test mode)
        """
        self.dataset_path = Path(dataset_path)
        self.num_brackets = num_brackets
        self.crop_size = crop_size
        self.is_training = is_training
        
        # Collect all image sets
        self.samples = self._collect_samples()
        
        # Limit samples if max_samples is specified (test mode)
        if max_samples is not None and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        
        # Image transforms
        self.to_tensor = transforms.ToTensor()
        
        print(f"Loaded {len(self.samples)} HDR image sets")
    
    def _find_edited_folder(self, folder_path: Path) -> Optional[Path]:
        """Find the edited folder (handles case variations)"""
        for name in ['edited', 'Edited', 'EDITED']:
            edited_path = folder_path / name
            if edited_path.exists():
                return edited_path
        return None
    
    def _collect_samples(self) -> List[Tuple[List[Path], Path]]:
        """
        Collect all valid image sets from the dataset.
        
        Returns:
            List of tuples: (list of 5 raw image paths, edited image path)
        """
        samples = []
        
        # Iterate through all subfolders
        for folder in self.dataset_path.iterdir():
            if not folder.is_dir():
                continue
            
            raw_folder = folder / 'raw'
            edited_folder = self._find_edited_folder(folder)
            
            if not raw_folder.exists() or edited_folder is None:
                continue
            
            # Get all edited images (these define our target outputs)
            edited_images = {}
            for img_path in edited_folder.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Store with lowercase stem for matching
                    edited_images[img_path.stem.upper()] = img_path
            
            # Get all raw images and sort them by numeric value
            raw_images = []
            for img_path in raw_folder.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    raw_images.append(img_path)
            
            # Sort raw images by numeric part of filename
            def extract_number(filename: str) -> int:
                match = re.search(r'(\d+)', filename)
                return int(match.group(1)) if match else 0
            
            raw_images.sort(key=lambda x: extract_number(x.stem))
            
            # Match raw image sets to edited images
            # The first image name in a set of 5 should match an edited image
            i = 0
            while i < len(raw_images):
                first_img = raw_images[i]
                first_stem = first_img.stem.upper()
                
                # Check if this is the start of a valid set
                if first_stem in edited_images:
                    # Get the next 5 images
                    if i + self.num_brackets <= len(raw_images):
                        bracket_set = raw_images[i:i + self.num_brackets]
                        
                        # Verify the images are consecutive by number
                        numbers = [extract_number(img.stem) for img in bracket_set]
                        is_consecutive = all(
                            numbers[j+1] - numbers[j] == 1 
                            for j in range(len(numbers) - 1)
                        )
                        
                        if is_consecutive:
                            edited_path = edited_images[first_stem]
                            samples.append((bracket_set, edited_path))
                            i += self.num_brackets
                        else:
                            # Skip this image, numbers aren't consecutive
                            i += 1
                    else:
                        i += 1
                else:
                    i += 1
        
        return samples
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load an image from path"""
        return Image.open(path).convert('RGB')
    
    def _random_crop(self, images: List[Image.Image]) -> List[Image.Image]:
        """Apply the same random crop to all images"""
        # Get dimensions from first image
        w, h = images[0].size
        
        # Calculate crop position
        if w < self.crop_size or h < self.crop_size:
            # Resize if image is smaller than crop size
            scale = max(self.crop_size / w, self.crop_size / h) * 1.1
            new_w, new_h = int(w * scale), int(h * scale)
            images = [img.resize((new_w, new_h), Image.LANCZOS) for img in images]
            w, h = new_w, new_h
        
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        
        cropped = []
        for img in images:
            cropped.append(img.crop((x, y, x + self.crop_size, y + self.crop_size)))
        
        return cropped
    
    def _random_flip(self, images: List[Image.Image]) -> List[Image.Image]:
        """Apply random horizontal/vertical flips to all images"""
        if random.random() > 0.5:
            images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
        if random.random() > 0.5:
            images = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in images]
        return images
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (input_tensor, target_tensor)
            - input_tensor: [num_brackets * 3, H, W] concatenated LDR images
            - target_tensor: [3, H, W] HDR target image
        """
        raw_paths, edited_path = self.samples[idx]
        
        # Load all images
        raw_images = [self._load_image(p) for p in raw_paths]
        target_image = self._load_image(edited_path)
        
        # Combine all images for consistent augmentation
        all_images = raw_images + [target_image]
        
        if self.is_training:
            # Apply random crop to all images consistently
            all_images = self._random_crop(all_images)
            # Apply random flips
            all_images = self._random_flip(all_images)
        
        # Split back
        raw_images = all_images[:-1]
        target_image = all_images[-1]
        
        # Convert to tensors
        raw_tensors = [self.to_tensor(img) for img in raw_images]
        target_tensor = self.to_tensor(target_image)
        
        # Concatenate raw images along channel dimension
        # Shape: [num_brackets * 3, H, W] = [15, H, W]
        input_tensor = torch.cat(raw_tensors, dim=0)
        
        return input_tensor, target_tensor


def create_dataloaders(
    dataset_path: str,
    batch_size: int = 12,
    crop_size: int = 384,
    num_workers: int = 8,
    pin_memory: bool = True,
    val_split: float = 0.1,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        dataset_path: Path to dataset root
        batch_size: Batch size for training
        crop_size: Crop size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        val_split: Fraction of data to use for validation
        max_samples: Maximum samples (for test mode)
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create full dataset
    full_dataset = HDRDataset(
        dataset_path=dataset_path,
        crop_size=crop_size,
        is_training=True,
        max_samples=max_samples
    )
    
    # Split into train and validation
    total_samples = len(full_dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    from config import TRAIN_CONFIG
    
    dataset = HDRDataset(
        dataset_path=TRAIN_CONFIG.dataset_path,
        crop_size=TRAIN_CONFIG.crop_size,
        is_training=True,
        max_samples=10  # Just test with 10 samples
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        inputs, target = dataset[0]
        print(f"Input shape: {inputs.shape}")  # Should be [15, 384, 384]
        print(f"Target shape: {target.shape}")  # Should be [3, 384, 384]

