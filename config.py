"""
Configuration for HDR Training Pipeline
Optimized for RTX 3090 (24GB VRAM) + 128GB RAM
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    
    # Dataset paths
    dataset_path: str = "C:/Users/Administrator/Documents/HDR dataset"
    output_dir: str = "C:/Users/Administrator/Documents/HDR_Dataset_Scripts/output"
    checkpoint_dir: str = "C:/Users/Administrator/Documents/HDR_Dataset_Scripts/checkpoints"
    log_dir: str = "C:/Users/Administrator/Documents/HDR_Dataset_Scripts/logs"
    
    # Hardware settings (RTX 3090 + 128GB RAM)
    device: str = "cuda"
    num_workers: int = 8  # Plenty of RAM for more workers
    pin_memory: bool = True
    
    # Image settings
    num_brackets: int = 5  # Number of bracketed exposures
    crop_size: int = 512  # Balanced for 24GB with perceptual loss
    
    # Training parameters - Full mode
    batch_size: int = 4  # For 512x512 crops on 24GB
    epochs: int = 100
    learning_rate: float = 2e-4  # Increased for L1-only training
    weight_decay: float = 0  # Disabled for simpler training
    
    # Training parameters - Test mode
    test_samples: int = 50
    test_epochs: int = 10
    test_batch_size: int = 4
    
    # Loss weights (L1 only - most stable)
    l1_weight: float = 1.0
    perceptual_weight: float = 0.0  # Disabled - causes instability
    ssim_weight: float = 0.0  # Disabled
    
    # Optimizer settings
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # "cosine" or "step"
    lr_step_size: int = 30
    lr_gamma: float = 0.5
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision for faster training
    
    # Checkpointing
    save_every: int = 5  # Save checkpoint every N epochs
    validate_every: int = 1  # Validate every N epochs
    
    # Logging
    log_every: int = 10  # Log every N batches
    
    def __post_init__(self):
        """Create output directories if they don't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class InferenceConfig:
    """Inference configuration parameters"""
    
    # Paths
    checkpoint_path: Optional[str] = None
    
    # Hardware settings
    device: str = "cuda"
    
    # Processing settings
    num_brackets: int = 5
    tile_size: int = 512  # Tile size for sliding window inference
    tile_overlap: int = 64  # Overlap between tiles
    batch_size: int = 1  # Process one image set at a time for full resolution
    
    # Output settings
    output_format: str = "jpg"
    jpeg_quality: int = 95


# Default configurations
TRAIN_CONFIG = TrainingConfig()
INFERENCE_CONFIG = InferenceConfig()

