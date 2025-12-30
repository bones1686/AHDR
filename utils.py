"""
Utility functions for HDR training pipeline
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    config: Optional[Dict] = None
):
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        path: Save path
        scheduler: Learning rate scheduler (optional)
        scaler: GradScaler for AMP (optional)
        config: Training configuration (optional)
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat()
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    if config is not None:
        checkpoint["config"] = config
    
    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint
    
    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        scaler: GradScaler to load state into (optional)
        device: Device to map tensors to (optional)
    
    Returns:
        Checkpoint dictionary with metadata
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.6f}"


class MetricTracker:
    """Track multiple metrics during training"""
    
    def __init__(self, metrics: list):
        self.metrics = {name: AverageMeter(name) for name in metrics}
    
    def update(self, values: Dict[str, float], n: int = 1):
        for name, value in values.items():
            if name in self.metrics:
                self.metrics[name].update(value, n)
    
    def reset(self):
        for meter in self.metrics.values():
            meter.reset()
    
    def get_averages(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.metrics.items()}
    
    def __str__(self):
        return " | ".join(str(meter) for meter in self.metrics.values())


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio
    
    Args:
        pred: Predicted image
        target: Target image
        max_val: Maximum pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0

