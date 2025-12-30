"""
HDR Training Script

Train AHDR-Net model for merging 5-bracket HDR photos.

Usage:
    # Test mode (quick validation with 50 samples, 10 epochs)
    python train.py --test --data "path/to/HDR dataset"
    
    # Full training
    python train.py --data "path/to/HDR dataset" --epochs 100
    
    # Resume training
    python train.py --data "path/to/HDR dataset" --resume checkpoints/latest.pth
"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import TrainingConfig
from dataset import HDRDataset, create_dataloaders
from model import AHDRNet, AHDRNetLite, create_model
from losses import CombinedLoss
from utils import (
    set_seed, get_device, count_parameters,
    save_checkpoint, load_checkpoint,
    AverageMeter, MetricTracker, calculate_psnr,
    format_time, get_lr
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train AHDR-Net for HDR merging")
    
    # Dataset
    parser.add_argument(
        "--data", type=str,
        default="C:/Users/Administrator/Documents/HDR dataset",
        help="Path to dataset directory"
    )
    
    # Training mode
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: train with 50 samples for 10 epochs"
    )
    
    # Model
    parser.add_argument(
        "--model", type=str, default="full",
        choices=["full", "lite"],
        help="Model type: 'full' for AHDRNet, 'lite' for AHDRNetLite"
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--crop-size", type=int, default=None, help="Crop size")
    
    # Resume training
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Output
    parser.add_argument(
        "--output", type=str, default="output",
        help="Output directory for checkpoints and logs"
    )
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers")
    
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = True,
    log_every: int = 10
) -> dict:
    """
    Train for one epoch
    
    Returns:
        Dictionary of average metrics for the epoch
    """
    model.train()
    
    metrics = MetricTracker(['loss', 'l1', 'perceptual', 'psnr'])
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss, loss_components = criterion(outputs, targets)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate PSNR
        with torch.no_grad():
            psnr = calculate_psnr(outputs, targets)
        
        # Update metrics
        batch_size = inputs.size(0)
        metrics.update({
            'loss': loss_components['total'],
            'l1': loss_components.get('l1', 0),
            'perceptual': loss_components.get('perceptual', 0),
            'psnr': psnr
        }, batch_size)
        
        # Update progress bar
        if batch_idx % log_every == 0:
            pbar.set_postfix({
                'loss': f"{metrics.metrics['loss'].avg:.4f}",
                'psnr': f"{metrics.metrics['psnr'].avg:.2f}"
            })
    
    return metrics.get_averages()


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    Validate the model
    
    Returns:
        Dictionary of average metrics
    """
    model.eval()
    
    metrics = MetricTracker(['loss', 'l1', 'perceptual', 'psnr'])
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss, loss_components = criterion(outputs, targets)
            
            psnr = calculate_psnr(outputs, targets)
            
            batch_size = inputs.size(0)
            metrics.update({
                'loss': loss_components['total'],
                'l1': loss_components.get('l1', 0),
                'perceptual': loss_components.get('perceptual', 0),
                'psnr': psnr
            }, batch_size)
    
    return metrics.get_averages()


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load config
    config = TrainingConfig()
    
    # Override with command line arguments
    if args.test:
        print("\n" + "="*50)
        print("TEST MODE: Training with 50 samples for 10 epochs")
        print("="*50 + "\n")
        max_samples = config.test_samples
        epochs = config.test_epochs
        batch_size = config.test_batch_size
    else:
        max_samples = None
        epochs = args.epochs if args.epochs else config.epochs
        batch_size = args.batch_size if args.batch_size else config.batch_size
    
    crop_size = args.crop_size if args.crop_size else config.crop_size
    lr = args.lr if args.lr else config.learning_rate
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{'test_' if args.test else ''}run_{timestamp}"
    
    output_dir = Path(args.output) / run_name
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(
        dataset_path=args.data,
        batch_size=batch_size,
        crop_size=crop_size,
        num_workers=args.workers,
        pin_memory=True,
        val_split=0.1,
        max_samples=max_samples
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Crop size: {crop_size}")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = create_model(args.model, num_brackets=config.num_brackets)
    model = model.to(device)
    
    params = count_parameters(model)
    print(f"Model parameters: {params['total']:,} ({params['trainable']:,} trainable)")
    
    # Create loss function
    criterion = CombinedLoss(
        l1_weight=config.l1_weight,
        perceptual_weight=config.perceptual_weight,
        ssim_weight=config.ssim_weight,
        use_charbonnier=True
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
        )
    
    # Mixed precision training
    scaler = GradScaler('cuda', enabled=config.use_amp)
    
    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0.0)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Learning rate: {lr}")
    print("-" * 60)
    
    total_start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, config.use_amp, config.log_every
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('PSNR/train', train_metrics['psnr'], epoch)
        writer.add_scalar('PSNR/val', val_metrics['psnr'], epoch)
        writer.add_scalar('LR', get_lr(optimizer), epoch)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Time: {format_time(epoch_time)} | "
              f"LR: {get_lr(optimizer):.2e} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Train PSNR: {train_metrics['psnr']:.2f} | "
              f"Val PSNR: {val_metrics['psnr']:.2f}")
        
        # Save checkpoints
        is_best = val_metrics['psnr'] > best_psnr
        if is_best:
            best_psnr = val_metrics['psnr']
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_metrics['loss'],
            str(checkpoint_dir / "latest.pth"),
            scheduler, scaler,
            config={
                'epochs': epochs,
                'batch_size': batch_size,
                'crop_size': crop_size,
                'lr': lr,
                'model_type': args.model,
                'best_psnr': best_psnr
            }
        )
        
        # Save best checkpoint
        if is_best:
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                str(checkpoint_dir / "best.pth"),
                scheduler, scaler,
                config={
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'crop_size': crop_size,
                    'lr': lr,
                    'model_type': args.model,
                    'best_psnr': best_psnr
                }
            )
            print(f"  -> New best model! PSNR: {best_psnr:.2f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                str(checkpoint_dir / f"epoch_{epoch+1:04d}.pth"),
                scheduler, scaler
            )
    
    # Training complete
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total time: {format_time(total_time)}")
    print(f"Best PSNR: {best_psnr:.2f}")
    print(f"Best model saved to: {checkpoint_dir / 'best.pth'}")
    print(f"Logs saved to: {log_dir}")
    
    writer.close()


if __name__ == "__main__":
    main()

