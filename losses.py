"""
Loss Functions for HDR Training - Realistic Textures Edition

Includes:
- L1 Loss (pixel-wise reconstruction)
- Perceptual Loss (VGG-based feature matching)
- SSIM Loss (structural similarity)
- Edge Loss (gradient/sharpness preservation)
- Frequency Loss (high-frequency detail preservation)
- Combined Loss (all of the above)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional


class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor for perceptual loss
    Extracts features from multiple layers for multi-scale perceptual comparison
    """
    
    def __init__(
        self,
        layer_indices: List[int] = [2, 7, 12, 21, 30],
        use_input_norm: bool = True
    ):
        """
        Args:
            layer_indices: Indices of VGG layers to extract features from
            use_input_norm: Whether to normalize input to ImageNet statistics
        """
        super().__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Extract feature layers
        self.features = nn.ModuleList()
        prev_idx = 0
        for idx in layer_indices:
            self.features.append(nn.Sequential(*list(vgg.features[prev_idx:idx + 1])))
            prev_idx = idx + 1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Normalization
        self.use_input_norm = use_input_norm
        if use_input_norm:
            # ImageNet normalization
            self.register_buffer(
                'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features
        
        Args:
            x: Input tensor [B, 3, H, W] in range [0, 1]
        
        Returns:
            List of feature tensors from different VGG layers
        """
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)
        
        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features
    Compares high-level features between prediction and target
    """
    
    def __init__(
        self,
        layer_weights: Optional[List[float]] = None,
        use_input_norm: bool = True
    ):
        """
        Args:
            layer_weights: Weights for each layer's contribution to the loss
            use_input_norm: Whether to normalize input to ImageNet statistics
        """
        super().__init__()
        
        self.vgg = VGGFeatureExtractor(use_input_norm=use_input_norm)
        
        # Default weights for each layer
        if layer_weights is None:
            self.layer_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        else:
            self.layer_weights = layer_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            Perceptual loss value
        """
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        loss = 0.0
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.layer_weights):
            loss += weight * F.l1_loss(pred_feat, target_feat)
        
        return loss


class L1Loss(nn.Module):
    """Simple L1 pixel-wise loss"""
    
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1)
    More robust to outliers than L1
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss
    Measures structural similarity between images
    """
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM computation"""
        def gaussian(window_size: int, sigma: float) -> torch.Tensor:
            gauss = torch.tensor([
                -(x - window_size // 2) ** 2 / (2 * sigma ** 2)
                for x in range(window_size)
            ]).exp()
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channel = pred.size(1)
        
        if channel != self.channel or self.window.device != pred.device:
            self.window = self._create_window(self.window_size, channel).to(pred.device)
            self.channel = channel
        
        return 1 - self._ssim(pred, target, self.window, self.window_size, channel, self.size_average)
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int,
        size_average: bool = True
    ) -> torch.Tensor:
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class EdgeLoss(nn.Module):
    """
    Edge/Gradient loss for preserving sharpness and details.
    Compares gradients (edges) between prediction and target using Sobel filters.
    This helps maintain sharp edges and fine details in the output.
    """
    
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        
        # Register as buffers so they move to GPU with the model
        self.register_buffer('sobel_x_buf', self.sobel_x)
        self.register_buffer('sobel_y_buf', self.sobel_y)
    
    def get_gradients(self, img: torch.Tensor) -> torch.Tensor:
        """Compute image gradients using Sobel filters"""
        # Convert to grayscale for gradient computation
        # Using luminance weights: 0.299*R + 0.587*G + 0.114*B
        weights = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)
        img_gray = (img * weights).sum(dim=1, keepdim=True)
        
        # Compute gradients
        grad_x = F.conv2d(img_gray, self.sobel_x_buf.to(img.device), padding=1)
        grad_y = F.conv2d(img_gray, self.sobel_y_buf.to(img.device), padding=1)
        
        # Gradient magnitude
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return gradient
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute edge loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            Edge loss value
        """
        pred_grad = self.get_gradients(pred)
        target_grad = self.get_gradients(target)
        return F.l1_loss(pred_grad, target_grad)


class FrequencyLoss(nn.Module):
    """
    Frequency domain loss for preserving high-frequency details (textures).
    Works in Fourier space to ensure fine details are preserved.
    This is especially important for textures like fabric, foliage, and skin.
    """
    
    def __init__(self, focus_on_high_freq: bool = True):
        """
        Args:
            focus_on_high_freq: If True, weights high frequencies more heavily
        """
        super().__init__()
        self.focus_on_high_freq = focus_on_high_freq
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            Frequency loss value
        """
        # FFT of both images
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # Compare amplitudes (magnitudes)
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        if self.focus_on_high_freq:
            # Create high-frequency emphasis mask
            B, C, H, W = pred_mag.shape
            # Distance from center (low frequencies) increases weight
            y_coords = torch.linspace(0, 1, H, device=pred.device).view(1, 1, H, 1)
            x_coords = torch.linspace(0, 1, W, device=pred.device).view(1, 1, 1, W)
            # Higher weight for higher frequencies
            freq_weight = torch.sqrt(y_coords ** 2 + x_coords ** 2).clamp(0.1, 1.0)
            
            loss = F.l1_loss(pred_mag * freq_weight, target_mag * freq_weight)
        else:
            loss = F.l1_loss(pred_mag, target_mag)
        
        return loss


class LaplacianLoss(nn.Module):
    """
    Laplacian loss for multi-scale edge preservation.
    Uses Laplacian pyramid to preserve edges at multiple scales.
    """
    
    def __init__(self, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        
        # Laplacian kernel
        self.laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
    
    def get_laplacian(self, img: torch.Tensor) -> torch.Tensor:
        """Apply Laplacian filter to image"""
        B, C, H, W = img.shape
        laplacian = self.laplacian.to(img.device)
        
        # Apply to each channel
        channels = []
        for c in range(C):
            channel = img[:, c:c+1, :, :]
            lap = F.conv2d(channel, laplacian, padding=1)
            channels.append(lap)
        
        return torch.cat(channels, dim=1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale Laplacian loss"""
        total_loss = 0.0
        
        current_pred = pred
        current_target = target
        
        for level in range(self.num_levels):
            # Compute Laplacian at current scale
            pred_lap = self.get_laplacian(current_pred)
            target_lap = self.get_laplacian(current_target)
            
            # Weight decreases for coarser scales
            weight = 1.0 / (2 ** level)
            total_loss += weight * F.l1_loss(pred_lap, target_lap)
            
            # Downsample for next level
            if level < self.num_levels - 1:
                current_pred = F.avg_pool2d(current_pred, 2)
                current_target = F.avg_pool2d(current_target, 2)
        
        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for HDR training - Realistic Textures Edition
    
    Combines multiple loss functions for high-quality HDR reconstruction:
    - L1/Charbonnier: Pixel-wise reconstruction
    - Perceptual: High-level texture matching via VGG features
    - SSIM: Structural similarity preservation
    - Edge: Gradient/sharpness preservation
    - Frequency: High-frequency detail preservation
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.2,
        ssim_weight: float = 0.1,
        edge_weight: float = 0.1,
        frequency_weight: float = 0.05,
        use_charbonnier: bool = True
    ):
        """
        Args:
            l1_weight: Weight for L1/Charbonnier loss (base reconstruction)
            perceptual_weight: Weight for perceptual loss (textures)
            ssim_weight: Weight for SSIM loss (structure)
            edge_weight: Weight for edge loss (sharpness)
            frequency_weight: Weight for frequency loss (fine details)
            use_charbonnier: Use Charbonnier instead of L1 (more robust)
        """
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.frequency_weight = frequency_weight
        
        # Initialize loss functions
        self.l1_loss = CharbonnierLoss() if use_charbonnier else L1Loss()
        
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
        
        if ssim_weight > 0:
            self.ssim_loss = SSIMLoss()
        else:
            self.ssim_loss = None
        
        if edge_weight > 0:
            self.edge_loss = EdgeLoss()
        else:
            self.edge_loss = None
        
        if frequency_weight > 0:
            self.frequency_loss = FrequencyLoss(focus_on_high_freq=True)
        else:
            self.frequency_loss = None
        
        # Log active losses
        active_losses = []
        if l1_weight > 0:
            active_losses.append(f"L1({l1_weight})")
        if perceptual_weight > 0:
            active_losses.append(f"Perceptual({perceptual_weight})")
        if ssim_weight > 0:
            active_losses.append(f"SSIM({ssim_weight})")
        if edge_weight > 0:
            active_losses.append(f"Edge({edge_weight})")
        if frequency_weight > 0:
            active_losses.append(f"Frequency({frequency_weight})")
        print(f"[CombinedLoss] Active losses: {', '.join(active_losses)}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            Tuple of (total_loss, dict of individual loss components)
        """
        losses = {}
        total_loss = 0.0
        
        # L1 / Charbonnier loss - base reconstruction
        l1 = self.l1_loss(pred, target)
        losses['l1'] = l1.item()
        total_loss += self.l1_weight * l1
        
        # Perceptual loss - texture matching via VGG features
        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(pred, target)
            losses['perceptual'] = perceptual.item()
            total_loss += self.perceptual_weight * perceptual
        
        # SSIM loss - structural similarity
        if self.ssim_loss is not None:
            ssim = self.ssim_loss(pred, target)
            losses['ssim'] = ssim.item()
            total_loss += self.ssim_weight * ssim
        
        # Edge loss - gradient/sharpness preservation
        if self.edge_loss is not None:
            edge = self.edge_loss(pred, target)
            losses['edge'] = edge.item()
            total_loss += self.edge_weight * edge
        
        # Frequency loss - high-frequency detail preservation
        if self.frequency_loss is not None:
            freq = self.frequency_loss(pred, target)
            losses['frequency'] = freq.item()
            total_loss += self.frequency_weight * freq
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


if __name__ == "__main__":
    # Test the loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy tensors
    pred = torch.rand(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)
    
    # Test individual losses
    print("\n--- Individual Losses ---")
    
    l1_loss = L1Loss()
    print(f"L1 Loss: {l1_loss(pred, target).item():.6f}")
    
    charbonnier = CharbonnierLoss()
    print(f"Charbonnier Loss: {charbonnier(pred, target).item():.6f}")
    
    perceptual = PerceptualLoss().to(device)
    print(f"Perceptual Loss: {perceptual(pred, target).item():.6f}")
    
    ssim = SSIMLoss()
    print(f"SSIM Loss: {ssim(pred, target).item():.6f}")
    
    edge = EdgeLoss().to(device)
    print(f"Edge Loss: {edge(pred, target).item():.6f}")
    
    freq = FrequencyLoss().to(device)
    print(f"Frequency Loss: {freq(pred, target).item():.6f}")
    
    laplacian = LaplacianLoss().to(device)
    print(f"Laplacian Loss: {laplacian(pred, target).item():.6f}")
    
    # Test combined loss with realistic textures configuration
    print("\n--- Combined Loss (Realistic Textures) ---")
    combined = CombinedLoss(
        l1_weight=1.0,
        perceptual_weight=0.2,
        ssim_weight=0.1,
        edge_weight=0.1,
        frequency_weight=0.05
    ).to(device)
    
    total, components = combined(pred, target)
    print(f"Total Loss: {total.item():.6f}")
    for name, value in components.items():
        print(f"  {name}: {value:.6f}")

