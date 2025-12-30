"""
AHDR-Net: Attention-guided HDR Reconstruction Network

Architecture based on the paper:
"Attention-guided Network for Ghost-free High Dynamic Range Imaging"

Key components:
1. Feature extraction with attention modules
2. Dilated Residual Dense Blocks (DRDB)
3. Attention-guided merging
4. Reconstruction network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ChannelAttention(nn.Module):
    """Channel attention module for feature recalibration"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on important regions"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class DenseLayer(nn.Module):
    """Dense layer for DRDB"""
    
    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv(x))
        return torch.cat([x, out], dim=1)


class DilatedResidualDenseBlock(nn.Module):
    """
    Dilated Residual Dense Block (DRDB)
    Combines dense connections with dilated convolutions for larger receptive field
    """
    
    def __init__(self, channels: int, growth_rate: int = 32, num_layers: int = 4):
        super().__init__()
        
        self.layers = nn.ModuleList()
        in_channels = channels
        
        for i in range(num_layers):
            # Alternate between different dilation rates
            dilation = 1 + (i % 3)  # 1, 2, 3, 1, ...
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, growth_rate, 3, padding=dilation, dilation=dilation, bias=True),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            in_channels += growth_rate
        
        # Local feature fusion
        self.lff = nn.Conv2d(in_channels, channels, 1, bias=True)
        
        # Attention
        self.attention = CBAM(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x
        for layer in self.layers:
            out = layer(features)
            features = torch.cat([features, out], dim=1)
        
        out = self.lff(features)
        out = self.attention(out)
        
        return x + out  # Residual connection


class AttentionModule(nn.Module):
    """
    Attention module for handling misalignment and saturation between exposures
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns:
            features: Extracted features
            attention_map: Attention weights for this exposure
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        features = self.relu(self.conv3(x))
        attention_map = self.attention(features)
        
        return features, attention_map


class FeatureExtractor(nn.Module):
    """Feature extraction network for each exposure"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class AHDRNet(nn.Module):
    """
    AHDR-Net: Attention-guided HDR Reconstruction Network
    
    Takes 5 bracketed LDR images and produces a tone-mapped HDR image.
    
    Input: [B, 15, H, W] - 5 images × 3 channels
    Output: [B, 3, H, W] - Reconstructed HDR image
    """
    
    def __init__(
        self,
        num_brackets: int = 5,
        base_channels: int = 64,
        num_drdb: int = 6,
        growth_rate: int = 32
    ):
        super().__init__()
        
        self.num_brackets = num_brackets
        self.base_channels = base_channels
        
        # Feature extractors for each exposure (shared weights)
        self.feature_extractor = FeatureExtractor(3, base_channels)
        
        # Attention modules for each exposure (individual)
        self.attention_modules = nn.ModuleList([
            AttentionModule(base_channels, base_channels)
            for _ in range(num_brackets)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * num_brackets, base_channels * 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Dilated Residual Dense Blocks
        self.drdb_blocks = nn.ModuleList([
            DilatedResidualDenseBlock(base_channels, growth_rate)
            for _ in range(num_drdb)
        ])
        
        # Global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(base_channels * num_drdb, base_channels, 1),
            nn.Conv2d(base_channels, base_channels, 3, padding=1)
        )
        
        # Reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 3, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 15, H, W] (5 images × 3 channels)
        
        Returns:
            Reconstructed HDR image [B, 3, H, W]
        """
        B, C, H, W = x.shape
        
        # Split into individual exposures
        exposures = torch.chunk(x, self.num_brackets, dim=1)  # 5 × [B, 3, H, W]
        
        # Extract features and compute attention for each exposure
        features_list = []
        attention_maps = []
        
        for i, exp in enumerate(exposures):
            # Shared feature extraction
            feat = self.feature_extractor(exp)
            # Individual attention
            feat, attn = self.attention_modules[i](feat)
            features_list.append(feat * attn)  # Apply attention
            attention_maps.append(attn)
        
        # Concatenate weighted features
        merged_features = torch.cat(features_list, dim=1)  # [B, base_channels*5, H, W]
        
        # Fusion
        fused = self.fusion(merged_features)  # [B, base_channels, H, W]
        
        # Pass through DRDB blocks
        drdb_outputs = []
        current = fused
        for drdb in self.drdb_blocks:
            current = drdb(current)
            drdb_outputs.append(current)
        
        # Global feature fusion
        global_features = torch.cat(drdb_outputs, dim=1)
        global_features = self.gff(global_features)
        
        # Residual learning
        enhanced = fused + global_features
        
        # Reconstruction
        output = self.reconstruction(enhanced)
        
        return output


class AHDRNetLite(nn.Module):
    """
    Lighter version of AHDR-Net for faster training/inference
    """
    
    def __init__(
        self,
        num_brackets: int = 5,
        base_channels: int = 48,
        num_drdb: int = 4,
        growth_rate: int = 24
    ):
        super().__init__()
        
        self.num_brackets = num_brackets
        in_channels = num_brackets * 3  # 15 channels
        
        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Attention-guided processing
        self.attention = CBAM(base_channels)
        
        # DRDB blocks
        self.drdb_blocks = nn.ModuleList([
            DilatedResidualDenseBlock(base_channels, growth_rate)
            for _ in range(num_drdb)
        ])
        
        # Feature fusion
        self.fusion = nn.Conv2d(base_channels * num_drdb, base_channels, 1)
        
        # Reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial features
        feat = self.head(x)
        feat = self.attention(feat)
        
        # DRDB processing
        outputs = []
        current = feat
        for drdb in self.drdb_blocks:
            current = drdb(current)
            outputs.append(current)
        
        # Fusion
        fused = self.fusion(torch.cat(outputs, dim=1))
        
        # Residual + reconstruction
        out = self.tail(feat + fused)
        
        return out


def create_model(model_type: str = "full", num_brackets: int = 5) -> nn.Module:
    """
    Factory function to create AHDR-Net model
    
    Args:
        model_type: "full" for AHDRNet, "lite" for AHDRNetLite
        num_brackets: Number of bracketed exposures
    
    Returns:
        Model instance
    """
    if model_type == "lite":
        return AHDRNetLite(num_brackets=num_brackets)
    else:
        return AHDRNet(num_brackets=num_brackets)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = AHDRNet(num_brackets=5).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 15, 256, 256).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test lite model
    print("\n--- Lite Model ---")
    model_lite = AHDRNetLite(num_brackets=5).to(device)
    total_params_lite = sum(p.numel() for p in model_lite.parameters())
    print(f"Lite model parameters: {total_params_lite:,}")
    
    with torch.no_grad():
        output_lite = model_lite(dummy_input)
    print(f"Lite output shape: {output_lite.shape}")

