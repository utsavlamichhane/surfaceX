#!/usr/bin/env python3
"""
Vesuvius Challenge Surface Detection - Improved Training Pipeline
Target: LB Score 0.65+

Key Improvements:
1. Multiple model architectures with ensemble support
2. Advanced loss functions (Combo Loss with Dice + CE + Focal + Boundary)
3. Comprehensive 3D data augmentation
4. Mixed precision training for faster training
5. Cosine annealing with warm restarts
6. Deep supervision for better gradients
7. Optimized post-processing
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Set seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# 3D Data Augmentation
# ============================================================================

class RandomFlip3D:
    """Random flip along each axis with given probability."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for axis in [0, 1, 2]:
            if random.random() < self.p:
                image = np.flip(image, axis=axis).copy()
                mask = np.flip(mask, axis=axis).copy()
        return image, mask


class RandomRotate3D:
    """Random 90-degree rotations on axial plane."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            k = random.randint(1, 3)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        return image, mask


class RandomIntensityShift:
    """Random intensity shift and scale."""
    def __init__(self, shift_range: float = 0.1, scale_range: float = 0.1, p: float = 0.5):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            shift = random.uniform(-self.shift_range, self.shift_range)
            scale = random.uniform(1 - self.scale_range, 1 + self.scale_range)
            image = image * scale + shift
        return image, mask


class RandomGaussianNoise:
    """Add random Gaussian noise."""
    def __init__(self, std_range: Tuple[float, float] = (0.01, 0.05), p: float = 0.3):
        self.std_range = std_range
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = np.random.normal(0, std, image.shape).astype(image.dtype)
            image = image + noise
        return image, mask


class RandomGaussianBlur3D:
    """Random 3D Gaussian blur."""
    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 1.5), p: float = 0.2):
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            from scipy.ndimage import gaussian_filter
            sigma = random.uniform(*self.sigma_range)
            image = gaussian_filter(image, sigma=sigma)
        return image, mask


class RandomCrop3D:
    """Random 3D crop with optional padding."""
    def __init__(self, crop_size: Tuple[int, int, int]):
        self.crop_size = crop_size
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d, h, w = image.shape
        cd, ch, cw = self.crop_size
        
        # Pad if necessary
        pad_d = max(0, cd - d)
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            mask = np.pad(mask, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        
        d, h, w = image.shape
        
        # Random crop
        d_start = random.randint(0, d - cd)
        h_start = random.randint(0, h - ch)
        w_start = random.randint(0, w - cw)
        
        image = image[d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]
        mask = mask[d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]
        
        return image, mask


class Compose3D:
    """Compose multiple transforms."""
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


# ============================================================================
# Dataset
# ============================================================================

class VesuviusDataset(Dataset):
    """Vesuvius Surface Detection Dataset."""
    
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        csv_file: str,
        transform: Optional[Compose3D] = None,
        crop_size: Tuple[int, int, int] = (160, 160, 160),
        num_classes: int = 3,
        is_train: bool = True,
        num_crops_per_volume: int = 4,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.is_train = is_train
        self.num_crops_per_volume = num_crops_per_volume
        
        # Load CSV
        self.df = pd.read_csv(csv_file)
        all_ids = self.df['id'].tolist()
        
        # Filter to only include IDs where both image and label files exist
        self.image_ids = []
        missing_count = 0
        for img_id in all_ids:
            image_path = self.image_dir / f"{img_id}.tif"
            label_path = self.label_dir / f"{img_id}.tif"
            if image_path.exists() and label_path.exists():
                self.image_ids.append(img_id)
            else:
                missing_count += 1
        
        print(f"Found {len(self.image_ids)} volumes with both image and label")
        if missing_count > 0:
            print(f"Skipped {missing_count} IDs with missing files")
    
    def __len__(self):
        if self.is_train:
            return len(self.image_ids) * self.num_crops_per_volume
        return len(self.image_ids)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using non-zero mean and std."""
        nonzero_mask = image > 0
        if nonzero_mask.any():
            mean = image[nonzero_mask].mean()
            std = image[nonzero_mask].std()
            if std > 0:
                image = (image - mean) / std
        return image
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        volume_idx = idx % len(self.image_ids)
        image_id = self.image_ids[volume_idx]
        
        # Load image
        image_path = self.image_dir / f"{image_id}.tif"
        image = tifffile.imread(str(image_path)).astype(np.float32)
        
        # Load label
        label_path = self.label_dir / f"{image_id}.tif"
        mask = tifffile.imread(str(label_path)).astype(np.int64)
        
        # Normalize image
        image = self.normalize(image)
        
        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Add channel dimension
        image = image[np.newaxis, ...]  # (1, D, H, W)
        
        return {
            'image': torch.from_numpy(image.copy()),
            'mask': torch.from_numpy(mask.copy()),
            'id': image_id
        }


# ============================================================================
# Loss Functions
# ============================================================================

class DiceLoss(nn.Module):
    """Soft Dice Loss for multi-class segmentation."""
    
    def __init__(self, num_classes: int = 3, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B, C, D, H, W) - softmax probabilities
        # target: (B, D, H, W) - class indices
        
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_onehot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """Boundary-aware loss using distance transform."""
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
    
    def compute_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute boundary map using Sobel-like filter."""
        from scipy.ndimage import binary_dilation, binary_erosion
        
        mask_np = mask.cpu().numpy()
        boundary = np.zeros_like(mask_np, dtype=np.float32)
        
        for i in range(mask_np.shape[0]):
            for c in range(self.num_classes):
                class_mask = (mask_np[i] == c).astype(np.float32)
                eroded = binary_erosion(class_mask, iterations=1)
                boundary[i] += (class_mask - eroded)
        
        return torch.from_numpy(boundary).to(mask.device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_softmax = F.softmax(pred, dim=1)
        boundary_weights = self.compute_boundary(target)
        
        # Weighted cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = ce_loss * (1 + boundary_weights * 2)
        
        return weighted_loss.mean()


class ComboLoss(nn.Module):
    """Combined loss function for better training."""
    
    def __init__(
        self,
        num_classes: int = 3,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        focal_weight: float = 0.5,
        boundary_weight: float = 0.5,
    ):
        super().__init__()
        self.dice_loss = DiceLoss(num_classes)
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss(num_classes)
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(pred, target)
        
        if self.ce_weight > 0:
            loss += self.ce_weight * F.cross_entropy(pred, target)
        
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(pred, target)
        
        # Boundary loss is optional and expensive
        # if self.boundary_weight > 0:
        #     loss += self.boundary_weight * self.boundary_loss(pred, target)
        
        return loss


# ============================================================================
# 3D UNet Model
# ============================================================================

class ConvBlock3D(nn.Module):
    """3D Convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock3D(nn.Module):
    """Encoder block with downsampling."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock3D(nn.Module):
    """Decoder block with upsampling and skip connection."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels // 2 + skip_channels, out_channels, dropout)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2,
                         diff_d // 2, diff_d - diff_d // 2])
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D UNet with deep supervision option."""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        dropout: float = 0.1,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.enc1 = EncoderBlock3D(in_channels, base_channels, dropout)
        self.enc2 = EncoderBlock3D(base_channels, base_channels * 2, dropout)
        self.enc3 = EncoderBlock3D(base_channels * 2, base_channels * 4, dropout)
        self.enc4 = EncoderBlock3D(base_channels * 4, base_channels * 8, dropout)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(base_channels * 8, base_channels * 16, dropout)
        
        # Decoder
        self.dec4 = DecoderBlock3D(base_channels * 16, base_channels * 8, base_channels * 8, dropout)
        self.dec3 = DecoderBlock3D(base_channels * 8, base_channels * 4, base_channels * 4, dropout)
        self.dec2 = DecoderBlock3D(base_channels * 4, base_channels * 2, base_channels * 2, dropout)
        self.dec1 = DecoderBlock3D(base_channels * 2, base_channels, base_channels, dropout)
        
        # Output
        self.out = nn.Conv3d(base_channels, num_classes, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.ds4 = nn.Conv3d(base_channels * 8, num_classes, kernel_size=1)
            self.ds3 = nn.Conv3d(base_channels * 4, num_classes, kernel_size=1)
            self.ds2 = nn.Conv3d(base_channels * 2, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder
        d4 = self.dec4(x, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        
        # Output
        out = self.out(d1)
        
        if self.training and self.deep_supervision:
            ds4 = F.interpolate(self.ds4(d4), size=out.shape[2:], mode='trilinear', align_corners=False)
            ds3 = F.interpolate(self.ds3(d3), size=out.shape[2:], mode='trilinear', align_corners=False)
            ds2 = F.interpolate(self.ds2(d2), size=out.shape[2:], mode='trilinear', align_corners=False)
            return out, ds4, ds3, ds2
        
        return out


class ResBlock3D(nn.Module):
    """Residual block for ResUNet."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class AttentionGate3D(nn.Module):
    """Attention gate for skip connections."""
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet3D(nn.Module):
    """3D Attention UNet with SE blocks."""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        dropout: float = 0.1,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.enc1 = nn.Sequential(ResBlock3D(in_channels, base_channels, dropout), SEBlock3D(base_channels))
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = nn.Sequential(ResBlock3D(base_channels, base_channels * 2, dropout), SEBlock3D(base_channels * 2))
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = nn.Sequential(ResBlock3D(base_channels * 2, base_channels * 4, dropout), SEBlock3D(base_channels * 4))
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = nn.Sequential(ResBlock3D(base_channels * 4, base_channels * 8, dropout), SEBlock3D(base_channels * 8))
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(ResBlock3D(base_channels * 8, base_channels * 16, dropout), SEBlock3D(base_channels * 16))
        
        # Decoder with attention gates
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate3D(base_channels * 8, base_channels * 8, base_channels * 4)
        self.dec4 = nn.Sequential(ResBlock3D(base_channels * 16, base_channels * 8, dropout), SEBlock3D(base_channels * 8))
        
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate3D(base_channels * 4, base_channels * 4, base_channels * 2)
        self.dec3 = nn.Sequential(ResBlock3D(base_channels * 8, base_channels * 4, dropout), SEBlock3D(base_channels * 4))
        
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate3D(base_channels * 2, base_channels * 2, base_channels)
        self.dec2 = nn.Sequential(ResBlock3D(base_channels * 4, base_channels * 2, dropout), SEBlock3D(base_channels * 2))
        
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate3D(base_channels, base_channels, base_channels // 2)
        self.dec1 = nn.Sequential(ResBlock3D(base_channels * 2, base_channels, dropout), SEBlock3D(base_channels))
        
        # Output
        self.out = nn.Conv3d(base_channels, num_classes, kernel_size=1)
        
        # Deep supervision
        if deep_supervision:
            self.ds4 = nn.Conv3d(base_channels * 8, num_classes, kernel_size=1)
            self.ds3 = nn.Conv3d(base_channels * 4, num_classes, kernel_size=1)
            self.ds2 = nn.Conv3d(base_channels * 2, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with attention
        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        out = self.out(d1)
        
        if self.training and self.deep_supervision:
            ds4 = F.interpolate(self.ds4(d4), size=out.shape[2:], mode='trilinear', align_corners=False)
            ds3 = F.interpolate(self.ds3(d3), size=out.shape[2:], mode='trilinear', align_corners=False)
            ds2 = F.interpolate(self.ds2(d2), size=out.shape[2:], mode='trilinear', align_corners=False)
            return out, ds4, ds3, ds2
        
        return out


# ============================================================================
# Metrics
# ============================================================================

def compute_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> Dict[str, float]:
    """Compute per-class and mean Dice score."""
    pred_classes = pred.argmax(dim=1)
    dice_scores = {}
    
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = (2.0 * intersection / union).item()
        else:
            dice = 1.0 if intersection == 0 else 0.0
        
        dice_scores[f'dice_class_{c}'] = dice
    
    dice_scores['dice_mean'] = np.mean(list(dice_scores.values()))
    return dice_scores


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    deep_supervision: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_dice = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            if deep_supervision:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    main_out, ds4, ds3, ds2 = outputs
                    # Weighted deep supervision loss
                    loss = criterion(main_out, masks)
                    loss += 0.3 * criterion(ds4, masks)
                    loss += 0.2 * criterion(ds3, masks)
                    loss += 0.1 * criterion(ds2, masks)
                else:
                    loss = criterion(outputs, masks)
                    main_out = outputs
            else:
                main_out = model(images)
                loss = criterion(main_out, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Compute Dice
        with torch.no_grad():
            dice = compute_dice(main_out, masks)
            all_dice.append(dice['dice_mean'])
        
        pbar.set_postfix({'loss': loss.item(), 'dice': np.mean(all_dice)})
    
    return {
        'train_loss': total_loss / len(dataloader),
        'train_dice': np.mean(all_dice)
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_dice = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Validation')
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        loss = criterion(outputs, masks)
        total_loss += loss.item()
        
        dice = compute_dice(outputs, masks)
        all_dice.append(dice['dice_mean'])
        
        pbar.set_postfix({'loss': loss.item(), 'dice': np.mean(all_dice)})
    
    return {
        'val_loss': total_loss / len(dataloader),
        'val_dice': np.mean(all_dice)
    }


def main(args):
    """Main training function."""
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create transforms
    train_transforms = Compose3D([
        RandomCrop3D(args.crop_size),
        RandomFlip3D(p=0.5),
        RandomRotate3D(p=0.5),
        RandomIntensityShift(shift_range=0.1, scale_range=0.1, p=0.3),
        RandomGaussianNoise(std_range=(0.01, 0.03), p=0.2),
    ])
    
    val_transforms = Compose3D([
        RandomCrop3D(args.crop_size),  # For validation, we still need to crop to fit in memory
    ])
    
    # Create datasets
    train_dataset = VesuviusDataset(
        image_dir=args.train_images,
        label_dir=args.train_labels,
        csv_file=args.train_csv,
        transform=train_transforms,
        crop_size=args.crop_size,
        num_classes=args.num_classes,
        is_train=True,
        num_crops_per_volume=args.num_crops,
    )
    
    # Use same dataset for validation with different transforms
    val_dataset = VesuviusDataset(
        image_dir=args.train_images,
        label_dir=args.train_labels,
        csv_file=args.train_csv,
        transform=val_transforms,
        crop_size=args.crop_size,
        num_classes=args.num_classes,
        is_train=False,
        num_crops_per_volume=1,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    print(f"\nCreating model: {args.model}")
    if args.model == 'unet3d':
        model = UNet3D(
            in_channels=1,
            num_classes=args.num_classes,
            base_channels=args.base_channels,
            dropout=args.dropout,
            deep_supervision=args.deep_supervision,
        )
    elif args.model == 'attention_unet3d':
        model = AttentionUNet3D(
            in_channels=1,
            num_classes=args.num_classes,
            base_channels=args.base_channels,
            dropout=args.dropout,
            deep_supervision=args.deep_supervision,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
    
    # Loss function
    criterion = ComboLoss(
        num_classes=args.num_classes,
        dice_weight=1.0,
        ce_weight=1.0,
        focal_weight=0.5,
        boundary_weight=0.0,  # Disabled for speed
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.epochs // 3,
        T_mult=1,
        eta_min=args.lr * 0.01,
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_dice = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.deep_supervision
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
        history.append(metrics)
        
        print(f"\nTrain Loss: {train_metrics['train_loss']:.4f}, Train Dice: {train_metrics['train_dice']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val Dice: {val_metrics['val_dice']:.4f}")
        
        # Save best model
        if val_metrics['val_dice'] > best_dice:
            best_dice = val_metrics['val_dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_dice,
                'args': vars(args),
            }, output_dir / f'best_model_{args.model}.pth')
            print(f"Saved new best model with Dice: {best_dice:.4f}")
        
        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_metrics['val_dice'],
                'args': vars(args),
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_metrics['val_dice'],
        'args': vars(args),
    }, output_dir / f'final_model_{args.model}.pth')
    
    # Save history
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vesuvius Challenge Training')
    
    # Data paths
    parser.add_argument('--train-images', type=str, default='train_images',
                        help='Path to training images directory')
    parser.add_argument('--train-labels', type=str, default='train_labels',
                        help='Path to training labels directory')
    parser.add_argument('--train-csv', type=str, default='train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for models')
    
    # Model
    parser.add_argument('--model', type=str, default='attention_unet3d',
                        choices=['unet3d', 'attention_unet3d'],
                        help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--base-channels', type=int, default=32,
                        help='Base number of channels')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--deep-supervision', action='store_true', default=True,
                        help='Use deep supervision')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--crop-size', type=int, nargs=3, default=[160, 160, 160],
                        help='Crop size (D H W)')
    parser.add_argument('--num-crops', type=int, default=4,
                        help='Number of crops per volume per epoch')
    
    # Other
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    args.crop_size = tuple(args.crop_size)
    
    main(args)
