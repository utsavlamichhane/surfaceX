#!/usr/bin/env python3
"""
Vesuvius Challenge Surface Detection - Inference Pipeline
With Model Ensemble and Optimized Post-Processing

Key Features:
1. Model ensemble support (multiple models, multiple seeds)
2. Test-Time Augmentation (TTA)
3. Sliding Window Inference with Gaussian weighting
4. Optimized post-processing with hysteresis thresholding
5. Memory-efficient inference for large volumes
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import tifffile
import zipfile
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import model definitions from train.py
from train import UNet3D, AttentionUNet3D, set_seed


# ============================================================================
# Sliding Window Inference
# ============================================================================

class SlidingWindowInference:
    """Sliding window inference with Gaussian weighting for 3D volumes."""
    
    def __init__(
        self,
        model: nn.Module,
        roi_size: Tuple[int, int, int],
        overlap: float = 0.5,
        mode: str = 'gaussian',
        batch_size: int = 1,
        device: torch.device = None,
    ):
        self.model = model
        self.roi_size = roi_size
        self.overlap = overlap
        self.mode = mode
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Precompute importance map
        self.importance_map = self._get_importance_map()
    
    def _get_importance_map(self) -> np.ndarray:
        """Get importance map for weighted averaging."""
        if self.mode == 'gaussian':
            # Create Gaussian importance map
            center = [s // 2 for s in self.roi_size]
            sigma = [s / 4 for s in self.roi_size]
            
            z, y, x = np.ogrid[:self.roi_size[0], :self.roi_size[1], :self.roi_size[2]]
            h = np.exp(-((z - center[0])**2 / (2 * sigma[0]**2) + 
                         (y - center[1])**2 / (2 * sigma[1]**2) + 
                         (x - center[2])**2 / (2 * sigma[2]**2)))
            return h.astype(np.float32)
        else:
            return np.ones(self.roi_size, dtype=np.float32)
    
    def __call__(self, volume: np.ndarray, num_classes: int = 3) -> np.ndarray:
        """
        Perform sliding window inference.
        
        Args:
            volume: Input volume of shape (1, D, H, W, 1) or (D, H, W)
            num_classes: Number of output classes
        
        Returns:
            Probability map of shape (D, H, W) for the surface class
        """
        # Handle different input shapes
        if volume.ndim == 5:
            volume = volume[0, ..., 0]  # Remove batch and channel dims
        elif volume.ndim == 4:
            volume = volume[0] if volume.shape[0] == 1 else volume[..., 0]
        
        d, h, w = volume.shape
        rd, rh, rw = self.roi_size
        
        # Compute step sizes
        step_d = int(rd * (1 - self.overlap))
        step_h = int(rh * (1 - self.overlap))
        step_w = int(rw * (1 - self.overlap))
        
        # Pad volume if necessary
        pad_d = max(0, rd - d)
        pad_h = max(0, rh - h)
        pad_w = max(0, rw - w)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        
        padded_shape = volume.shape
        
        # Create output arrays
        output_sum = np.zeros((num_classes,) + padded_shape, dtype=np.float32)
        weight_sum = np.zeros(padded_shape, dtype=np.float32)
        
        # Collect all patches
        patches = []
        positions = []
        
        for z in range(0, padded_shape[0] - rd + 1, step_d):
            for y in range(0, padded_shape[1] - rh + 1, step_h):
                for x in range(0, padded_shape[2] - rw + 1, step_w):
                    patch = volume[z:z+rd, y:y+rh, x:x+rw]
                    patches.append(patch)
                    positions.append((z, y, x))
        
        # Add final patches at boundaries
        for z in [max(0, padded_shape[0] - rd)]:
            for y in [max(0, padded_shape[1] - rh)]:
                for x in [max(0, padded_shape[2] - rw)]:
                    if (z, y, x) not in positions:
                        patch = volume[z:z+rd, y:y+rh, x:x+rw]
                        patches.append(patch)
                        positions.append((z, y, x))
        
        # Process in batches
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch_patches = patches[i:i+self.batch_size]
                batch_positions = positions[i:i+self.batch_size]
                
                # Stack and convert to tensor
                batch = np.stack([p[np.newaxis, ...] for p in batch_patches])  # (B, 1, D, H, W)
                batch_tensor = torch.from_numpy(batch).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Convert to probabilities
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                
                # Accumulate results
                for j, (z, y, x) in enumerate(batch_positions):
                    output_sum[:, z:z+rd, y:y+rh, x:x+rw] += probs[j] * self.importance_map
                    weight_sum[z:z+rd, y:y+rh, x:x+rw] += self.importance_map
        
        # Normalize
        weight_sum = np.maximum(weight_sum, 1e-8)
        output_avg = output_sum / weight_sum[np.newaxis, ...]
        
        # Remove padding
        output_avg = output_avg[:, :d, :h, :w]
        
        # Return surface class probability (assuming class 1 is surface)
        return output_avg[1]


# ============================================================================
# Test Time Augmentation
# ============================================================================

def predict_with_tta(
    volume: np.ndarray,
    swi: SlidingWindowInference,
    num_classes: int = 3,
    tta_flips: bool = True,
    tta_rotations: bool = True,
) -> np.ndarray:
    """
    Predict with test-time augmentation.
    
    Args:
        volume: Input volume of shape (1, D, H, W, 1)
        swi: Sliding window inference object
        num_classes: Number of classes
        tta_flips: Use flip augmentations
        tta_rotations: Use rotation augmentations
    
    Returns:
        Averaged probability map
    """
    probs = []
    
    # Original
    probs.append(swi(volume, num_classes))
    
    # Flips (spatial axes)
    if tta_flips:
        for axis in [1, 2, 3]:
            img_f = np.flip(volume, axis=axis).copy()
            p = swi(img_f, num_classes)
            # Flip back
            actual_axis = axis - 1  # Adjust for squeezed dimensions
            p = np.flip(p, axis=actual_axis).copy()
            probs.append(p)
    
    # Axial rotations (H, W plane)
    if tta_rotations:
        for k in [1, 2, 3]:
            img_r = np.rot90(volume.squeeze(), k=k, axes=(1, 2))
            img_r = img_r[np.newaxis, ..., np.newaxis]
            p = swi(img_r, num_classes)
            p = np.rot90(p, k=-k, axes=(1, 2)).copy()
            probs.append(p)
    
    return np.mean(probs, axis=0)


# ============================================================================
# Post-Processing
# ============================================================================

def build_anisotropic_struct(z_radius: int, xy_radius: int) -> Optional[np.ndarray]:
    """Build anisotropic structuring element."""
    z, r = z_radius, xy_radius
    if z == 0 and r == 0:
        return None
    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[0, cy + dy, cx + dx] = True
        return struct
    if z > 0 and r == 0:
        struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
        struct[:, 0, 0] = True
        return struct
    depth = 2 * z + 1
    size = 2 * r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz, cy, cx = z, r, r
    for dz in range(-z, z + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[cz + dz, cy + dy, cx + dx] = True
    return struct


def topo_postprocess(
    probs: np.ndarray,
    T_low: float = 0.40,
    T_high: float = 0.85,
    z_radius: int = 1,
    xy_radius: int = 1,
    dust_min_size: int = 150,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Topological post-processing with optimized parameters.
    
    Key improvements:
    1. Lower T_low for better recall
    2. Slightly lower T_high for more surface detection
    3. Added xy_radius for better lateral connectivity
    4. Increased dust_min_size for cleaner results
    5. Optional hole filling
    """
    # Step 1: 3D Hysteresis thresholding
    strong = probs >= T_high
    weak = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)  # 26-connectivity
    mask = ndi.binary_propagation(
        strong, mask=weak, structure=struct_hyst
    )

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # Step 2: 3D Anisotropic Closing (to connect nearby regions)
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)
    
    # Step 3: Fill holes in each slice (optional)
    if fill_holes:
        for i in range(mask.shape[0]):
            mask[i] = ndi.binary_fill_holes(mask[i])
    
    # Step 4: Opening to remove thin connections (noise reduction)
    struct_open = ndi.generate_binary_structure(3, 1)
    mask = ndi.binary_opening(mask, structure=struct_open, iterations=1)
    
    # Step 5: Dust Removal (remove small connected components)
    if dust_min_size > 0:
        mask = remove_small_objects(
            mask.astype(bool), min_size=dust_min_size
        )

    return mask.astype(np.uint8)


def topo_postprocess_v2(
    probs: np.ndarray,
    T_low: float = 0.35,
    T_high: float = 0.80,
    z_radius: int = 2,
    xy_radius: int = 1,
    dust_min_size: int = 200,
    smooth_sigma: float = 0.5,
) -> np.ndarray:
    """
    Alternative post-processing with probability smoothing.
    """
    # Smooth probabilities
    if smooth_sigma > 0:
        probs = gaussian_filter(probs, sigma=smooth_sigma)
    
    # Hysteresis
    strong = probs >= T_high
    weak = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # Closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)
    
    # Remove small objects
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)


# ============================================================================
# Model Ensemble
# ============================================================================

class ModelEnsemble:
    """Ensemble of multiple models for better predictions."""
    
    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        device: torch.device = None,
    ):
        """
        Args:
            model_configs: List of dicts with 'checkpoint_path', 'model_type', etc.
            device: Torch device
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.swis = []
        
        for config in model_configs:
            model = self._load_model(config)
            model.eval()
            self.models.append(model)
            
            # Create SWI for each model
            roi_size = config.get('roi_size', (160, 160, 160))
            swi = SlidingWindowInference(
                model,
                roi_size=roi_size,
                overlap=config.get('overlap', 0.5),
                mode='gaussian',
                batch_size=config.get('batch_size', 1),
                device=self.device,
            )
            self.swis.append(swi)
        
        print(f"Loaded {len(self.models)} models for ensemble")
    
    def _load_model(self, config: Dict[str, Any]) -> nn.Module:
        """Load a single model from checkpoint."""
        checkpoint = torch.load(config['checkpoint_path'], map_location=self.device)
        
        # Get model args from checkpoint
        model_args = checkpoint.get('args', {})
        model_type = config.get('model_type', model_args.get('model', 'unet3d'))
        num_classes = config.get('num_classes', model_args.get('num_classes', 3))
        base_channels = config.get('base_channels', model_args.get('base_channels', 32))
        
        if model_type == 'unet3d':
            model = UNet3D(
                in_channels=1,
                num_classes=num_classes,
                base_channels=base_channels,
                deep_supervision=False,  # Disable for inference
            )
        elif model_type == 'attention_unet3d':
            model = AttentionUNet3D(
                in_channels=1,
                num_classes=num_classes,
                base_channels=base_channels,
                deep_supervision=False,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Loaded {model_type} from {config['checkpoint_path']}")
        print(f"  Validation Dice: {checkpoint.get('val_dice', 'N/A')}")
        
        return model
    
    def predict(
        self,
        volume: np.ndarray,
        num_classes: int = 3,
        use_tta: bool = True,
    ) -> np.ndarray:
        """
        Ensemble prediction with optional TTA.
        
        Returns:
            Averaged probability map from all models
        """
        all_probs = []
        
        for i, (model, swi) in enumerate(zip(self.models, self.swis)):
            if use_tta:
                probs = predict_with_tta(volume, swi, num_classes)
            else:
                probs = swi(volume, num_classes)
            all_probs.append(probs)
        
        return np.mean(all_probs, axis=0)


# ============================================================================
# Main Inference
# ============================================================================

def normalize_volume(image: np.ndarray) -> np.ndarray:
    """Normalize image using non-zero mean and std."""
    nonzero_mask = image > 0
    if nonzero_mask.any():
        mean = image[nonzero_mask].mean()
        std = image[nonzero_mask].std()
        if std > 0:
            image = (image - mean) / std
    return image


def load_volume(path: str) -> np.ndarray:
    """Load and preprocess volume."""
    vol = tifffile.imread(path)
    vol = vol.astype(np.float32)
    vol = normalize_volume(vol)
    vol = vol[np.newaxis, ..., np.newaxis]  # (1, D, H, W, 1)
    return vol


def inference_pipeline(
    volume: np.ndarray,
    ensemble: Optional[ModelEnsemble] = None,
    swi: Optional[SlidingWindowInference] = None,
    use_tta: bool = True,
    pp_version: str = 'v1',
    T_low: float = 0.40,
    T_high: float = 0.85,
    z_radius: int = 1,
    xy_radius: int = 1,
    dust_min_size: int = 150,
) -> np.ndarray:
    """
    Full inference pipeline.
    
    Args:
        volume: Input volume
        ensemble: Model ensemble (if None, use single model via swi)
        swi: Single model SWI (used if ensemble is None)
        use_tta: Whether to use TTA
        pp_version: Post-processing version ('v1' or 'v2')
        T_low: Low threshold for hysteresis
        T_high: High threshold for hysteresis
        z_radius: Z radius for closing
        xy_radius: XY radius for closing
        dust_min_size: Minimum object size
    
    Returns:
        Binary segmentation mask
    """
    # Get probabilities
    if ensemble is not None:
        probs = ensemble.predict(volume, use_tta=use_tta)
    elif swi is not None:
        if use_tta:
            probs = predict_with_tta(volume, swi)
        else:
            probs = swi(volume)
    else:
        raise ValueError("Either ensemble or swi must be provided")
    
    # Post-processing
    if pp_version == 'v1':
        mask = topo_postprocess(
            probs,
            T_low=T_low,
            T_high=T_high,
            z_radius=z_radius,
            xy_radius=xy_radius,
            dust_min_size=dust_min_size,
        )
    else:
        mask = topo_postprocess_v2(
            probs,
            T_low=T_low,
            T_high=T_high,
            z_radius=z_radius,
            xy_radius=xy_radius,
            dust_min_size=dust_min_size,
        )
    
    return mask


def main(args):
    """Main inference function."""
    set_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test CSV
    test_df = pd.read_csv(args.test_csv)
    print(f"Found {len(test_df)} test volumes")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model(s)
    if args.ensemble:
        # Load multiple models for ensemble
        model_configs = []
        for ckpt_path in args.checkpoints:
            model_configs.append({
                'checkpoint_path': ckpt_path,
                'roi_size': tuple(args.roi_size),
                'overlap': args.overlap,
                'batch_size': args.batch_size,
            })
        ensemble = ModelEnsemble(model_configs, device)
        swi = None
    else:
        # Single model
        checkpoint = torch.load(args.checkpoints[0], map_location=device)
        model_args = checkpoint.get('args', {})
        model_type = model_args.get('model', 'unet3d')
        
        if model_type == 'unet3d':
            model = UNet3D(
                in_channels=1,
                num_classes=args.num_classes,
                base_channels=model_args.get('base_channels', 32),
                deep_supervision=False,
            )
        else:
            model = AttentionUNet3D(
                in_channels=1,
                num_classes=args.num_classes,
                base_channels=model_args.get('base_channels', 32),
                deep_supervision=False,
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        swi = SlidingWindowInference(
            model,
            roi_size=tuple(args.roi_size),
            overlap=args.overlap,
            mode='gaussian',
            batch_size=args.batch_size,
            device=device,
        )
        ensemble = None
        
        print(f"Loaded model: {args.checkpoints[0]}")
        print(f"Validation Dice: {checkpoint.get('val_dice', 'N/A')}")
    
    # Process each test volume
    results = []
    
    with zipfile.ZipFile(
        args.output_dir + '/submission.zip', 'w', compression=zipfile.ZIP_DEFLATED
    ) as z:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Processing'):
            image_id = row['id']
            tif_path = f"{args.test_images}/{image_id}.tif"
            
            # Load and preprocess
            volume = load_volume(tif_path)
            
            # Inference
            mask = inference_pipeline(
                volume,
                ensemble=ensemble,
                swi=swi,
                use_tta=args.use_tta,
                pp_version=args.pp_version,
                T_low=args.t_low,
                T_high=args.t_high,
                z_radius=args.z_radius,
                xy_radius=args.xy_radius,
                dust_min_size=args.dust_min_size,
            )
            
            # Save mask
            out_path = output_dir / f"{image_id}.tif"
            tifffile.imwrite(str(out_path), mask.astype(np.uint8))
            
            # Add to zip
            z.write(str(out_path), arcname=f"{image_id}.tif")
            
            # Clean up
            os.remove(str(out_path))
            
            # Stats
            results.append({
                'id': image_id,
                'surface_pixels': mask.sum(),
                'total_pixels': mask.size,
                'surface_ratio': mask.sum() / mask.size,
            })
    
    # Save results summary
    pd.DataFrame(results).to_csv(output_dir / 'inference_stats.csv', index=False)
    
    print(f"\nInference completed!")
    print(f"Submission saved to: {args.output_dir}/submission.zip")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vesuvius Challenge Inference')
    
    # Paths
    parser.add_argument('--test-images', type=str, default='test_images',
                        help='Path to test images directory')
    parser.add_argument('--test-csv', type=str, default='test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Path(s) to model checkpoint(s)')
    
    # Model settings
    parser.add_argument('--num-classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--roi-size', type=int, nargs=3, default=[160, 160, 160],
                        help='ROI size for sliding window')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap for sliding window')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use model ensemble')
    
    # TTA settings
    parser.add_argument('--use-tta', action='store_true', default=True,
                        help='Use test-time augmentation')
    parser.add_argument('--no-tta', action='store_false', dest='use_tta',
                        help='Disable test-time augmentation')
    
    # Post-processing settings
    parser.add_argument('--pp-version', type=str, default='v1',
                        choices=['v1', 'v2'],
                        help='Post-processing version')
    parser.add_argument('--t-low', type=float, default=0.40,
                        help='Low threshold for hysteresis')
    parser.add_argument('--t-high', type=float, default=0.85,
                        help='High threshold for hysteresis')
    parser.add_argument('--z-radius', type=int, default=1,
                        help='Z radius for morphological closing')
    parser.add_argument('--xy-radius', type=int, default=1,
                        help='XY radius for morphological closing')
    parser.add_argument('--dust-min-size', type=int, default=150,
                        help='Minimum object size (smaller removed)')
    
    args = parser.parse_args()
    
    main(args)
