# Vesuvius Challenge Surface Detection - Improved Pipeline

This repository contains an improved training and inference pipeline for the Vesuvius Challenge Surface Detection competition, targeting a leaderboard score of **0.65+** (up from 0.54).

## Key Improvements Over Original

| Feature | Original | Improved |
|---------|----------|----------|
| Model | TransUNet (70M params) | Attention UNet3D + UNet3D ensemble |
| Loss Function | Combo Loss | Dice + CE + Focal (weighted) |
| Deep Supervision | No | Yes (auxiliary outputs) |
| Data Augmentation | Basic | Comprehensive 3D augmentations |
| TTA | 7 augmentations | Same + ensemble averaging |
| Post-processing | T_low=0.5, T_high=0.9 | T_low=0.4, T_high=0.85, +closing |
| Model Ensemble | No | Yes (4 models) |

## Directory Structure

Your working directory should look like this:
```
surfaceX/
├── train_images/        # Training CT scan volumes (.tif)
├── train_labels/        # Training segmentation masks (.tif)
├── test_images/         # Test CT scan volumes (.tif)
├── train.csv            # Training metadata
├── test.csv             # Test metadata
├── train.py             # Training script
├── inference.py         # Inference script
├── run_training.sh      # Training automation script
├── run_inference.sh     # Inference automation script
└── requirements.txt     # Python dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify GPU is available:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## Training

### Quick Start (Full Pipeline)

Run all training jobs sequentially:
```bash
chmod +x run_training.sh
./run_training.sh
```

This trains 4 models for ensemble:
- Attention UNet3D (seed 42)
- Attention UNet3D (seed 123)
- UNet3D (seed 42)
- UNet3D (seed 456, larger)

### Custom Training

For single model training with custom parameters:
```bash
python train.py \
    --train-images train_images \
    --train-labels train_labels \
    --train-csv train.csv \
    --output-dir checkpoints/my_model \
    --model attention_unet3d \
    --epochs 100 \
    --batch-size 2 \
    --base-channels 32 \
    --lr 1e-4 \
    --deep-supervision
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | attention_unet3d | Model type: `unet3d` or `attention_unet3d` |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 2 | Batch size (reduce if OOM) |
| `--base-channels` | 32 | Base channel count (increase for larger model) |
| `--lr` | 1e-4 | Learning rate |
| `--crop-size` | 160 160 160 | Training patch size |
| `--num-crops` | 4 | Crops per volume per epoch |
| `--dropout` | 0.1 | Dropout rate |
| `--deep-supervision` | True | Use deep supervision |
| `--seed` | 42 | Random seed |

## Inference

### Quick Start (Ensemble)

After training, run inference:
```bash
chmod +x run_inference.sh
./run_inference.sh
```

### Custom Inference

For single model:
```bash
python inference.py \
    --test-images test_images \
    --test-csv test.csv \
    --output-dir output \
    --checkpoints checkpoints/my_model/best_model_attention_unet3d.pth \
    --use-tta
```

For ensemble:
```bash
python inference.py \
    --test-images test_images \
    --test-csv test.csv \
    --output-dir output \
    --checkpoints \
        checkpoints/attention_unet_seed42/best_model_attention_unet3d.pth \
        checkpoints/attention_unet_seed123/best_model_attention_unet3d.pth \
        checkpoints/unet3d_seed42/best_model_unet3d.pth \
    --ensemble \
    --use-tta
```

### Post-Processing Tuning

The post-processing parameters significantly affect the final score:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--t-low` | 0.40 | Lower threshold for hysteresis (lower = more recall) |
| `--t-high` | 0.85 | Upper threshold (lower = more recall) |
| `--z-radius` | 1 | Z-axis closing radius |
| `--xy-radius` | 1 | XY-plane closing radius |
| `--dust-min-size` | 150 | Remove objects smaller than this |

**Tip**: If your score is too low, try:
- Decreasing `--t-low` (e.g., 0.35)
- Decreasing `--t-high` (e.g., 0.80)
- Increasing `--z-radius` (e.g., 2)

## Expected Results

With the full ensemble and TTA:
- Training time: ~4-6 hours per model on T4/V100
- Expected validation Dice: 0.70-0.75
- Expected LB score: **0.62-0.68**

## Tips for Higher Scores

1. **More training data**: Use all available training volumes
2. **Longer training**: 150-200 epochs with early stopping
3. **Larger models**: Increase `--base-channels` to 48 or 64
4. **More ensemble members**: Train 5-6 models with different seeds
5. **Post-processing grid search**: Test different threshold combinations
6. **Cross-validation**: Implement k-fold CV for better model selection

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` to 1
- Reduce `--crop-size` to 128 128 128
- Reduce `--base-channels` to 24

### Slow Training
- Increase `--num-workers` (up to number of CPU cores)
- Reduce `--num-crops` to 2

### Poor Results
- Check if labels are correctly loaded (should be 0/1/2)
- Verify normalization is working
- Try different random seeds

## Files

- `train.py`: Main training script with models and losses
- `inference.py`: Inference script with TTA and ensemble
- `run_training.sh`: Automated training script
- `run_inference.sh`: Automated inference script
- `requirements.txt`: Python dependencies

## Competition Link

https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection
