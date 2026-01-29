#!/bin/bash
# Vesuvius Challenge Surface Detection - Training Script
# Run this from your surfaceX directory

set -e

# Configuration
EPOCHS=200
BATCH_SIZE=2
NUM_WORKERS=4
BASE_CHANNELS=64
LR=1e-4

# Directory structure (adjust if needed)
TRAIN_IMAGES="train_images"
TRAIN_LABELS="train_labels"
TRAIN_CSV="train.csv"
OUTPUT_DIR="checkpoints"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Vesuvius Challenge - Training Pipeline"
echo "=============================================="
echo "Train Images: $TRAIN_IMAGES"
echo "Train Labels: $TRAIN_LABELS"
echo "Train CSV: $TRAIN_CSV"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Check if GPU is available
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Strategy 1: Train Attention UNet with different seeds for ensemble
echo ""
echo "=== Training Attention UNet (Seed 42) ==="
python3 train.py \
    --train-images $TRAIN_IMAGES \
    --train-labels $TRAIN_LABELS \
    --train-csv $TRAIN_CSV \
    --output-dir ${OUTPUT_DIR}/attention_unet_seed42 \
    --model attention_unet3d \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --base-channels $BASE_CHANNELS \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --seed 42 \
    --deep-supervision

echo ""
echo "=== Training Attention UNet (Seed 123) ==="
python3 train.py \
    --train-images $TRAIN_IMAGES \
    --train-labels $TRAIN_LABELS \
    --train-csv $TRAIN_CSV \
    --output-dir ${OUTPUT_DIR}/attention_unet_seed123 \
    --model attention_unet3d \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --base-channels $BASE_CHANNELS \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --seed 123 \
    --deep-supervision

echo ""
echo "=== Training UNet3D (Seed 42) ==="
python3 train.py \
    --train-images $TRAIN_IMAGES \
    --train-labels $TRAIN_LABELS \
    --train-csv $TRAIN_CSV \
    --output-dir ${OUTPUT_DIR}/unet3d_seed42 \
    --model unet3d \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --base-channels $BASE_CHANNELS \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --seed 42 \
    --deep-supervision

echo ""
echo "=== Training UNet3D (Seed 456) ==="
python3 train.py \
    --train-images $TRAIN_IMAGES \
    --train-labels $TRAIN_LABELS \
    --train-csv $TRAIN_CSV \
    --output-dir ${OUTPUT_DIR}/unet3d_seed456 \
    --model unet3d \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --base-channels 48 \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --seed 456 \
    --deep-supervision

echo ""
echo "=============================================="
echo "Training completed!"
echo "Models saved to: $OUTPUT_DIR"
echo "=============================================="
echo ""
echo "To run inference with ensemble:"
echo "python3 inference.py \\"
echo "    --test-images test_images \\"
echo "    --test-csv test.csv \\"
echo "    --output-dir output \\"
echo "    --checkpoints ${OUTPUT_DIR}/attention_unet_seed42/best_model_attention_unet3d.pth \\"
echo "                  ${OUTPUT_DIR}/attention_unet_seed123/best_model_attention_unet3d.pth \\"
echo "                  ${OUTPUT_DIR}/unet3d_seed42/best_model_unet3d.pth \\"
echo "                  ${OUTPUT_DIR}/unet3d_seed456/best_model_unet3d.pth \\"
echo "    --ensemble \\"
echo "    --use-tta"
