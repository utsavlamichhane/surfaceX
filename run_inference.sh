#!/bin/bash
# Vesuvius Challenge Surface Detection - Inference Script
# Run this from your surfaceX directory

set -e

# Configuration
TEST_IMAGES="test_images"
TEST_CSV="test.csv"
OUTPUT_DIR="output"
CHECKPOINT_DIR="checkpoints"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Vesuvius Challenge - Inference Pipeline"
echo "=============================================="

# Check available models
echo "Looking for trained models in $CHECKPOINT_DIR..."

# Collect all available checkpoints
CHECKPOINTS=""
if [ -f "${CHECKPOINT_DIR}/attention_unet_seed42/best_model_attention_unet3d.pth" ]; then
    CHECKPOINTS="$CHECKPOINTS ${CHECKPOINT_DIR}/attention_unet_seed42/best_model_attention_unet3d.pth"
    echo "  Found: attention_unet_seed42"
fi
if [ -f "${CHECKPOINT_DIR}/attention_unet_seed123/best_model_attention_unet3d.pth" ]; then
    CHECKPOINTS="$CHECKPOINTS ${CHECKPOINT_DIR}/attention_unet_seed123/best_model_attention_unet3d.pth"
    echo "  Found: attention_unet_seed123"
fi
if [ -f "${CHECKPOINT_DIR}/unet3d_seed42/best_model_unet3d.pth" ]; then
    CHECKPOINTS="$CHECKPOINTS ${CHECKPOINT_DIR}/unet3d_seed42/best_model_unet3d.pth"
    echo "  Found: unet3d_seed42"
fi
if [ -f "${CHECKPOINT_DIR}/unet3d_seed456/best_model_unet3d.pth" ]; then
    CHECKPOINTS="$CHECKPOINTS ${CHECKPOINT_DIR}/unet3d_seed456/best_model_unet3d.pth"
    echo "  Found: unet3d_seed456"
fi

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No trained models found in $CHECKPOINT_DIR"
    echo "Please run training first with: bash run_training.sh"
    exit 1
fi

# Count number of models
NUM_MODELS=$(echo $CHECKPOINTS | wc -w)
echo ""
echo "Found $NUM_MODELS model(s)"

# Run inference
if [ $NUM_MODELS -gt 1 ]; then
    echo "Running ensemble inference with TTA..."
    python3 inference.py \
        --test-images $TEST_IMAGES \
        --test-csv $TEST_CSV \
        --output-dir $OUTPUT_DIR \
        --checkpoints $CHECKPOINTS \
        --ensemble \
        --use-tta \
        --t-low 0.40 \
        --t-high 0.85 \
        --z-radius 1 \
        --xy-radius 1 \
        --dust-min-size 150
else
    echo "Running single model inference with TTA..."
    python3 inference.py \
        --test-images $TEST_IMAGES \
        --test-csv $TEST_CSV \
        --output-dir $OUTPUT_DIR \
        --checkpoints $CHECKPOINTS \
        --use-tta \
        --t-low 0.40 \
        --t-high 0.85 \
        --z-radius 1 \
        --xy-radius 1 \
        --dust-min-size 150
fi

echo ""
echo "=============================================="
echo "Inference completed!"
echo "Submission file: $OUTPUT_DIR/submission.zip"
echo "=============================================="
