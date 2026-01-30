#!/bin/bash
# Vesuvius Challenge Surface Detection - Inference Script
# Run this from your surfaceX directory

set -e

# ============================================
# IMPORTANT: Prioritize user-installed packages
# This fixes conflicts with system packages on HPC
# ============================================
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"

# Configuration
TEST_IMAGES="test_images"
TEST_CSV="test.csv"

# Intermediate files go to scratch
SCRATCH_DIR="/scratch/$USER/vesuvius"
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints"
TEMP_OUTPUT_DIR="${SCRATCH_DIR}/output"

# Final submission goes to current directory
FINAL_OUTPUT_DIR="."

# Create directories
mkdir -p $TEMP_OUTPUT_DIR

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

# Run inference (output to scratch first)
echo "Intermediate files: $TEMP_OUTPUT_DIR"
echo "Final submission: $FINAL_OUTPUT_DIR/submission.zip"
echo ""

if [ $NUM_MODELS -gt 1 ]; then
    echo "Running ensemble inference with TTA..."
    python3 inference.py \
        --test-images $TEST_IMAGES \
        --test-csv $TEST_CSV \
        --output-dir $TEMP_OUTPUT_DIR \
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
        --output-dir $TEMP_OUTPUT_DIR \
        --checkpoints $CHECKPOINTS \
        --use-tta \
        --t-low 0.40 \
        --t-high 0.85 \
        --z-radius 1 \
        --xy-radius 1 \
        --dust-min-size 150
fi

# Copy final submission to current directory
echo ""
echo "Copying submission to current directory..."
cp "${TEMP_OUTPUT_DIR}/submission.zip" "${FINAL_OUTPUT_DIR}/submission.zip"

echo ""
echo "=============================================="
echo "Inference completed!"
echo "Submission file: ${FINAL_OUTPUT_DIR}/submission.zip"
echo "Intermediate files: ${TEMP_OUTPUT_DIR}"
echo "=============================================="
