#!/usr/bin/env bash
# Deployment script for Waymo Occupancy Flow Prediction
# Runs training followed by evaluation

set -e  # Exit on error

echo "============================================"
echo "Waymo Occupancy Flow Prediction - Deployment"
echo "============================================"

# Extract input data if compressed
if [ -f "waymo_open_dataset.tar.gz" ]; then
    echo "Extracting dataset..."
    tar -xzvf waymo_open_dataset.tar.gz
fi

# Run training
echo ""
echo "Starting model training..."
python3 train.py > flow_prediction.out 2>&1
echo "Model training completed. See flow_prediction.out for details."

# Run evaluation
echo ""
echo "Running evaluation..."
python3 evaluate.py
echo "Evaluation completed."

# Archive output files
echo ""
echo "Archiving output files..."
if [ -f "model_weights_ResNet50V2.h5" ] && [ -f "model_ResNet50V2.h5" ]; then
    tar -czf output_models.tar.gz \
        model_weights_ResNet50V2.h5 \
        model_ResNet50V2.h5 \
        batch_count.txt \
        flow_prediction.out
    echo "Output files archived to output_models.tar.gz"
else
    echo "Warning: Model checkpoints not found. Skipping archive."
fi

echo ""
echo "Deployment completed successfully!"