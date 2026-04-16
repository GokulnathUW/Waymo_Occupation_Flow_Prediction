#!/usr/bin/env python3
"""Main entry point for training the Occupancy Flow Prediction model.

Usage:
    python train.py [--epochs N] [--batch-size N] [--encoder NAME]

This script:
1. Loads configuration from config files
2. Creates data iterators for training and validation
3. Initializes the model
4. Runs the training loop with periodic validation
5. Saves model checkpoints
"""

import argparse
import sys

from config.config import ModelConfig, PathConfig, TaskConfig
from data.data_loader import create_data_iterators, get_or_count_batches
from trainers.trainer import OccupancyFlowTrainer
from utils.preprocessing import create_model_inputs, preprocess_inputs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Occupancy Flow Prediction model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Encoder architecture name (default: from config)",
    )
    parser.add_argument(
        "--force-recount",
        action="store_true",
        help="Force recount of dataset batches",
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    print("=" * 60)
    print("Waymo Occupancy Flow Prediction - Training")
    print("=" * 60)

    # Load configurations
    task_config = TaskConfig().get()
    model_config = ModelConfig()
    path_config = PathConfig()

    # Override with command line arguments if provided
    if args.batch_size is not None:
        model_config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        model_config.TRAIN_EPOCHS = args.epochs
    if args.encoder is not None:
        model_config.ENCODER = args.encoder

    print(f"Configuration:")
    print(f"  Encoder: {model_config.ENCODER}")
    print(f"  Batch size: {model_config.BATCH_SIZE}")
    print(f"  Epochs: {model_config.TRAIN_EPOCHS}")
    print(f"  Learning rate: {model_config.LEARNING_RATE}")

    # Create data iterators
    print("\nLoading datasets...")
    train_it, val_it, test_it = create_data_iterators(
        train_files=path_config.TRAIN_FILES,
        val_files=path_config.VAL_FILES,
        test_files=path_config.TEST_FILES,
        batch_size=model_config.BATCH_SIZE,
    )
    print("Datasets loaded successfully")

    # Get batch counts (uses file if exists, otherwise counts from datasets)
    print("Getting batch counts...")
    num_train_batches, num_val_batches, num_test_batches = (
        get_or_count_batches(
            train_files=path_config.TRAIN_FILES,
            val_files=path_config.VAL_FILES,
            test_files=path_config.TEST_FILES,
            batch_size=model_config.BATCH_SIZE,
            batch_count_file=path_config.BATCH_COUNT_FILE,
            force_recount=args.force_recount,
        )
    )
    print(
        f"Training: {num_train_batches} batches, "
        f"Validation: {num_val_batches} batches, "
        f"Test: {num_test_batches} batches"
    )

    # Get sample batch for model initialization
    print("Initializing model...")
    sample_inputs = next(test_it)
    sample_timestep_grids, _, sample_vis_grids = preprocess_inputs(
        sample_inputs, task_config
    )
    sample_model_inputs = create_model_inputs(
        sample_timestep_grids, sample_vis_grids
    )

    # Initialize trainer
    trainer = OccupancyFlowTrainer(
        config=task_config,
        model_config=model_config,
        path_config=path_config,
    )

    # Build model with sample input shape
    trainer._build_model(sample_model_inputs)
    print(f"Model built with input shape: {sample_model_inputs.shape}")

    # Run training
    print("\nStarting training...")
    trainer.train(
        num_train_batches=num_train_batches,
        num_val_batches=num_val_batches,
        train_iterator=train_it,
        val_iterator=val_it,
        epochs=model_config.TRAIN_EPOCHS,
    )

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
