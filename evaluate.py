#!/usr/bin/env python3
"""Main entry point for evaluating the Occupancy Flow Prediction model.

Usage:
    python evaluate.py [--weights-path PATH] [--model-path PATH]

This script:
1. Loads the trained model from checkpoints
2. Creates test data iterator
3. Evaluates model performance
4. Reports loss metrics
"""

import argparse

import tensorflow as tf
from waymo_open_dataset.utils import occupancy_flow_data

from config.config import ModelConfig, PathConfig, TaskConfig
from data.data_loader import get_or_count_batches
from evaluators.evaluator import OccupancyFlowEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Occupancy Flow Prediction model"
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Path to model weights file (default: from config)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to full model file (alternative to weights-path)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config)",
    )
    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()

    print("=" * 60)
    print("Waymo Occupancy Flow Prediction - Evaluation")
    print("=" * 60)

    # Load configurations
    task_config = TaskConfig().get()
    model_config = ModelConfig()
    path_config = PathConfig()

    # Override with command line arguments if provided
    if args.batch_size is not None:
        model_config.BATCH_SIZE = args.batch_size

    print(f"Configuration:")
    print(f"  Encoder: {model_config.ENCODER}")
    print(f"  Batch size: {model_config.BATCH_SIZE}")

    # Create test data iterator only (no train/val needed for evaluation)
    print("\nLoading test dataset...")
    test_filenames = tf.io.matching_files(path_config.TEST_FILES)
    test_dataset = tf.data.TFRecordDataset(test_filenames)
    test_dataset = test_dataset.map(occupancy_flow_data.parse_tf_example)
    test_dataset = test_dataset.batch(model_config.BATCH_SIZE)
    test_it = iter(test_dataset)
    print("Test dataset loaded")

    # Get batch counts (uses file if exists, otherwise counts from datasets)
    print("Getting batch counts...")
    _, _, num_test_batches = get_or_count_batches(
        train_files=path_config.TRAIN_FILES,
        val_files=path_config.VAL_FILES,
        test_files=path_config.TEST_FILES,
        batch_size=model_config.BATCH_SIZE,
        batch_count_file=path_config.BATCH_COUNT_FILE,
    )
    print(f"Test batches: {num_test_batches}")

    # Initialize evaluator
    evaluator = OccupancyFlowEvaluator(
        weights_path=args.weights_path,
        model_path=args.model_path,
        config=task_config,
        model_config=model_config,
        path_config=path_config,
    )

    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.run_evaluation(test_it, num_test_batches)

    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
