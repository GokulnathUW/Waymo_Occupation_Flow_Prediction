#!/usr/bin/env python3
"""Visualization script for Occupancy Flow Prediction data.

Usage:
    python visualize.py --output-dir ./visualizations [--num-samples N]

This script:
1. Loads sample data from the dataset
2. Generates trajectory visualizations for all agents
3. Creates animations from visualization frames
4. Saves visualizations to output directory
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import occupancy_flow_data

from config.config import PathConfig
from viz.visualization import create_animation, visualize_all_agents_smooth


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Occupancy Flow Prediction data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for data loading",
    )
    parser.add_argument(
        "--create-animation",
        action="store_true",
        help="Create MP4 animations instead of individual frames",
    )
    return parser.parse_args()


def save_frame(image: np.ndarray, filepath: str) -> None:
    """Save a single frame as an image.

    Args:
        image: RGB image as numpy array.
        filepath: Path to save the image.
    """
    plt.imsave(filepath, image)


def save_animation(anim, filepath: str) -> None:
    """Save animation as MP4 video.

    Args:
        anim: Matplotlib animation object.
        filepath: Path to save the video.
    """
    anim.save(filepath, writer="ffmpeg", fps=10)


def main():
    """Main visualization entry point."""
    args = parse_args()

    print("=" * 60)
    print("Waymo Occupancy Flow Prediction - Visualization")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Load dataset
    path_config = PathConfig()
    print("\nLoading dataset...")

    # Use test files for visualization
    filenames = tf.io.matching_files(path_config.TEST_FILES)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.take(args.num_samples)

    iterator = iter(dataset)
    print(f"Dataset loaded. Visualizing {args.num_samples} samples...")

    for sample_idx in range(args.num_samples):
        print(f"\nProcessing sample {sample_idx + 1}/{args.num_samples}")

        try:
            inputs = next(iterator)
        except StopIteration:
            print("No more samples available")
            break

        # Extract first example from batch
        inputs_no_batch = {k: v[0] for k, v in inputs.items()}

        # Generate visualization frames
        images = visualize_all_agents_smooth(inputs_no_batch)
        print(f"  Generated {len(images)} frames")

        sample_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        if args.create_animation:
            # Create animation (sample every 5 frames for smoother viewing)
            anim = create_animation(images[::5], interval=100)
            anim_path = os.path.join(sample_dir, "trajectory.mp4")
            save_animation(anim, anim_path)
            print(f"  Animation saved to {anim_path}")
        else:
            # Save individual frames
            for frame_idx, image in enumerate(images):
                frame_path = os.path.join(
                    sample_dir, f"frame_{frame_idx:04d}.png"
                )
                save_frame(image, frame_path)
            print(f"  Frames saved to {sample_dir}/")

    print("\n" + "=" * 60)
    print("Visualization completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
