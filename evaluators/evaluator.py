from __future__ import annotations

"""
Evaluation and testing for Occupancy Flow Prediction model.
Implements model evaluation pipeline:
- Load trained model from checkpoints
- Evaluate on test dataset
- Report loss metrics
"""

import os
from typing import Optional

import tensorflow as tf
import tensorflow_graphics.image.transformer as tfg_transformer
from waymo_open_dataset.utils import (
    occupancy_flow_data,
    occupancy_flow_grids,
    occupancy_flow_metrics,
    occupancy_flow_renderer,
    occupancy_flow_vis,
)
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2

from config.config import ModelConfig, PathConfig, TaskConfig
from data.data_loader import create_data_iterators, get_or_count_batches
from models.resnet_encoder import create_occupancy_flow_model, get_pred_waypoint_logits
from utils.loss_functions import occupancy_flow_loss
from utils.preprocessing import create_model_inputs, preprocess_inputs


class OccupancyFlowEvaluator:
    """Manages model evaluation on test/validation datasets."""

    def __init__(
        self,
        weights_path: str = None,
        model_path: str = None,
        config: Optional[occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig] = None,
        model_config: Optional[ModelConfig] = None,
        path_config: Optional[PathConfig] = None,
    ):
        """Initialize evaluator with configuration and model paths"""
        self.config       = config or TaskConfig().get()
        self.model_config = model_config or ModelConfig()
        self.path_config  = path_config or PathConfig()

        self.weights_path = weights_path or self.path_config.get_weights_path(
            self.model_config.ENCODER
        )
        self.model_path = model_path or self.path_config.get_model_path(
            self.model_config.ENCODER
        )

        self.model = None

    def load_model(self, input_shape):
        """Load model architecture and weights"""
        # Create model architecture
        self.model = create_occupancy_flow_model(
            config=self.config,
            num_pred_channels=self.model_config.NUM_PRED_CHANNELS,
            input_shape=input_shape,
            encoder_name=self.model_config.ENCODER,
        )

        # Load weights
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print(f"Loaded weights from {self.weights_path}")
        else:
            print(f"Warning: Weights file not found at {self.weights_path}")

    def evaluate(self, data_iterator, num_batches):
        """Evaluate model on a dataset"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        total_loss      = 0.0
        total_loss_dict = {"observed_xe": 0.0, "occluded_xe": 0.0, "flow": 0.0}

        for batch_idx in range(num_batches):
            inputs = next(data_iterator)

            # Preprocess inputs
            timestep_grids, true_waypoints, vis_grids = preprocess_inputs(inputs, self.config)

            # Forward pass
            model_inputs  = create_model_inputs(timestep_grids, vis_grids)
            model_outputs = self.model(model_inputs, training=False)

            pred_waypoint_logits = get_pred_waypoint_logits(
                model_outputs, self.config, self.model_config.NUM_PRED_CHANNELS
            )

            # Compute loss
            loss_dict = occupancy_flow_loss(
                config=self.config,
                true_waypoints=true_waypoints,
                pred_waypoint_logits=pred_waypoint_logits,
            )
            batch_loss  = tf.math.add_n(loss_dict.values())
            total_loss += batch_loss

            # Accumulate individual losses
            for key in total_loss_dict:
                total_loss_dict[key] += loss_dict[key]

            if (batch_idx + 1) % 50 == 0:
                print(f"Evaluated {batch_idx + 1}/{num_batches} batches")

        # Average over batches
        total_loss /= num_batches
        for key in total_loss_dict:
            total_loss_dict[key] /= num_batches

        return total_loss, total_loss_dict

    def run_evaluation(self, test_iterator, num_test_batches):
        """Full evaluation pipeline with reporting"""
        # Get sample batch for input shape WITHOUT consuming the main iterator
        # We create a separate dataset for sampling to avoid exhaustion
        print("Building model from sample input...")
        sample_dataset  = self._create_sample_dataset()
        sample_iterator = iter(sample_dataset)
        sample_inputs   = next(sample_iterator)

        sample_timestep_grids, _, sample_vis_grids = preprocess_inputs(
            sample_inputs, self.config
        )
        sample_model_inputs = create_model_inputs(
            sample_timestep_grids, sample_vis_grids
        )
        input_shape = sample_model_inputs.shape[1:]

        # Load model
        self.load_model(input_shape)

        # Run evaluation on the original iterator (all batches)
        print(f"Evaluating on {num_test_batches} batches...")
        total_loss, loss_dict = self.evaluate(test_iterator, num_test_batches)

        # Report results
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Total Loss: {float(total_loss):.4f}")
        print(f"  Observed Occupancy CE: {loss_dict['observed_xe']:.4f}")
        print(f"  Occluded Occupancy CE: {loss_dict['occluded_xe']:.4f}")
        print(f"  Flow Loss: {loss_dict['flow']:.4f}")
        print("=" * 60)

        return {
            "total_loss": float(total_loss),
            "observed_xe": float(loss_dict["observed_xe"]),
            "occluded_xe": float(loss_dict["occluded_xe"]),
            "flow": float(loss_dict["flow"]),
        }

    def _create_sample_dataset(self) -> tf.data.Dataset:
        """Create a small dataset for sampling input shape"""
        from waymo_open_dataset.utils import occupancy_flow_data

        filenames = tf.io.matching_files(self.path_config.TEST_FILES)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(occupancy_flow_data.parse_tf_example)
        dataset = dataset.batch(self.model_config.BATCH_SIZE)
        dataset = dataset.take(1)  # Only need one batch
        return dataset


def main():
    """Main evaluation entry point."""
    print("Beginning evaluation")

    # Load configurations
    task_config  = TaskConfig().get()
    model_config = ModelConfig()
    path_config  = PathConfig()

    # Create test data iterator
    _, _, test_it = create_data_iterators(
        train_files=path_config.TRAIN_FILES,
        val_files=path_config.VAL_FILES,
        test_files=path_config.TEST_FILES,
        batch_size=model_config.BATCH_SIZE,
    )

    # Get batch counts
    _, _, num_test_batches = get_or_count_batches(
        None, None, test_it, path_config.BATCH_COUNT_FILE
    )

    # Initialize evaluator
    evaluator = OccupancyFlowEvaluator(
        config=task_config,
        model_config=model_config,
        path_config=path_config,
    )

    # Run evaluation
    results = evaluator.run_evaluation(test_it, num_test_batches)

    print("Evaluation completed successfully")
    return results


if __name__ == "__main__":
    main()
