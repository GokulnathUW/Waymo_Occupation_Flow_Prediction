from __future__ import annotations

"""
Training loop for Occupancy Flow Prediction model.
Implements the training pipeline with:
- Single-step training with gradient computation
- Epoch-based training with validation
- Loss tracking and model checkpointing
"""

import os
from typing import Optional

import tensorflow as tf
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2

from config.config import ModelConfig, PathConfig, TaskConfig
from data.data_loader import create_data_iterators
from models.resnet_encoder import create_occupancy_flow_model, get_pred_waypoint_logits
from utils.loss_functions import occupancy_flow_loss
from utils.preprocessing import create_model_inputs, preprocess_inputs


class OccupancyFlowTrainer:
    """Manages model training with validation and checkpointing."""

    def __init__(
        self,
        config: Optional[occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig] = None,
        model_config: Optional[ModelConfig] = None,
        path_config: Optional[PathConfig] = None,
    ):
        """Initialize trainer with configuration"""
        self.config = config or TaskConfig().get()
        self.model_config = model_config or ModelConfig()
        self.path_config = path_config or PathConfig()

        self.model = None
        self.optimizer = None

    def _build_model(self, sample_model_inputs):
        """Build and initialize the model with sample inputs"""
        input_shape = sample_model_inputs.shape[1:]
        self.model = create_occupancy_flow_model(
            config=self.config,
            num_pred_channels=self.model_config.NUM_PRED_CHANNELS,
            input_shape=input_shape,
            encoder_name=self.model_config.ENCODER,
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_config.LEARNING_RATE
        )

        return self.model

    def train_one_step(self, inputs):
        """Performs a single training step with gradient update"""
        with tf.GradientTape() as tape:
            # Preprocess inputs
            timestep_grids, true_waypoints, vis_grids = preprocess_inputs(
                inputs, self.config
            )

            # Prepare model inputs
            model_inputs = create_model_inputs(timestep_grids, vis_grids)

            # Forward pass
            model_outputs = self.model(model_inputs, training=True)
            pred_waypoint_logits = get_pred_waypoint_logits(
                model_outputs, self.config, self.model_config.NUM_PRED_CHANNELS
            )

            # Compute loss
            loss_dict = occupancy_flow_loss(
                config=self.config,
                true_waypoints=true_waypoints,
                pred_waypoint_logits=pred_waypoint_logits,
            )
            total_loss = tf.math.add_n(loss_dict.values())

        # Compute and apply gradients (filter out None gradients)
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        grads_and_vars = [
            (g, v)
            for g, v in zip(grads, self.model.trainable_weights)
            if g is not None
        ]

        if not grads_and_vars:
            raise ValueError(
                "No gradients computed! Check if the model is connected "
                "to the loss computation graph."
            )

        self.optimizer.apply_gradients(grads_and_vars)

        return total_loss, loss_dict

    def evaluate(self, data_iterator, num_batches):
        """Evaluates model on a dataset without gradient computation"""
        total_loss = 0.0
        total_loss_dict = {"observed_xe": 0.0, "occluded_xe": 0.0, "flow": 0.0}

        for _ in range(num_batches):
            inputs = next(data_iterator)

            # Preprocess inputs
            timestep_grids, true_waypoints, vis_grids = preprocess_inputs(
                inputs, self.config
            )

            # Forward pass
            model_inputs = create_model_inputs(timestep_grids, vis_grids)
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
            batch_loss = tf.math.add_n(loss_dict.values())
            total_loss += batch_loss

            # Accumulate individual losses
            for key in total_loss_dict:
                total_loss_dict[key] += loss_dict[key]

        # Average over batches
        total_loss /= num_batches
        for key in total_loss_dict:
            total_loss_dict[key] /= num_batches

        return total_loss, total_loss_dict

    def write_model_summary(self, output_file = None):
        """Writes detailed model summary to file"""
        if self.model is None:
            raise RuntimeError(
                "Model has not been built yet. Call _build_model() first."
            )

        if output_file is None:
            output_file = f"{self.model_config.ENCODER}_summary.txt"

        with open(output_file, "w") as f:
            f.write(f"{'=' * 100}\n")
            f.write("Model Architecture Summary\n")
            f.write(f"{'=' * 100}\n\n")

            # Model architecture
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

            f.write(f"\n{'=' * 100}\n")
            f.write("Model Variables\n")
            f.write(f"{'=' * 100}\n")
            for var in self.model.trainable_variables:
                f.write(f"  {var.name}: {var.shape}\n")
            f.write(f"{'=' * 100}\n")

        print(f"Model summary written to {output_file}")

    def train(self, num_train_batches, num_val_batches, train_iterator, val_iterator, epochs = None):
        """Full training loop with validation and checkpointing"""
        if epochs is None:
            epochs = self.model_config.TRAIN_EPOCHS

        print(f"Starting training for {epochs} epochs")
        print(f"Training batches: {num_train_batches}")
        print(f"Validation batches: {num_val_batches}")

        for epoch in range(epochs):
            # Training phase
            for batch in range(num_train_batches):
                train_inputs = next(train_iterator)
                train_total_loss, train_loss_dict = self.train_one_step(
                    train_inputs
                )

                if batch % 10 == 0:  # Log every 10 batches
                    print(
                        f"Epoch {epoch}/{epochs} | "
                        f"Batch {batch}/{num_train_batches} | "
                        f"Train loss: {float(train_total_loss):.4f}"
                    )

            # Validation phase
            val_total_loss, val_loss_dict = self.evaluate(
                val_iterator, num_val_batches
            )
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Validation loss: {float(val_total_loss):.4f} | "
                f"  observed_xe: {val_loss_dict['observed_xe']:.4f} | "
                f"  occluded_xe: {val_loss_dict['occluded_xe']:.4f} | "
                f"  flow: {val_loss_dict['flow']:.4f}"
            )

        # Save model checkpoints
        self.save_checkpoints()

        # Write model summary
        self.write_model_summary()

    def save_checkpoints(self):
        """Save model weights and architecture to disk."""
        weights_path = self.path_config.get_weights_path(
            self.model_config.ENCODER
        )
        model_path = self.path_config.get_model_path(self.model_config.ENCODER)

        self.model.save_weights(weights_path)
        self.model.save(model_path)

        print(f"Model weights saved to {weights_path}")
        print(f"Model architecture saved to {model_path}")


def main():
    """Main training entry point."""
    print("Beginning training")

    # Load configurations
    task_config  = TaskConfig().get()
    model_config = ModelConfig()
    path_config  = PathConfig()

    # Create data iterators
    train_it, val_it, test_it = create_data_iterators(
        train_files=path_config.TRAIN_FILES,
        val_files=path_config.VAL_FILES,
        test_files=path_config.TEST_FILES,
        batch_size=model_config.BATCH_SIZE,
    )

    # Get batch counts (from file or recount)
    from data.data_loader import get_or_count_batches

    num_train_batches, num_val_batches, num_test_batches = (
        get_or_count_batches(
            train_files=path_config.TRAIN_FILES,
            val_files=path_config.VAL_FILES,
            test_files=path_config.TEST_FILES,
            batch_size=model_config.BATCH_SIZE,
            batch_count_file=path_config.BATCH_COUNT_FILE,
        )
    )

    # Get sample batch for model initialization
    sample_inputs = next(test_it)
    sample_timestep_grids, sample_true_waypoints, sample_vis_grids = (
        preprocess_inputs(sample_inputs, task_config)
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

    # Run training
    trainer.train(
        num_train_batches=num_train_batches,
        num_val_batches=num_val_batches,
        train_iterator=train_it,
        val_iterator=val_it,
        epochs=model_config.TRAIN_EPOCHS,
    )

    print("Model training completed successfully")


if __name__ == "__main__":
    main()
