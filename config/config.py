"""
Configuration management for Waymo Occupation Flow Prediction.
Provides centralized configuration handling:
- TaskConfig: Loads Waymo task-specific parameters from protobuf config files
- ModelConfig: Model architecture and training hyperparameters
- PathConfig: Dataset paths and checkpoint file locations
"""

import os
from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2


class TaskConfig:
    """
    Manages the occupancy flow task configuration.
    Loads configuration from a protobuf text format file that specifies
    grid dimensions, temporal parameters, and normalization settings.
    """

    DEFAULT_CONFIG_FILE = "config/task_config.txt"

    def __init__(self, config_file: str = None):
        """Initialize task configuration"""
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config = self._load_config()

    def _load_config(self) -> occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig:
        """Load configuration from protobuf text file"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file, "r") as f:
            config_text = f.read()

        config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        text_format.Parse(config_text, config)
        return config

    def get(self) -> occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig:
        """Get the loaded configuration"""
        return self.config


class ModelConfig:
    """
    Model-specific configuration parameters.
    Contains architecture choices, hyperparameters, and training settings.
    """

    # Number of channels output by the model per waypoint:
    # - Occupancy of currently-observed vehicles: 1 channel
    # - Occupancy of currently-occluded vehicles: 1 channel
    # - Flow of all vehicles: 2 channels (dx, dy)
    NUM_PRED_CHANNELS = 4

    # Encoder architecture
    ENCODER = "ResNet50V2"

    # Training hyperparameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE    = 16
    TRAIN_EPOCHS  = 15


class PathConfig:
    """
    Dataset and file path configuration.
    Centralizes all file paths used in the project for easy modification.
    """

    # Dataset root directory
    DATASET_FOLDER = "waymo_open_dataset"

    # TFRecord file patterns for different data splits
    TRAIN_FILES = f"{DATASET_FOLDER}/training/training_tfexample.tfrecord*"
    VAL_FILES   = f"{DATASET_FOLDER}/validation/validation_tfexample.tfrecord*"
    TEST_FILES  = f"{DATASET_FOLDER}/test/testing_tfexample.tfrecord*"

    # Model checkpoint patterns
    MODEL_WEIGHTS_FILE = "model_weights_{encoder}.h5"
    MODEL_FILE         = "model_{encoder}.h5"

    @classmethod
    def get_weights_path(cls, encoder: str = "ResNet50V2") -> str:
        """Get model weights file path"""
        return cls.MODEL_WEIGHTS_FILE.format(encoder=encoder)

    @classmethod
    def get_model_path(cls, encoder: str = "ResNet50V2") -> str:
        """Get model file path"""
        return cls.MODEL_FILE.format(encoder=encoder)
