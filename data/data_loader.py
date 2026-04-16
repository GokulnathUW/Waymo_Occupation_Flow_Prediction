from __future__ import annotations

"""
Data loading for Waymo Occupation Flow Prediction.
Provides TFRecord dataset creation with proper shuffling, batching,
and prefetching. No batch counting -- datasets are iterated directly.
"""

import tensorflow as tf
from waymo_open_dataset.utils import occupancy_flow_data


def create_training_dataset(file_pattern, batch_size = 16) -> tf.data.Dataset:
    """
    Create a training dataset (shuffled, batched, prefetched)
    This dataset iterates through the data once (one epoch)
    For multi-epoch training, recreate the dataset each epoch
    """
    filenames = tf.io.matching_files(file_pattern)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_validation_dataset(file_pattern, batch_size = 16) -> tf.data.Dataset:
    """
    Create a validation dataset (no shuffle, batched, prefetched)
    This dataset iterates through the data once
    """
    filenames = tf.io.matching_files(file_pattern)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_test_dataset(file_pattern, batch_size = 16) -> tf.data.Dataset:
    """Create a test dataset (no shuffle, no repeat, batched)
    This dataset iterates through the data exactly once
    """
    filenames = tf.io.matching_files(file_pattern)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
