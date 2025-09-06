import os
import pathlib
from typing import Sequence
import uuid
import zlib

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_graphics.image.transformer as tfg_transformer

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_metrics
from waymo_open_dataset.utils import occupancy_flow_renderer
from waymo_open_dataset.utils import occupancy_flow_vis

from flow_prediction import BATCH_SIZE, ENCODER_USED, get_batch_counts, inference

## Waymo Open Dataset location
DATASET_FOLDER = 'waymo_open_dataset'
TEST_FILES = f'{DATASET_FOLDER}/test/testing_tfexample.tfrecord*'


## TESTING DATA
test_filenames = tf.io.matching_files(TEST_FILES)
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(occupancy_flow_data.parse_tf_example)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_it = iter(test_dataset)

_, _, num_test_batches = get_batch_counts(_, _, test_it)

## Load config
config_file = "config.txt"
with open (config_file, "r") as f:
    config_text = ''.join(f.readlines())
    
config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
text_format.Parse(config_text, config)


model = tf.keras.models.load_model(f"model_{ENCODER_USED}.h5", custom_objects={
    'tf': tf,
    'tfg_transformer': tfg_transformer,
    'occupancy_flow_metrics': occupancy_flow_metrics,
    'occupancy_flow_grids': occupancy_flow_grids,
    'occupancy_flow_renderer': occupancy_flow_renderer,
    'occupancy_flow_vis': occupancy_flow_vis
})

tf.keras.models.load_weights(model, f"model_weights_{ENCODER_USED}.h5", by_name=True)

test_total_loss = inference(test_it, num_test_batches)
print(f'Validation loss: {float(test_total_loss):.4f}')

