## TODO: Include validation loss
print("Beginning totrain")
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

from model import _make_model
from loss_functions import _occupancy_flow_loss



## Waymo Open Dataset location
DATASET_FOLDER = 'waymo_open_dataset'
BATCH_COUNT_FILE = 'batch_count.txt'

BATCH_SIZE = 16
ENCODER_USED = "ResNet50V2" # change line 127 accordingly

# TFRecord dataset
TRAIN_FILES = f'{DATASET_FOLDER}/training/training_tfexample.tfrecord*'
VAL_FILES = f'{DATASET_FOLDER}/validation/validation_tfexample.tfrecord*'
TEST_FILES = f'{DATASET_FOLDER}/test/testing_tfexample.tfrecord*'


def write_model_summary():
    with open(f"{ENCODER_USED}.txt", "wb") as f:
        f.write(f"{'-'*100}\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write(f"{'-'*100}\n")
        
        f.write(f"\n\n\n\n")
        
        f.write(f"{'-'*100}\n")
        f.write("\nModel Variables:\n")
        for var in model.trainable_variables:
            f.write(f"{var.name} {var.shape}\n")
        f.write(f"{'-'*100}\n")
        
def count_batches(it):
    while True:
        try:
            batch = next(it)
            num_batches += 1
        except StopIteration:
            break
        
def get_batch_counts(train_it, val_it, test_it):
    if os.path.getsize(BATCH_COUNT_FILE) == 0:
        num_train_batches = count_batches(train_it)
        num_val_batches = count_batches(val_it)
        num_test_batches = count_batches(test_it)

        with open(BATCH_COUNT_FILE, "w") as f:
            f.write(f"Number of training batches: {num_train_batches}\n")
            f.write(f"Number of validation batches: {num_val_batches}\n")
            f.write(f"Number of testing batches: {num_test_batches}\n")
    else:
        with open(BATCH_COUNT_FILE, "r") as f:
            lines = f.readlines()
            num_train_batches = int(lines[0].split(":")[1])
            num_val_batches = int(lines[1].split(":")[1])
            num_test_batches = int(lines[2].split(":")[1])
    return num_train_batches, num_val_batches, num_test_batches

def _get_pred_waypoint_logits(
    model_outputs: tf.Tensor,
) -> occupancy_flow_grids.WaypointGrids:
  """Slices model predictions into occupancy and flow grids."""
  pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

  # Slice channels into output predictions.
  for k in range(config.num_waypoints):
    index = k * NUM_PRED_CHANNELS
    waypoint_channels = model_outputs[
        :, :, :, index : index + NUM_PRED_CHANNELS
    ]
    pred_observed_occupancy = waypoint_channels[:, :, :, :1]
    pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
    pred_flow = waypoint_channels[:, :, :, 2:]
    pred_waypoint_logits.vehicles.observed_occupancy.append(
        pred_observed_occupancy
    )
    pred_waypoint_logits.vehicles.occluded_occupancy.append(
        pred_occluded_occupancy
    )
    pred_waypoint_logits.vehicles.flow.append(pred_flow)

  return pred_waypoint_logits

def _preprocess_inputs(
    inputs: dict[str, tf.Tensor],
):
  """Preprocesses inputs"""
  
  # Extracts the self driving car (SDC/ego vehicle) state from the input data, 
  # storing it in the input dictionary (different keys). This is needed during
  # sdc centric task like cenrtering the roadmap, SDC centric predictions like
  # occluded occupation. 
  # Only some agents are of interest depending on the task. SDCs though, are always 
  # of interest. It is much easier to work with them by caching it in the input 
  # itself under different keys
  inputs = occupancy_flow_data.add_sdc_fields(inputs)
  
  # Scene is divided into a grid of cells (scene dimension and cell dimension in config.txt). 
  # For each cell, occupancy is marked 1 if is occupied by a part of an agent, 0 otherwise. ---> 1 channel ([batch_size, height, width, num_steps])
  # Each cell also captures flow (where does agent move next) if occupied. ---> 2 channels ([batch_size, height, width, num_steps, 2])
  # Flow is represented as a vector(dx, dy), hence 2 channels
  # The occupancy and flow grid is created for each timestep in the input data. (Past, current, future)
  # timestep_grids has grid for each agent type (timestep_grids.vehicle, timestep_grids.pedestrian, etc.)
  # Agent type: vehicle, pedestrian, cyclist
  # For each agent type, there are grids each for: occupancy (past, current, future (observed and occluded), all (91 steps)) and flow (all, 81 steps)  
  timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
      inputs, config
  )

  # Subset of timestep_grids. A list of all grids for each waypoint (not all steps)
  true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
      timestep_grids, config
  )
  
  # Creates visualization-friendly versions of the timestep grids. (heatmap and stuff)
  vis_grids = occupancy_flow_grids.create_ground_truth_vis_grids(
      inputs, timestep_grids, config
  )

  return timestep_grids, true_waypoints, vis_grids

def _make_model_inputs(
    timestep_grids: occupancy_flow_grids.TimestepGrids,
    vis_grids: occupancy_flow_grids.VisGrids,
) -> tf.Tensor:
  """Concatenates all occupancy grids over past, current to a single tensor."""
  model_inputs = tf.concat(
      [
          vis_grids.roadgraph,
          timestep_grids.vehicles.past_occupancy,
          timestep_grids.vehicles.current_occupancy,
          tf.clip_by_value(
              timestep_grids.pedestrians.past_occupancy
              + timestep_grids.cyclists.past_occupancy,
              0,
              1,
          ),
          tf.clip_by_value(
              timestep_grids.pedestrians.current_occupancy
              + timestep_grids.cyclists.current_occupancy,
              0,
              1,
          ),
      ],
      axis=-1,
  )
  return model_inputs

def train_one_step(inputs: dict[str, tf.Tensor]) -> tf.Tensor:
    with tf.GradientTape() as tape:
        # Preprocess inputs.
        timestep_grids, true_waypoints, vis_grids = _preprocess_inputs(inputs=inputs)
    
        # Prepare model inputs
        # although vis_grids is for visualizations, it contains useful information 
        # like roadgraph that provides context of road layout, important for prediction
        # function concats current and past occupancy of all agent types and roadgraph (because cyclists and 
        # pedestrians could occupy same cells, they are superimposed to one channel)
        model_inputs = _make_model_inputs(timestep_grids, vis_grids)
        model_outputs = model(model_inputs, training=training)  # [batch_size, grid_height_cells, grid_width_cells, 32]
        pred_waypoint_logits = _get_pred_waypoint_logits(model_outputs = model_outputs)
        
        # Compute loss
        loss_dict = _occupancy_flow_loss(
            config=config,
            true_waypoints=true_waypoints,
            pred_waypoint_logits=pred_waypoint_logits,
        )
        total_loss = tf.math.add_n(loss_dict.values())
        
    # Compute gradients
    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return total_loss

def inference(it, num_batches):
    total_loss = 0
    for batch in range(num_batches):
        inputs = next(it)
        timestep_grids, true_waypoints, vis_grids = _preprocess_inputs(inputs=inputs)
    
        model_inputs = _make_model_inputs(timestep_grids, vis_grids)
        model_outputs = model(model_inputs, training=training)
        pred_waypoint_logits = _get_pred_waypoint_logits(model_outputs = model_outputs)
        
        # Compute loss
        loss_dict = _occupancy_flow_loss(
            config=config,
            true_waypoints=true_waypoints,
            pred_waypoint_logits=pred_waypoint_logits,
        )
        total_loss += tf.math.add_n(loss_dict.values())
        
    return total_loss



## TRAINING DATA
train_filenames = tf.io.matching_files(TRAIN_FILES)                     # Get all the training files
train_dataset = tf.data.TFRecordDataset(train_filenames)                # Create a dataset of tf.Example protos.
train_dataset = train_dataset.repeat()                                  # Repeat the dataset indefinitely. Useful for training multiple epochs
train_dataset = train_dataset.map(occupancy_flow_data.parse_tf_example) # Parse the tf.Example protos into a dataset of (input, output) pairs.
train_dataset = train_dataset.batch(BATCH_SIZE)                         # Batch the dataset.
train_it = iter(train_dataset)                                          # Create an iterator over the dataset.

## VALIDATION DATA
val_filenames = tf.io.matching_files(VAL_FILES)
val_dataset = tf.data.TFRecordDataset(train_filenames)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.map(occupancy_flow_data.parse_tf_example)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_it = iter(val_dataset)

## TESTING DATA
test_filenames = tf.io.matching_files(TEST_FILES)
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(occupancy_flow_data.parse_tf_example)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_it = iter(test_dataset)


num_train_batches, num_val_batches, num_test_batches = get_batch_counts(train_it, val_it, test_it)
## Load config
config_file = "config.txt"
with open (config_file, "r") as f:
    config_text = ''.join(f.readlines())
    
config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
text_format.Parse(config_text, config)


## Model build
# Number of channels output by the model.
# Occupancy of currently-observed vehicles: 1 channel.
# Occupancy of currently-occluded vehicles: 1 channel.
# Flow of all vehicles: 2 channels.
NUM_PRED_CHANNELS = 4

# encoder = tf.keras.applications.ResNet50V2(
#     include_top=False, weights=None, input_tensor=model_inputs
# )

sample_inputs = next(test_it)
sample_timestep_grids, sample_true_waypoints, sample_vis_grids = _preprocess_inputs(inputs=sample_inputs)
sample_model_inputs = _make_model_inputs(sample_timestep_grids, sample_vis_grids)

model = _make_model(model_inputs = sample_model_inputs, config=config, num_pred_channels=NUM_PRED_CHANNELS)
# model = _make_model(config=config, num_pred_channels=NUM_PRED_CHANNELS, encoder=encoder)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


## Training
training = True

## TFRecord dataset stores no metadata about the dataset. So there's no way to know how many batches are in the dataset.
## IDEA: Run a seperate code without report to count number of batches in the dataset.
## For now, we will just run for 1000 epochs.


TRAIN_EPOCHS = 15
for epoch in range(TRAIN_EPOCHS):
    for batch in range(num_train_batches):
        train_inputs = next(train_it)
        train_total_loss = train_one_step(train_inputs)
        print(f'Training loss after batch {batch}/{num_train_batches}: {float(train_total_loss):.4f}')
        
    val_total_loss = inference(val_it, num_val_batches)
    print(f'Validation loss after epoch {epoch}/{TRAIN_EPOCHS}: {float(val_total_loss):.4f}')
    
# Save model checkpoint
model.save_weights(f"model_weights_{ENCODER_USED}.h5")
model.save(f"model_{ENCODER_USED}.h5")
write_model_summary()