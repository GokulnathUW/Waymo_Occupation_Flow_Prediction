from __future__ import annotations

"""
Preprocessing utilities for Occupancy Flow Prediction.
Handles input data transformation into model-ready grid representations.
"""

import tensorflow as tf
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids

from config.config import TaskConfig


def preprocess_inputs(
    inputs: dict[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig = None,
) -> tuple:
    """Preprocesses raw inputs into grid representations for training/evaluation"""
    if config is None:
        config = TaskConfig().get()

    # Extract SDC (self-driving car) state and cache in inputs
    inputs = occupancy_flow_data.add_sdc_fields(inputs)

    # Create ground truth timestep grids
    # Each cell: occupancy (1 if occupied, 0 otherwise) + flow vector (dx, dy)
    timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
        inputs, config
    )

    # Create waypoint grids (subset of timesteps at prediction intervals)
    true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
        timestep_grids, config
    )

    # Create visualization grids (heatmap-friendly format)
    vis_grids = occupancy_flow_grids.create_ground_truth_vis_grids(
        inputs, timestep_grids, config
    )

    return timestep_grids, true_waypoints, vis_grids


def create_model_inputs(
    timestep_grids: occupancy_flow_grids.TimestepGrids,
    vis_grids: occupancy_flow_grids.VisGrids,
) -> tf.Tensor:
    """Concatenates occupancy grids into a single model input tensor"""
    model_inputs = tf.concat(
        [
            # Road graph: static context (lane markings, road edges, etc.)
            vis_grids.roadgraph,
            # Vehicle occupancy (past and current)
            timestep_grids.vehicles.past_occupancy,
            timestep_grids.vehicles.current_occupancy,
            # Pedestrian + Cyclist occupancy (combined, may overlap)
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
