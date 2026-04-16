from __future__ import annotations

"""
Loss functions for Occupancy Flow Prediction.
Implements multi-component loss including:
- Observed occupancy cross-entropy
- Occluded occupancy cross-entropy
- Flow prediction loss (L1)
"""

import tensorflow as tf
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_grids


def occupancy_flow_loss(
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    true_waypoints: occupancy_flow_grids.WaypointGrids,
    pred_waypoint_logits: occupancy_flow_grids.WaypointGrids,
) -> dict[str, tf.Tensor]:
    """Computes the multi-component occupancy flow loss"""
    loss_dict = {}

    # Initialize lists to store per-waypoint losses
    loss_dict["observed_xe"] = []
    loss_dict["occluded_xe"] = []
    loss_dict["flow"]        = []

    # Iterate over waypoints and accumulate losses
    for k in range(config.num_waypoints):
        # Get predicted and true occupancy for observed vehicles
        pred_observed_occupancy_logit = (
            pred_waypoint_logits.vehicles.observed_occupancy[k]
        )
        pred_occluded_occupancy_logit = (
            pred_waypoint_logits.vehicles.occluded_occupancy[k]
        )
        true_observed_occupancy = true_waypoints.vehicles.observed_occupancy[k]
        true_occluded_occupancy = true_waypoints.vehicles.occluded_occupancy[k]

        # Compute cross-entropy losses
        loss_dict["observed_xe"].append(
            sigmoid_cross_entropy_loss(
                true_occupancy=true_observed_occupancy,
                pred_occupancy=pred_observed_occupancy_logit,
            )
        )
        loss_dict["occluded_xe"].append(
            sigmoid_cross_entropy_loss(
                true_occupancy=true_occluded_occupancy,
                pred_occupancy=pred_occluded_occupancy_logit,
            )
        )

        # Compute flow loss
        pred_flow = pred_waypoint_logits.vehicles.flow[k]
        true_flow = true_waypoints.vehicles.flow[k]
        loss_dict["flow"].append(flow_loss(pred_flow, true_flow))

    # Average losses over all waypoints
    loss_dict["observed_xe"] = (
        tf.math.add_n(loss_dict["observed_xe"]) / config.num_waypoints
    )
    loss_dict["occluded_xe"] = (
        tf.math.add_n(loss_dict["occluded_xe"]) / config.num_waypoints
    )
    loss_dict["flow"] = tf.math.add_n(loss_dict["flow"]) / config.num_waypoints

    return loss_dict


def sigmoid_cross_entropy_loss(true_occupancy, pred_occupancy, loss_weight = 1000.0):
    """Computes weighted sigmoid cross-entropy loss over all grid cells"""
    # Compute sum of cross-entropy over all pixels
    xe_sum = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=batch_flatten(true_occupancy),
            logits=batch_flatten(pred_occupancy),
        )
    )

    # Return weighted mean
    return loss_weight * xe_sum / tf.size(pred_occupancy, out_type=tf.float32)


def flow_loss(true_flow, pred_flow, loss_weight = 1.0):
    """Computes L1 flow loss with masking for zero-flow regions"""
    diff = true_flow - pred_flow

    # Create mask for non-zero flow regions
    # Split into dx and dy components
    true_flow_dx, true_flow_dy = tf.split(true_flow, 2, axis=-1)

    # Mask where flow exists (either dx or dy is non-zero)
    flow_exists = tf.logical_or(
        tf.not_equal(true_flow_dx, 0.0),
        tf.not_equal(true_flow_dy, 0.0),
    )
    flow_exists = tf.cast(flow_exists, tf.float32)

    # Apply mask to difference
    diff = diff * flow_exists

    # Compute L1 norm of masked difference
    diff_norm = tf.linalg.norm(diff, ord=1, axis=-1)

    # Compute mean only over valid (non-zero flow) regions
    # Divide by 2 since (dx, dy) is counted twice in the norm
    mean_diff = tf.math.divide_no_nan(
        tf.reduce_sum(diff_norm),
        tf.reduce_sum(flow_exists) / 2,
    )

    return loss_weight * mean_diff


def batch_flatten(input_tensor):
    """Flattens tensor to shape [batch_size, -1]"""
    image_shape = tf.shape(input_tensor)
    return tf.reshape(
        input_tensor, tf.concat([image_shape[0:1], [-1]], axis=0)
    )
