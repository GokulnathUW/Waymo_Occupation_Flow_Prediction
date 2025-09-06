import tensorflow as tf

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_grids

## define loss functions
def _occupancy_flow_loss(
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    true_waypoints: occupancy_flow_grids.WaypointGrids,
    pred_waypoint_logits: occupancy_flow_grids.WaypointGrids,
) -> dict[str, tf.Tensor]:
  """Loss function.

  Args:
    config: OccupancyFlowTaskConfig proto message.
    true_waypoints: Ground truth labels.
    pred_waypoint_logits: Predicted occupancy logits and flows.

  Returns:
    A dict containing different loss tensors:
      observed_xe: Observed occupancy cross-entropy loss.
      occluded_xe: Occluded occupancy cross-entropy loss.
      flow: Flow loss.
  """
  loss_dict = {}
  # Store loss tensors for each waypoint and average at the end.
  loss_dict['observed_xe'] = []
  loss_dict['occluded_xe'] = []
  loss_dict['flow'] = []

  # Iterate over waypoints.
  for k in range(config.num_waypoints):
    # Occupancy cross-entropy loss.
    pred_observed_occupancy_logit = (
        pred_waypoint_logits.vehicles.observed_occupancy[k]
    )
    pred_occluded_occupancy_logit = (
        pred_waypoint_logits.vehicles.occluded_occupancy[k]
    )
    true_observed_occupancy = true_waypoints.vehicles.observed_occupancy[k]
    true_occluded_occupancy = true_waypoints.vehicles.occluded_occupancy[k]

    # Accumulate over waypoints.
    loss_dict['observed_xe'].append(
        _sigmoid_xe_loss(
            true_occupancy=true_observed_occupancy,
            pred_occupancy=pred_observed_occupancy_logit,
        )
    )
    loss_dict['occluded_xe'].append(
        _sigmoid_xe_loss(
            true_occupancy=true_occluded_occupancy,
            pred_occupancy=pred_occluded_occupancy_logit,
        )
    )

    # Flow loss.
    pred_flow = pred_waypoint_logits.vehicles.flow[k]
    true_flow = true_waypoints.vehicles.flow[k]
    loss_dict['flow'].append(_flow_loss(pred_flow, true_flow))

  # Mean over waypoints.
  loss_dict['observed_xe'] = (
      tf.math.add_n(loss_dict['observed_xe']) / config.num_waypoints
  )
  loss_dict['occluded_xe'] = (
      tf.math.add_n(loss_dict['occluded_xe']) / config.num_waypoints
  )
  loss_dict['flow'] = tf.math.add_n(loss_dict['flow']) / config.num_waypoints

  return loss_dict


def _sigmoid_xe_loss(
    true_occupancy: tf.Tensor,
    pred_occupancy: tf.Tensor,
    loss_weight: float = 1000,
) -> tf.Tensor:
  """Computes sigmoid cross-entropy loss over all grid cells."""
  # Since the mean over per-pixel cross-entropy values can get very small,
  # we compute the sum and multiply it by the loss weight before computing
  # the mean.
  xe_sum = tf.reduce_sum(
      tf.nn.sigmoid_cross_entropy_with_logits(
          labels=_batch_flatten(true_occupancy),
          logits=_batch_flatten(pred_occupancy),
      )
  )
  # Return mean.
  return loss_weight * xe_sum / tf.size(pred_occupancy, out_type=tf.float32)


def _flow_loss(
    true_flow: tf.Tensor,
    pred_flow: tf.Tensor,
    loss_weight: float = 1,
) -> tf.Tensor:
  """Computes L1 flow loss."""
  diff = true_flow - pred_flow
  # Ignore predictions in areas where ground-truth flow is zero.
  # [batch_size, height, width, 1], [batch_size, height, width, 1]
  true_flow_dx, true_flow_dy = tf.split(true_flow, 2, axis=-1)
  # [batch_size, height, width, 1]
  flow_exists = tf.logical_or(
      tf.not_equal(true_flow_dx, 0.0),
      tf.not_equal(true_flow_dy, 0.0),
  )
  flow_exists = tf.cast(flow_exists, tf.float32)
  diff = diff * flow_exists
  diff_norm = tf.linalg.norm(diff, ord=1, axis=-1)  # L1 norm.
  mean_diff = tf.math.divide_no_nan(
      tf.reduce_sum(diff_norm), tf.reduce_sum(flow_exists) / 2
  )  # / 2 since (dx, dy) is counted twice.
  return loss_weight * mean_diff


def _batch_flatten(input_tensor: tf.Tensor) -> tf.Tensor:
  """Flatten tensor to a shape [batch_size, -1]."""
  image_shape = tf.shape(input_tensor)
  return tf.reshape(input_tensor, tf.concat([image_shape[0:1], [-1]], 0))
