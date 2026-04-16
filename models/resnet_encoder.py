"""
Occupancy Flow Prediction Model Architecture.
Implements a ResNet-based encoder-decoder architecture for predicting
vehicle occupancy and flow in autonomous driving scenarios.
"""

import tensorflow as tf
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2


def create_occupancy_flow_model(
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    num_pred_channels: int = 4,
    input_shape: tuple = None,
    encoder_name: str = "ResNet50V2",
) -> tf.keras.Model:
    """Builds a convolutional encoder-decoder model for occupancy flow prediction"""
    # Define encoder architecture
    encoder = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        input_shape=input_shape,
    )

    # Calculate total output channels
    num_output_channels = num_pred_channels * config.num_waypoints

    # Decoder channel configuration
    decoder_channels = [32, 64, 128, 256, 512]

    # Convolution layer parameters
    conv2d_kwargs = {
        "kernel_size": 3,
        "strides": 1,
        "padding": "same",
    }

    # Build encoder
    if input_shape is not None:
        inputs = tf.keras.Input(shape=input_shape)
    else:
        inputs = tf.keras.Input(tensor=None)

    # Encoder forward pass
    x = encoder(inputs)

    # Decoder: Progressive upsampling with convolutions
    for i in [4, 3, 2, 1, 0]:
        x = tf.keras.layers.Conv2D(
            filters=decoder_channels[i],
            activation="relu",
            name=f"upconv_{i}_0",
            **conv2d_kwargs,
        )(x)
        x = tf.keras.layers.UpSampling2D(name=f"upsample_{i}")(x)
        x = tf.keras.layers.Conv2D(
            filters=decoder_channels[i],
            activation="relu",
            name=f"upconv_{i}_1",
            **conv2d_kwargs,
        )(x)

    # Output layer: no activation (logits for occupancy, raw values for flow)
    outputs = tf.keras.layers.Conv2D(
        filters=num_output_channels,
        activation=None,
        name="outconv",
        **conv2d_kwargs,
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="occupancy_flow_model",
    )

    return model


def get_pred_waypoint_logits(
    model_outputs: tf.Tensor,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    num_pred_channels: int = 4,
) -> "occupancy_flow_grids.WaypointGrids":
    """Slices model predictions into occupancy and flow grids for each waypoint"""
    from waymo_open_dataset.utils import occupancy_flow_grids

    pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

    # Slice channels into output predictions for each waypoint
    for k in range(config.num_waypoints):
        index = k * num_pred_channels
        waypoint_channels = model_outputs[
            :, :, :, index : index + num_pred_channels
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
