...existing code from model.py...import tensorflow as tf
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2


def _make_model(
    model_inputs: tf.Tensor,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    NUM_PRED_CHANNELS: int,
    # encoder: tf.keras.Model
) -> tf.keras.Model:
  """Simple convolutional model."""
  inputs = tf.keras.Input(tensor=model_inputs)

  # Can be removed if encoder is an argument.
  encoder = tf.keras.applications.ResNet50V2(
      include_top=False, weights=None, input_tensor=inputs
  )
  

  num_output_channels = NUM_PRED_CHANNELS * config.num_waypoints
  decoder_channels = [32, 64, 128, 256, 512]

  conv2d_kwargs = {
      'kernel_size': 3,
      'strides': 1,
      'padding': 'same',
  }

  x = encoder(inputs)

  for i in [4, 3, 2, 1, 0]:
    x = tf.keras.layers.Conv2D(
        filters=decoder_channels[i],
        activation='relu',
        name=f'upconv_{i}_0',
        **conv2d_kwargs,
    )(x)
    x = tf.keras.layers.UpSampling2D(name=f'upsample_{i}')(x)
    x = tf.keras.layers.Conv2D(
        filters=decoder_channels[i],
        activation='relu',
        name=f'upconv_{i}_1',
        **conv2d_kwargs,
    )(x)

  outputs = tf.keras.layers.Conv2D(
      filters=num_output_channels,
      activation=None,
      name=f'outconv',
      **conv2d_kwargs,
  )(x)

  return tf.keras.Model(
      inputs=inputs, outputs=outputs, name='occupancy_flow_model'
  )