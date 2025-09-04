import tensorflow as tf
from waymo_open_dataset.utils import occupancy_flow_data

## Waymo Open Dataset location
DATASET_FOLDER = 'flow_prediction_stat453/waymo_open_dataset'

# TFRecord dataset
TRAIN_FILES = f'{DATASET_FOLDER}/training/training_tfexample.tfrecord*'
VAL_FILES = f'{DATASET_FOLDER}/validation/validation_tfexample.tfrecord*'
TEST_FILES = f'{DATASET_FOLDER}/testing/testing_tfexample.tfrecord*'

BATCH_SIZE = 16


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

num_train_batches = 0
num_val_batches = 0
num_test_batches = 0

def count_batches(it):
    while True:
        try:
            batch = next(it)
            num_batches += 1
        except StopIteration:
            break

while True:
    try:
        batch = next(val_it)
        num_val_batches += 1
    except StopIteration:
        break
print(f'Number of batches: {num_val_batches}')

while True:
    try:
        batch = next(test_it)
        num_test_batches += 1
    except StopIteration:
        break
print(f'Number of batches: {num_test_batches}')

with open('batch_count.txt', 'w') as f:
    f.write(f'Number of training batches: {num_train_batches}\n')
    f.write(f'Number of validation batches: {num_val_batches}\n')
    f.write(f'Number of testing batches: {num_test_batches}\n')