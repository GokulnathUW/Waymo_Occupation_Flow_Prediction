# Waymo Occupancy Flow Prediction

Predicting future vehicle occupancy and motion flow for autonomous driving using the Waymo Open Dataset.

## Overview

This project implements a deep learning-based approach to predict where vehicles will be in future timesteps, including both observed and occluded vehicles. The model uses a ResNet50V2 encoder-decoder architecture to predict:

1. **Observed Occupancy**: Grid cells occupied by currently visible vehicles
2. **Occluded Occupancy**: Grid cells occupied by hidden/occluded vehicles  
3. **Vehicle Flow**: Direction vectors (dx, dy) showing vehicle movement

The task is formulated as a grid-based prediction problem where the scene is discretized into cells, and the model predicts occupancy and flow for each cell across multiple future waypoints.

## Architecture

```
Input Grids (Roadgraph + Past/Present Occupancy)
    ↓
ResNet50V2 Encoder
    ↓
Progressive Upsampling Decoder
    ↓
Multi-channel Output (Occupancy + Flow per Waypoint)
```

### Model Components

- **Encoder**: ResNet50V2 backbone for feature extraction
- **Decoder**: 5-stage upsampling pipeline with intermediate convolutions
- **Output**: 4 channels per waypoint (observed occupancy, occluded occupancy, flow dx, flow dy)

### Loss Function

The training objective combines three components:

```
L_total = L_observed_CE + L_occluded_CE + L_flow
```

- **L_observed_CE**: Sigmoid cross-entropy for observed vehicle occupancy (weight: 1000)
- **L_occluded_CE**: Sigmoid cross-entropy for occluded vehicle occupancy (weight: 1000)  
- **L_flow**: L1 loss on flow vectors (masked for zero-flow regions, weight: 1)

## Project Structure

```
├── config/                  # Configuration management
│   ├── config.py           # Task, model, and path configurations
│   └── task_config.txt     # Waymo task-specific parameters
├── data/                    # Data loading utilities
│   └── data_loader.py      # TFRecord dataset creation and batching
├── models/                  # Model architectures
│   └── resnet_encoder.py   # ResNet-based encoder-decoder
├── trainers/                # Training pipeline
│   └── trainer.py          # Training loop with validation
├── evaluators/              # Evaluation pipeline
│   └── evaluator.py        # Model evaluation and metrics
├── utils/                   # Utility functions
│   ├── loss_functions.py   # Multi-component loss implementations
│   └── preprocessing.py    # Input grid creation and transformation
├── viz/                     # Visualization utilities
│   └── visualization.py    # Trajectory and animation generation
├── train.py                 # Main training entry point
├── evaluate.py              # Main evaluation entry point
├── visualize.py             # Visualization script
├── deploy.sh                # Deployment script
├── deploy.sub               # HTCondor submission file
├── Dockerfile               # Container definition
└── requirements.txt         # Python dependencies
```

## Installation

### Requirements

- Python 3.8+
- TensorFlow 2.12+
- Waymo Open Dataset toolkit
- Matplotlib, NumPy

### Setup

```bash
# Clone repository
git clone <repository-url>
cd Waymo_Occupation_Flow_Prediction

# Install dependencies
pip install -r requirements.txt

# Download Waymo Open Dataset
# Place dataset in waymo_open_dataset/ directory with:
#   - training/training_tfexample.tfrecord*
#   - validation/validation_tfexample.tfrecord*
#   - test/testing_tfexample.tfrecord*
```

### Docker

```bash
# Build image
docker build -t waymo-occupancy-flow .

# Run training
docker run --gpus all -v /path/to/data:/app/data waymo-occupancy-flow python train.py
```

## Usage

### Training

```bash
# Default training
python train.py

# Custom parameters
python train.py --epochs 20 --batch-size 8 --encoder ResNet50V2

# Force batch recount
python train.py --force-recount
```

### Evaluation

```bash
# Evaluate with default checkpoints
python evaluate.py

# Custom weights path
python evaluate.py --weights-path path/to/weights.h5
```

### Visualization

```bash
# Generate frame visualizations
python visualize.py --output-dir ./viz_output --num-samples 10

# Create animations (requires ffmpeg)
python visualize.py --output-dir ./viz_output --create-animation
```

## Configuration

Task-specific parameters are defined in `config/task_config.txt`:

| Parameter                  | Value | Description                                  |
|---------------------------|-------|----------------------------------------------|
| num_past_steps           | 10    | Number of past timesteps                     |
| num_future_steps         | 80    | Number of future timesteps to predict        |
| num_waypoints            | 8     | Number of prediction waypoints               |
| grid_height_cells        | 256   | Grid height in cells                         |
| grid_width_cells         | 256   | Grid width in cells                          |
| pixels_per_meter         | 3.2   | Spatial resolution                           |
| normalize_sdc_yaw        | true  | Normalize to SDC-centric coordinates         |

Model hyperparameters are in `config/config.py`:

| Parameter         | Value  | Description                    |
|------------------|--------|--------------------------------|
| LEARNING_RATE    | 1e-3   | Adam optimizer learning rate   |
| BATCH_SIZE       | 16     | Training batch size            |
| TRAIN_EPOCHS     | 15     | Number of training epochs      |
| ENCODER          | ResNet50V2 | Encoder architecture       |

## Dataset

This project uses the **Waymo Open Dataset** Occupancy and Flow prediction task.

- **Input**: 10 past timesteps of agent positions + roadgraph
- **Output**: 80 future timesteps aggregated into 8 waypoints
- **Grid**: 256×256 cells centered on the self-driving car

## Methodology

### Data Preprocessing

1. Parse TFRecord examples using Waymo Open Dataset utilities
2. Extract SDC (self-driving car) state for coordinate normalization
3. Create ground truth grids:
   - **TimestepGrids**: Occupancy + flow for each timestep
   - **WaypointGrids**: Subsampled timesteps at prediction intervals
   - **VisGrids**: Visualization-friendly format

### Model Input

The model receives a multi-channel tensor:
- Roadgraph (static context)
- Vehicle past occupancy
- Vehicle current occupancy
- Pedestrian+cyclist past occupancy (superimposed)
- Pedestrian+cyclist current occupancy (superimposed)

### Training Pipeline

For each training step:
1. Forward pass through encoder-decoder
2. Slice predictions into waypoint-specific grids
3. Compute multi-component loss
4. Backpropagate gradients with Adam optimizer

## Performance Metrics

The evaluation reports three loss components:
- **Observed Occupancy CE**: Lower is better (accuracy of visible vehicle prediction)
- **Occluded Occupancy CE**: Lower is better (ability to predict hidden vehicles)
- **Flow Loss**: Lower is better (accuracy of motion vector prediction)

## Reproducing Results

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Count dataset batches (first run only)
python -c "from data.data_loader import create_data_iterators, get_or_count_batches; \
           from config.config import PathConfig, ModelConfig; \
           mc = ModelConfig(); pc = PathConfig(); \
           train_it, val_it, test_it = create_data_iterators(pc.TRAIN_FILES, pc.VAL_FILES, pc.TEST_FILES, mc.BATCH_SIZE); \
           get_or_count_batches(train_it, val_it, test_it, pc.BATCH_COUNT_FILE)"

# 3. Train model
python train.py --epochs 15

# 4. Evaluate
python evaluate.py

# 5. Visualize results
python visualize.py --num-samples 5
```

## Citation

If you use this code, please cite the Waymo Open Dataset paper:

```
@misc{waymo_open_dataset,
  title={Waymo Open Dataset: Occupancy and Flow Prediction},
  author={Waymo},
  year={2023}
}
```

## License

This project is for academic/educational use. See the Waymo Open Dataset terms of use for dataset licensing.

## Acknowledgments

- Waymo Open Dataset for providing the dataset and evaluation utilities
- TensorFlow for deep learning framework support
