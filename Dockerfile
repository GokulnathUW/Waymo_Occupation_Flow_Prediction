# Dockerfile for Waymo Occupancy Flow Prediction
# Based on TensorFlow GPU image with Waymo Open Dataset support

FROM tensorflow/tensorflow:2.12.0-gpu

# Install Waymo Open Dataset toolkit
RUN pip install waymo-open-dataset-tf-2-12-0==1.6.4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Default command
CMD ["python", "train.py", "--help"]