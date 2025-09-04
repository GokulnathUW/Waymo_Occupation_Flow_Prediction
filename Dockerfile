# Use an official Python image as a base (you can choose a version like 3.8)
FROM tensorflow/tensorflow:2.12.0-gpu

RUN pip install waymo-open-dataset-tf-2-12-0==1.6.4

RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory inside the container
WORKDIR /app
# Copy the rest of your application code (if any)
# COPY . /app

# Command to run when the container starts (can be modified for your needs)
CMD ["echo", "hello"]