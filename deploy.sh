#! /usr/bin/bash

#Unzip input files
tar -xzvf waymo_open_dataset.tar.gz

python3 flow_prediction.py > flow_prediction.out

echo "Model training completed"

echo "Running testing script"

python3 testing.py 
#Zip output files
# tar -czf 