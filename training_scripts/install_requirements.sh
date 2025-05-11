#!/bin/bash

# Install Python 3.8 and venv module
sudo apt-get update
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev

# Create a new Python virtual environment
python3.8 -m venv mlops_env

# Activate the virtual environment
source mlops_env/bin/activate

# Install requirements from the specified path
python3.8 -m pip install -r /home/cc/MLOps_Project/training_scripts/arcface_torch/requirement.txt

echo "Setup completed successfully!"