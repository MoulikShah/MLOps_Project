#!/bin/bash

# Install Python 3.8 and venv module
sudo apt-get update
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev

# Create Python virtual environment in user's home directory
python3.8 -m venv $HOME/mlops_env

# Activate the virtual environment
source $HOME/mlops_env/bin/activate

# Install requirements from the specified path
python3.8 -m pip install -r $HOME/MLOps_Project/training_scripts/arcface_torch/requirement.txt

# Patch mxnet/numpy/utils.py to fix bool alias issue
MXNET_UTILS_FILE="$HOME/mlops_env/lib/python3.8/site-packages/mxnet/numpy/utils.py"
if [[ -f "$MXNET_UTILS_FILE" ]]; then
  echo "⚙️ Patching mxnet numpy/utils.py..."
  sed -i 's/bool_ = onp.bool_/bool_ = bool/' "$MXNET_UTILS_FILE"
  sed -i 's/bool = onp.bool/bool = bool/' "$MXNET_UTILS_FILE"
  echo "✅ Patch applied."
else
  echo "⚠️ mxnet numpy/utils.py not found. Patch skipped."
fi

echo "Setup completed successfully!"