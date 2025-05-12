#!/bin/bash

set -e  # Exit immediately on error

# Update system packages
sudo apt-get update

# Create Python virtual environment using system Python
python3 -m venv $HOME/mlops_env

# Activate the virtual environment
source $HOME/mlops_env/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install ROCm-compatible PyTorch and Torchvision
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3

# Install remaining requirements
python3 -m pip install -r $HOME/MLOps_Project/training_scripts/arcface_torch/requirement_amd.txt

# Patch mxnet/numpy/utils.py to fix bool alias issue
MXNET_UTILS_FILE="$HOME/mlops_env/lib/python3.12/site-packages/mxnet/numpy/utils.py"
if [[ -f "$MXNET_UTILS_FILE" ]]; then
  echo "⚙️ Patching mxnet numpy/utils.py..."
  sed -i 's/bool_ = onp.bool_/bool_ = bool/' "$MXNET_UTILS_FILE"
  sed -i 's/bool = onp.bool/bool = bool/' "$MXNET_UTILS_FILE"
  echo "✅ Patch applied."
else
  echo "⚠️ mxnet numpy/utils.py not found. Patch skipped."
fi

echo "✅ AMD MLOps environment setup completed successfully!"
