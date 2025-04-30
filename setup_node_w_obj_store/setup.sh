#!/bin/bash
set -e

echo "🚀  Starting MLOps Node Setup..."

# 1. Install rclone if not already installed
if ! command -v rclone &> /dev/null; then
  echo "Installing rclone..."
  curl https://rclone.org/install.sh | sudo bash
else
  echo "✅  rclone already installed."
fi

# 2. Fix FUSE permissions
echo "Ensuring FUSE user_allow_other permissions..."
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

# 3. Create rclone config directory
echo "Setting up rclone config directory..."
mkdir -p ~/.config/rclone

# 4. Check if rclone.conf exists
if [ ! -f ~/.config/rclone/rclone.conf ]; then
  echo "❌  ERROR: ~/.config/rclone/rclone.conf missing!"
  echo "👉  Please upload your rclone.conf file before proceeding."
  exit 1
else
  echo "✅  rclone.conf found."
fi

# 5. Create /mnt/object and mount object storage
echo "Mounting object storage..."
sudo mkdir -p /mnt/object
sudo chown $USER:$USER /mnt/object || true

rclone mount chi_tacc:object-persist-project14 /mnt/object --allow-other --read-only --daemon

# 6. Ensure faces_dataset exists inside object store
if [ ! -d /mnt/object/faces_dataset ]; then
  echo "Creating faces_dataset in object storage..."
  rclone mkdir chi_tacc:object-persist-project14/faces_dataset
  echo "✅  Created faces_dataset folder."
else
  echo "✅  faces_dataset already exists."
fi

# 7. Create workspace directory
mkdir -p ~/workspace

# 8. Install Docker if not installed
if ! command -v docker &> /dev/null; then
  echo "Installing Docker..."
  curl -sSL https://get.docker.com/ | sudo sh
  sudo groupadd -f docker
  sudo usermod -aG docker $USER
  echo "✅  Docker installed. Please run 'newgrp docker' manually or re-login if needed."
else
  echo "✅  Docker already installed."
fi

# 9. Create docker-compose.yml locally
sudo docker run -d --rm \
  --privileged \
  -p 8888:8888 \
  --shm-size 8G \
  -e FACE_DATA_DIR=/mnt/faces_dataset \
  -v ~/workspace:/home/jovyan/work/ \
  --mount type=bind,source=/mnt/object,target=/mnt/faces_dataset,readonly \
  --name jupyter \
  quay.io/jupyter/pytorch-notebook:latest \
  start-notebook.sh \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.allow_remote_access=True \
  --ServerApp.ip=0.0.0.0 \
  --ServerApp.root_dir=/home/jovyan \
  --ServerApp.default_url=/lab

echo "✅  docker-compose.yml created successfully."
