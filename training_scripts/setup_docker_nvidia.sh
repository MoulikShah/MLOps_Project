#!/bin/bash

set -e  # Exit on any error
echo "Starting setup for Docker and NVIDIA container toolkit..."

# 1. Install Docker
echo "Installing Docker..."
curl -sSL https://get.docker.com/ | sudo sh

# 2. Add current user to docker group
echo "Adding user $USER to docker group..."
sudo groupadd -f docker
sudo usermod -aG docker "$USER"

echo "IMPORTANT: You must log out and log back in (or reconnect SSH) for Docker group permissions to take effect."

# 3. Set up NVIDIA container toolkit
echo "Setting up NVIDIA container toolkit..."
sudo mkdir -p /usr/share/keyrings

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 4. Install NVIDIA container runtime
echo "Installing NVIDIA container runtime..."
sudo apt update
sudo apt-get install -y nvidia-container-toolkit

# 5. Configure Docker runtime
echo "Configuring Docker runtime for NVIDIA..."
sudo nvidia-ctk runtime configure --runtime=docker

# Handle issue #48
if [ -f /etc/docker/daemon.json ]; then
  sudo jq 'if has("exec-opts") then . else . + {"exec-opts": ["native.cgroupdriver=cgroupfs"]} end' \
    /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp > /dev/null
  sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
else
  echo '{"exec-opts": ["native.cgroupdriver=cgroupfs"]}' | sudo tee /etc/docker/daemon.json > /dev/null
fi

# 6. Restart Docker
echo "Restarting Docker..."
sudo systemctl restart docker

# 7. Install nvtop
echo "Installing nvtop (GPU monitor)..."
sudo apt update
sudo apt -y install nvtop

echo "Setup complete. Please reconnect your session to apply Docker group changes."
