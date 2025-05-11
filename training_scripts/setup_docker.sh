#!/bin/bash

set -e  # Exit on any error
echo "Starting setup for Docker..."

# 1. Install Docker
echo "Installing Docker..."
curl -sSL https://get.docker.com/ | sudo sh

# 2. Add current user to docker group
echo "Adding user $USER to docker group..."
sudo groupadd -f docker
sudo usermod -aG docker "$USER"

echo "IMPORTANT: You must log out and log back in (or reconnect SSH) for Docker group permissions to take effect."

