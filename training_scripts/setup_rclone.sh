#!/bin/bash

set -e  # Exit on any error
echo "Starting rclone setup..."

# Step 1: Install rclone
echo "Installing rclone..."
curl https://rclone.org/install.sh | sudo bash

# Step 2: Check rclone installed
echo "Checking rclone version..."
rclone version

# Step 3: Create rclone config folder
echo "Creating rclone config directory..."
mkdir -p ~/.config/rclone

# Step 4: Copy config file
CONFIG_SOURCE="$HOME/MLOps_Project/object_store_config"
CONFIG_TARGET="$HOME/.config/rclone/rclone.conf"

if [[ ! -f "$CONFIG_SOURCE" ]]; then
    echo "ERROR: Config file '$CONFIG_SOURCE' not found!"
    exit 1
fi

echo "Copying config from $CONFIG_SOURCE to $CONFIG_TARGET..."
cp "$CONFIG_SOURCE" "$CONFIG_TARGET"

# Step 5: Test rclone connection
echo "Testing rclone connection to chi_tacc:"
rclone lsd chi_tacc:

echo "✅ Rclone setup complete."
