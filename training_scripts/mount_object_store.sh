#!/bin/bash
# Script to mount Chameleon object storage with Swift optimization
# Save this file as mount_object_store.sh

# Set variables
CONTAINER_NAME="object-persist-project-14"
MOUNT_POINT="/mnt/object"
RCLONE_CONFIG_NAME="chi_tacc"

# Print start message
echo "Starting object storage mount script..."
echo "Container: $CONTAINER_NAME"
echo "Mount point: $MOUNT_POINT"

# Check if mount point exists, create if not
if [ ! -d "$MOUNT_POINT" ]; then
    echo "Creating mount point directory..."
    sudo mkdir -p "$MOUNT_POINT"
fi

# Set proper ownership
echo "Setting directory permissions..."
sudo chown -R $(whoami) "$MOUNT_POINT"
sudo chgrp -R $(whoami) "$MOUNT_POINT"

# Check if already mounted, unmount if needed
if mountpoint -q "$MOUNT_POINT"; then
    echo "Unmounting existing mount point..."
    fusermount -u "$MOUNT_POINT"
fi

# Mount the object store with optimization flags
echo "Mounting object storage..."
rclone mount "$RCLONE_CONFIG_NAME:$CONTAINER_NAME" "$MOUNT_POINT" \
    --read-only \
    --allow-other \
    --vfs-cache-mode=full \
    --dir-cache-time=72h \
    --swift-fetch-until-empty-page \
    --daemon

# Wait a moment for the mount to initialize
sleep 3

# Check if mount was successful
if mountpoint -q "$MOUNT_POINT"; then
    echo "Mount successful!"
    echo "Available directories:"
    ls -la "$MOUNT_POINT"
    
    # Check dataset structure
    if [ -d "$MOUNT_POINT/dataset/datasets" ]; then
        echo ""
        echo "Dataset directories:"
        ls -la "$MOUNT_POINT/dataset/datasets"
        
        # Count files in subdirectories
        for dir in "$MOUNT_POINT/dataset/datasets"/*; do
            if [ -d "$dir" ]; then
                dir_name=$(basename "$dir")
                file_count=$(ls -la "$dir" | wc -l)
                echo "$dir_name directory contains approximately $file_count items"
            fi
        done
    fi
else
    echo "Mount failed! Please check for errors."
fi

echo ""
echo "Mount process complete."