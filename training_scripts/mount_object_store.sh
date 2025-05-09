#!/bin/bash

set -e  # Exit on any error

MOUNT_DIR="/mnt/object"
REMOTE_NAME="chi_tacc"
CONTAINER_NAME="object-persist-project-14"

echo "Creating mount point at $MOUNT_DIR..."
sudo mkdir -p "$MOUNT_DIR"
sudo chown -R cc "$MOUNT_DIR"
sudo chgrp -R cc "$MOUNT_DIR"

echo "Ensuring 'user_allow_other' is enabled in /etc/fuse.conf..."
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

echo "Mounting $REMOTE_NAME:$CONTAINER_NAME to $MOUNT_DIR using rclone..."
rclone mount "${REMOTE_NAME}:${CONTAINER_NAME}" "$MOUNT_DIR" \
  --read-only \
  --allow-other \
  --daemon

echo "Listing contents of $MOUNT_DIR..."
ls "$MOUNT_DIR"

echo "✅ Object store mounted successfully at $MOUNT_DIR"
