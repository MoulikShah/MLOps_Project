#!/bin/bash

set -e

MOUNT_POINT="/mnt/block"
DEVICE="/dev/vdb1"

echo "Creating mount point at $MOUNT_POINT..."
sudo mkdir -p "$MOUNT_POINT"

echo "Mounting $DEVICE to $MOUNT_POINT..."
sudo mount "$DEVICE" "$MOUNT_POINT"

echo "Listing contents of $MOUNT_POINT..."
ls "$MOUNT_POINT"

echo "✅ Volume mounted successfully at $MOUNT_POINT."
