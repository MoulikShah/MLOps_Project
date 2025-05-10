#!/bin/bash
# Script to set up rclone for Chameleon Cloud object storage

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo or as root"
  exit 1
fi

# Install rclone
echo "Installing rclone..."
curl https://rclone.org/install.sh | bash

# Configure FUSE for allowing other users to access mounts
echo "Configuring FUSE..."
sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

# Prompt for application credential details
echo "Please enter your Chameleon Cloud application credential details:"
read -p "User ID: " USER_ID
read -p "Application Credential ID: " APP_CRED_ID
read -sp "Application Credential Secret: " APP_CRED_SECRET
echo ""

# Create rclone config directory
mkdir -p ~/.config/rclone

# Create rclone config file
cat > ~/.config/rclone/rclone.conf << EOF
[chi_tacc]
type = swift
user_id = $USER_ID
application_credential_id = $APP_CRED_ID
application_credential_secret = $APP_CRED_SECRET
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
EOF

echo "rclone configuration created at ~/.config/rclone/rclone.conf"

# Test the configuration
echo "Testing the configuration..."
rclone lsd chi_tacc:

# Create mount point
echo "Creating mount point at /mnt/object..."
mkdir -p /mnt/object
chown -R $SUDO_USER:$SUDO_USER /mnt/object

echo ""
echo "Setup complete! You can now mount your object storage with:"
echo "rclone mount chi_tacc:your-container-name /mnt/object --read-only --allow-other --daemon"
echo ""
echo "To check the contents of your mount:"
echo "ls /mnt/object"
echo ""
echo "To unmount:"
echo "fusermount -u /mnt/object" 