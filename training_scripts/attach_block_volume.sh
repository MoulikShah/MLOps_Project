#!/bin/bash

set -e

source ~/openrc

VOLUME_NAME="block-persist-group14"

echo "Looking up volume ID for volume named '$VOLUME_NAME'..."
VOLUME_ID=$(openstack volume list --name "$VOLUME_NAME" -f value -c ID)

if [[ -z "$VOLUME_ID" ]]; then
  echo "ERROR: Volume '$VOLUME_NAME' not found."
  exit 1
fi

echo "Found volume ID: $VOLUME_ID"

# Fetch this instance's server ID via OpenStack metadata service
echo "Fetching current server ID from OpenStack metadata..."
SERVER_ID=$(curl -s http://169.254.169.254/openstack/latest/meta_data.json | jq -r '.uuid')

if [[ -z "$SERVER_ID" || "$SERVER_ID" == "null" ]]; then
  echo "ERROR: Could not determine server ID from metadata service."
  exit 1
fi

echo "Attaching volume $VOLUME_NAME to server ID $SERVER_ID..."
openstack server add volume "$SERVER_ID" "$VOLUME_ID"

echo "✅ Volume attached successfully to server $SERVER_ID."
