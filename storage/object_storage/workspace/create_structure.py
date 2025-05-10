#!/usr/bin/env python
# Script to initialize directory structure in object storage

import openstack
import os
import argparse

def create_container_structure(container_name, structure):
    """Create a directory structure in object storage container"""
    print(f"Connecting to OpenStack...")
    conn = openstack.connect()
    
    # Check if container exists, create if it doesn't
    try:
        conn.object_store.get_container_metadata(container_name)
        print(f"Container '{container_name}' exists")
    except Exception:
        print(f"Creating container '{container_name}'...")
        conn.object_store.create_container(name=container_name)
    
    # Create directory structure by creating empty marker objects
    for directory in structure:
        object_name = f"{directory}/.keep"
        print(f"Creating directory structure: {directory}")
        
        # Create temporary empty file
        with open(".keep", "w") as f:
            pass
        
        # Upload empty file to create directory
        conn.object_store.upload_object(
            container=container_name,
            name=object_name,
            filename=".keep"
        )
    
    # Remove temporary file
    if os.path.exists(".keep"):
        os.remove(".keep")
    
    print(f"✅ Directory structure created in container '{container_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create directory structure in object storage")
    parser.add_argument("--container", required=True, help="Name of the container")
    args = parser.parse_args()
    
    # Define the directory structure for ML datasets
    dataset_structure = [
        "datasets/train",
        "datasets/test",
        "datasets/validation",
        "models",
        "logs",
        "configs"
    ]
    
    create_container_structure(args.container, dataset_structure) 