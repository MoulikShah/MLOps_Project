# Storage Management in MLOps Project

This directory contains resources for managing different types of storage in our MLOps project on Chameleon Cloud.

## Directory Structure

- **object_storage/**: Resources for working with Chameleon Cloud object storage
  - `object_persist_group14.ipynb`: Jupyter notebook with instructions and code for managing object storage
  
- **block_storage/**: Resources for working with block storage volumes

## Object Storage

Object storage is ideal for:
- Storing and retrieving large datasets
- Sharing data between multiple compute nodes
- Long-term persistence of data beyond the lifespan of compute instances

Key benefits:
- Data is replicated for durability
- Accessible from any compute instance
- Can be mounted as a filesystem using tools like rclone

## Block Storage

Block storage is suited for:
- Direct attachment to a single compute instance
- Use cases requiring filesystem operations and low latency
- Database storage and other applications requiring consistent I/O performance

## Usage

See the notebooks in each directory for detailed instructions on how to use each storage type. 