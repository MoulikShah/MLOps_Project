# Object Storage Workspace

This directory contains scripts and notebooks for working with Chameleon Cloud object storage.

## Contents

- `train.py` - Example training script that loads data from mounted object storage
- `create_structure.py` - Script to initialize directory structure in object storage

## Getting Started

### 1. Set up the environment

Make sure you have the required Python packages:

```bash
pip install openstacksdk python-swiftclient torch torchvision
```

### 2. Mount the object storage

First, set up rclone using the `setup_rclone.sh` script in the parent directory:

```bash
cd ..
sudo ./setup_rclone.sh
```

Then mount your object storage container:

```bash
rclone mount chi_tacc:object-persist-group14 /mnt/object --allow-other --daemon
```

### 3. Initialize the directory structure

Create the necessary directory structure in your object storage container:

```bash
python create_structure.py --container object-persist-group14
```

### 4. Start the Docker containers

From the parent directory, start the Docker containers using docker-compose:

```bash
cd ..
docker-compose up -d
```

### 5. Upload your data

Upload your dataset to the object storage. You can use the Jupyter notebook in the Docker container or the OpenStack SDK directly:

```python
import openstack
import os

conn = openstack.connect()
container_name = 'object-persist-group14'
local_dataset_path = '/path/to/your/dataset'

# Upload files
for root, dirs, files in os.walk(local_dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        object_name = os.path.relpath(file_path, start=local_dataset_path)
        
        conn.object_store.upload_object(
            container=container_name,
            name=f"datasets/{object_name}",
            filename=file_path
        )
```

### 6. Run the training script

With the data in place, you can run the training script in the Docker container:

```bash
docker exec -it model-training python /workspace/train.py
```

## Notes

- Object storage is mounted in read-only mode by default for data safety
- For training results persistence, models are saved to the local `models` directory
- Remember to unmount the object storage when done: `fusermount -u /mnt/object` 