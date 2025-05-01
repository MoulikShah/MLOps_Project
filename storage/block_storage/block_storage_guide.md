# Block Storage Guide for Chameleon Cloud

This guide explains how to create, attach, and use block storage volumes with Chameleon Cloud compute instances.

## Creating a Block Storage Volume

1. Log in to the CHI@TACC Horizon dashboard
2. Navigate to Project > Volumes > Volumes
3. Click "Create Volume"
4. Specify:
   - Volume Name (e.g., "data-volume")
   - Description (optional)
   - Volume Source (typically "No source, empty volume")
   - Size (in GB)
   - Availability Zone (typically dependent on your compute resources)
5. Click "Create Volume"

## Attaching a Volume to a Compute Instance

1. From the Volumes page, find your volume
2. Click the dropdown menu on the right side of the volume entry
3. Select "Manage Attachments"
4. Choose your instance from the "Attach to Instance" dropdown
5. Specify the device name (e.g., `/dev/vdb` - this will be the device path on your instance)
6. Click "Attach Volume"

## Preparing and Mounting the Volume

Once attached, you need to create a filesystem and mount the volume:

```bash
# Check if the volume is visible (should show your device, e.g., /dev/vdb)
lsblk

# Format the volume with a filesystem (only needed the first time you use the volume)
# WARNING: This will erase any existing data on the volume
sudo mkfs.ext4 /dev/vdb

# Create a mount point
sudo mkdir -p /mnt/data

# Mount the volume
sudo mount /dev/vdb /mnt/data

# Change ownership to your user (if needed)
sudo chown -R $USER:$USER /mnt/data
```

## Automounting on System Boot

To ensure the volume is mounted automatically when the instance boots:

1. Get the UUID of the volume:
```bash
sudo blkid /dev/vdb
```

2. Edit the fstab file:
```bash
sudo nano /etc/fstab
```

3. Add a line like this:
```
UUID=your-uuid-here /mnt/data ext4 defaults,nofail 0 2
```

4. Save and exit

## Detaching a Volume

Before detaching a volume from an instance:

1. Unmount the volume:
```bash
sudo umount /mnt/data
```

2. In the Horizon dashboard, go to Volumes
3. Click "Manage Attachments"
4. Click "Detach Volume"

## Using Block Storage with Docker

To make your block storage available to Docker containers:

```bash
docker run -d \
  -v /mnt/data:/data \
  your-image-name
```

This will mount the `/mnt/data` directory from your host to `/data` in the container.

## Best Practices

1. Always unmount volumes before detaching them
2. Back up important data regularly
3. Use volumes for data that needs to persist beyond the life of the instance
4. Consider performance requirements when sizing and using volumes 