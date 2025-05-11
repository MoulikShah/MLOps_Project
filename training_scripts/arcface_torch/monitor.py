from pyrsmi import rocminfo

gpus = rocminfo.get_device_info()
for i, gpu in enumerate(gpus):
    print(f"[GPU {i}] Name: {gpu.name}, VRAM: {gpu.vram_total} MB")
