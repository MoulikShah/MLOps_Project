#!/bin/bash


echo "🔄 Updating apt packages..."
sudo apt update

echo "⬇️ Downloading AMDGPU installer..."
wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb

echo "📦 Installing AMDGPU installer..."
sudo apt -y install ./amdgpu-install_6.3.60303-1_all.deb

echo "🔄 Updating apt packages again..."
sudo apt update

echo "🧱 Installing amdgpu-dkms drivers..."
sudo amdgpu-install --usecase=dkms

echo "🛠 Installing ROCm SMI utility..."
sudo apt -y install rocm-smi

echo "👤 Adding user '$USER' to video and render groups..."
sudo usermod -aG video,render $USER

echo "✅ Base ROCm setup complete."
echo "🔁 System will reboot now to apply changes."
echo "Reconnect after reboot and run 'rocm-smi' to confirm GPU access."

rm amdgpu-install_6.3.60303-1_all.deb
sleep 3
sudo reboot
