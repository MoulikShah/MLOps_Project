#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=lantern_gen
#SBATCH --output=output_%j.txt
#SBATCH --error=output_%j.txt
source /home/aryan/miniconda3/etc/profile.d/conda.sh
conda activate mlops
cd /home/aryan/MLOps_Project/
python3 try_ethnicity_split.py