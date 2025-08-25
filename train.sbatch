#!/bin/bash
#SBATCH --job-name="OMR Training"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=72G
#SBATCH --time=0-16:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/stud/wiel/music_recognition/outputs/luca_model-training-%j.out
#SBATCH --error=/home/stud/wiel/music_recognition/outputs/luca_model-training-%j.out
srun python train.py --config configs/luca_model.yaml