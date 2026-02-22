#!/bin/bash
#SBATCH --job-name="OMR Training"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=72G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/stud/wiel/music_recognition/outputs/luca_model-training-%j.out
#SBATCH --error=/home/stud/wiel/music_recognition/outputs/luca_model-training-%j.out
source activate music_recognition
srun python -m src.train --config configs/luca_model.yaml