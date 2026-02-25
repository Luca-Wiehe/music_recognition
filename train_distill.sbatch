#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=72G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=NONE
#SBATCH --job-name="distill"
#SBATCH --output=outputs/distill-%j.out
#SBATCH --error=outputs/distill-%j.out

# Usage: sbatch train_distill.sbatch [configs/distillation.yaml]

CONFIG=${1:-configs/distillation.yaml}

source activate music_recognition
cd /home/stud/wiel/music_recognition
srun python -m src.train --config "$CONFIG"
