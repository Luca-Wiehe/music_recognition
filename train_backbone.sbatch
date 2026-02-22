#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=72G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=NONE

# Usage: sbatch --job-name="resnet50" --output=outputs/resnet50-%j.out --error=outputs/resnet50-%j.out train_backbone.sbatch configs/backbones/resnet50.yaml
# Or use launch_sweep.sh to submit all backbones at once.

CONFIG=${1:?Usage: sbatch train_backbone.sbatch <config.yaml>}

source activate music_recognition
cd /home/stud/wiel/music_recognition
srun python -m src.train --config "$CONFIG"
