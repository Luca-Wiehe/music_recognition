#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=NONE

# Usage:
#   sbatch --job-name="deit_small" --output=outputs/deit_small-%j.out --error=outputs/deit_small-%j.out \
#     train_backbone.sbatch facebook/deit-small-patch16-224
#
# With encoder mode:
#   sbatch --job-name="deit_lora_r8" --output=outputs/deit_lora_r8-%j.out --error=outputs/deit_lora_r8-%j.out \
#     train_backbone.sbatch facebook/deit-small-patch16-224 --encoder-mode lora --lora-rank 8

ENCODER=${1:?Usage: sbatch train_backbone.sbatch <encoder-name> [extra args...]}
shift

source activate music_recognition
cd /home/stud/wiel/music_recognition
pip install -q peft bitsandbytes
srun python -m src.train --encoder "$ENCODER" --wandb-project music-recognition "$@"
