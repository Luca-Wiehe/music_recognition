#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=NONE

# Usage:
#   sbatch --job-name="dec-lstm" --output=outputs/dec_lstm-%j.out --error=outputs/dec_lstm-%j.out \
#     train_decoder_ablation.sbatch lstm [1x|2x|4x]

DECODER=${1:?Usage: sbatch train_decoder_ablation.sbatch <lstm|rnn|gru|transformer> [scale]}
SCALE=${2:-1x}

source activate music_recognition
cd /home/stud/wiel/music_recognition
srun python train_decoder_ablation.py --decoder "$DECODER" --scale "$SCALE" --wandb-project music-recognition
