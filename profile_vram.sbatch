#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=72G
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=NONE

# Usage: sbatch --job-name="profile-resnet50" --output=outputs/profile-resnet50-%j.out profile_vram.sbatch configs/backbones/resnet50.yaml
# Or use launch_profile.sh to submit all backbones at once.

CONFIG=${1:?Usage: sbatch profile_vram.sbatch <config.yaml>}

source activate music_recognition
cd /home/stud/wiel/music_recognition
srun python -m src.profiling.vram_profiler --config "$CONFIG" --output-dir profiling_results
