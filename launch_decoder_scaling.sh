#!/bin/bash
# Launch decoder scaling ablation (2x and 4x parameter budgets) on SLURM.
# Usage: bash launch_decoder_scaling.sh

set -e
cd "$(dirname "$0")"

mkdir -p outputs

for scale in 2x 4x; do
    for decoder in lstm rnn gru transformer; do
        echo "Submitting dec-${decoder}-${scale} ..."
        sbatch \
            --job-name="dec-${decoder}-${scale}" \
            --output="outputs/dec_${decoder}_${scale}-%j.out" \
            --error="outputs/dec_${decoder}_${scale}-%j.out" \
            train_decoder_ablation.sbatch "$decoder" "$scale"
    done
done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"
