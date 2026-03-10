#!/bin/bash
# Launch all decoder ablation training jobs in parallel on SLURM.
# Usage: bash launch_decoder_ablation.sh

set -e
cd "$(dirname "$0")"

mkdir -p outputs

for decoder in lstm rnn gru transformer; do
    echo "Submitting decoder-ablation-${decoder} ..."
    sbatch \
        --job-name="dec-${decoder}" \
        --output="outputs/dec_${decoder}-%j.out" \
        --error="outputs/dec_${decoder}-%j.out" \
        train_decoder_ablation.sbatch "$decoder"
done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"
