#!/bin/bash
# Launch all efficiency sweep training jobs on SLURM.
# Usage: bash launch_efficiency_sweep.sh

set -e
cd "$(dirname "$0")"

mkdir -p outputs

CONFIGS=(
    configs/efficiency/deit_frozen.yaml
    configs/efficiency/deit_lora_r4.yaml
    configs/efficiency/deit_lora_r8.yaml
    configs/efficiency/deit_lora_r16.yaml
    configs/efficiency/deit_qlora_r8.yaml
)

for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)

    echo "Submitting $name ($config)..."
    sbatch \
        --job-name="$name" \
        --output="outputs/${name}-%j.out" \
        --error="outputs/${name}-%j.out" \
        train_backbone.sbatch "$config"
done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"
