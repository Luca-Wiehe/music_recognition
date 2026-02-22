#!/bin/bash
# Launch all backbone sweep training jobs in parallel on SLURM.
# Usage: bash launch_sweep.sh

set -e
cd "$(dirname "$0")"

mkdir -p outputs

CONFIGS=(
    configs/backbones/convnext_tiny.yaml
    configs/backbones/resnet50.yaml
    configs/backbones/efficientnet_b0.yaml
    configs/backbones/vit_small.yaml
    configs/backbones/deit_small.yaml
    configs/backbones/swin_tiny.yaml
    configs/backbones/mobilevit_small.yaml
)

for config in "${CONFIGS[@]}"; do
    # Extract backbone name from filename (e.g., "resnet50" from "configs/backbones/resnet50.yaml")
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
