#!/bin/bash
# Profile VRAM for all backbone configurations + distillation.
# After all jobs complete, generate plots with:
#   python -m src.profiling.plot_results --results-dir profiling_results
#
# Usage: bash launch_profile.sh

set -e
cd "$(dirname "$0")"

mkdir -p outputs profiling_results

CONFIGS=(
    configs/backbones/convnext_tiny.yaml
    configs/backbones/resnet50.yaml
    configs/backbones/efficientnet_b0.yaml
    configs/backbones/vit_small.yaml
    configs/backbones/deit_small.yaml
    configs/backbones/swin_tiny.yaml
    configs/backbones/mobilevit_small.yaml
    configs/distillation.yaml
)

for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)

    echo "Submitting profile-$name ($config)..."
    sbatch \
        --job-name="profile-$name" \
        --output="outputs/profile-${name}-%j.out" \
        --error="outputs/profile-${name}-%j.out" \
        profile_vram.sbatch "$config"
done

echo ""
echo "All profiling jobs submitted. Check status with: squeue -u \$USER"
echo "After completion, generate plots: python -m src.profiling.plot_results"
