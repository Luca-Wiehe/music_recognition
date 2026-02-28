#!/usr/bin/env python3
"""
Test the rendering pipeline with the DeiT-Small pitch-only checkpoint.

Loads the model, runs inference on a few validation samples, decodes
BPE token IDs back to ABC notation, and renders via Verovio.
"""

import torch
import yaml
import numpy as np
from pathlib import Path
from PIL import Image

from src.train import create_model, setup_data_loaders
from src.rendering import render_prediction
from src.metrics import strip_special_tokens


def main():
    config_path = "configs/backbones/deit_small_pitch_only.yaml"
    checkpoint_path = "networks/checkpoints/deit_small_pitch_only/stage_1/best_checkpoint.pt"

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Loading config from {config_path}")
    print(f"Loading checkpoint from {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (we need the tokenizer and a few samples)
    train_loader, val_loader, test_loader, vocab_size = setup_data_loaders(config)
    dataset = val_loader.dataset
    tokenizer = dataset.tokenizer

    print(f"Vocab size: {vocab_size}")
    print(f"PAD={dataset.pad_token_id}, BOS={dataset.bos_token_id}, EOS={dataset.eos_token_id}")

    # Create model and load checkpoint
    model = create_model(
        config, vocab_size,
        pad_token_id=dataset.pad_token_id,
        bos_token_id=dataset.bos_token_id,
        eos_token_id=dataset.eos_token_id,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Run inference on a few validation samples
    n_samples = 3
    output_dir = Path("rendering_test_output")
    output_dir.mkdir(exist_ok=True)

    for i in range(n_samples):
        print(f"\n--- Sample {i+1}/{n_samples} ---")

        image_tensor, target_ids = dataset[i]
        image_batch = image_tensor.unsqueeze(0).to(device)

        # Generate prediction
        with torch.no_grad():
            pred_ids = model.generate(image_batch, max_length=2048)

        # Strip special tokens
        pred_tokens = strip_special_tokens(
            pred_ids[0], dataset.pad_token_id, dataset.bos_token_id, dataset.eos_token_id
        )
        gt_tokens = strip_special_tokens(
            target_ids, dataset.pad_token_id, dataset.bos_token_id, dataset.eos_token_id
        )

        # Decode BPE tokens → ABC string
        pred_abc = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        gt_abc = tokenizer.decode(gt_tokens, skip_special_tokens=True)

        print(f"Ground truth ABC (first 200 chars):\n  {gt_abc[:200]}")
        print(f"Predicted ABC (first 200 chars):\n  {pred_abc[:200]}")

        # Render ground truth
        gt_image = render_prediction(gt_abc, format="abc")
        if gt_image is not None:
            gt_path = output_dir / f"sample_{i+1}_gt.png"
            Image.fromarray(gt_image).save(gt_path)
            print(f"Ground truth rendered → {gt_path} (shape: {gt_image.shape})")
        else:
            print("Ground truth rendering FAILED")

        # Render prediction
        pred_image = render_prediction(pred_abc, format="abc")
        if pred_image is not None:
            pred_path = output_dir / f"sample_{i+1}_pred.png"
            Image.fromarray(pred_image).save(pred_path)
            print(f"Prediction rendered → {pred_path} (shape: {pred_image.shape})")
        else:
            print("Prediction rendering FAILED")

    print(f"\nDone! Check {output_dir}/ for rendered outputs.")


if __name__ == "__main__":
    main()
