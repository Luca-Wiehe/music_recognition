#!/usr/bin/env python3
"""
Evaluate a trained MusicTrOCR checkpoint on the test (or validation) set.

Reports SER, CER, sequence accuracy, and optionally prints per-sample
token comparisons.

Usage:
    # Evaluate on test split using the config stored in the checkpoint:
    python evaluate.py --checkpoint networks/checkpoints/deit_small/stage_1/best_checkpoint.pt

    # Override config (e.g. to change batch size or dataset):
    python evaluate.py --checkpoint best_checkpoint.pt --config configs/backbones/deit_small.yaml

    # Evaluate on validation split instead of test:
    python evaluate.py --checkpoint best_checkpoint.pt --split val

    # Limit number of batches (quick sanity check):
    python evaluate.py --checkpoint best_checkpoint.pt --max-batches 10

    # Print per-sample comparisons:
    python evaluate.py --checkpoint best_checkpoint.pt --verbose --max-batches 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.train import load_config, create_model, setup_data_loaders
from src.metrics import (
    compute_metrics,
    aggregate_metrics,
    strip_special_tokens,
    edit_distance,
)


# ------------------------------------------------------------------
# Core evaluation
# ------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_length: int = 512,
    max_batches: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run greedy decoding on *data_loader* and return aggregated metrics.

    Args:
        model:       Model with a ``generate()`` method.
        data_loader: DataLoader yielding (images, targets).
        device:      CUDA / CPU.
        max_length:  Maximum generation length.
        max_batches: Evaluate at most this many batches.
        verbose:     Print per-sample token comparisons.

    Returns:
        Aggregated dict with ``ser``, ``cer``, ``sequence_acc``,
        ``num_samples``.
    """
    dataset = data_loader.dataset
    pad_id = dataset.pad_token_id
    bos_id = dataset.bos_token_id
    eos_id = dataset.eos_token_id
    index_to_vocab = getattr(dataset, "index_to_vocabulary", None)

    model.eval()
    batch_metrics: list[dict] = []
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            predictions = model.generate(images, max_length=max_length)
            predictions_cpu = predictions.cpu()

            m = compute_metrics(
                predictions_cpu, targets, pad_id, bos_id, eos_id,
                index_to_vocab=index_to_vocab,
            )
            batch_metrics.append(m)

            if verbose:
                _print_batch_details(
                    predictions_cpu, targets, pad_id, bos_id, eos_id,
                    index_to_vocab, sample_offset=sample_count,
                )

            sample_count += images.shape[0]

            if (batch_idx + 1) % 10 == 0:
                running = aggregate_metrics(batch_metrics)
                print(f"  [{batch_idx + 1:>4} batches, {sample_count} samples] "
                      f"SER={running['ser']*100:.2f}%  "
                      f"CER={running['cer']*100:.2f}%  "
                      f"SeqAcc={running['sequence_acc']*100:.1f}%")

    return aggregate_metrics(batch_metrics)


def _print_batch_details(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int, bos_id: int, eos_id: int,
    index_to_vocab: dict | None,
    sample_offset: int = 0,
):
    """Print per-sample token-by-token comparisons."""
    for i in range(predictions.shape[0]):
        pred_ids = strip_special_tokens(predictions[i], pad_id, bos_id, eos_id)
        gt_ids = strip_special_tokens(targets[i], pad_id, bos_id, eos_id)

        ed = edit_distance(pred_ids, gt_ids)
        ser = ed / max(len(gt_ids), 1)
        exact = pred_ids == gt_ids

        print(f"\n--- Sample {sample_offset + i + 1} ---")
        print(f"  GT  length: {len(gt_ids)}  Pred length: {len(pred_ids)}")
        print(f"  Edit dist:  {ed}   SER: {ser*100:.1f}%   Exact: {'Y' if exact else 'N'}")

        if index_to_vocab:
            gt_str = " ".join(index_to_vocab.get(t, f"<{t}>") for t in gt_ids[:30])
            pred_str = " ".join(index_to_vocab.get(t, f"<{t}>") for t in pred_ids[:30])
            if len(gt_ids) > 30:
                gt_str += " …"
            if len(pred_ids) > 30:
                pred_str += " …"
            print(f"  GT:   {gt_str}")
            print(f"  Pred: {pred_str}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OMR checkpoint with SER/CER/SeqAcc"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default=None,
                        help="Override config YAML (default: use config from checkpoint)")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Maximum batches to evaluate")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample token comparisons")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results as JSON to this path")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_config = ckpt.get("config", {})
    epoch = ckpt.get("epoch", "?")
    ckpt_loss = ckpt.get("loss", "?")
    print(f"  Epoch: {epoch}   Loss: {ckpt_loss}")

    # Config
    if args.config:
        config = load_config(args.config)
    else:
        config = ckpt_config
    if not config:
        print("Error: no config in checkpoint and --config not provided")
        sys.exit(1)

    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Data
    train_loader, val_loader, test_loader, vocab_size = setup_data_loaders(config)
    eval_loader = val_loader if args.split == "val" else test_loader
    dataset = eval_loader.dataset
    print(f"Evaluating on {args.split} split: {len(dataset)} samples")

    # Model
    model = create_model(
        config, vocab_size,
        pad_token_id=dataset.pad_token_id,
        bos_token_id=dataset.bos_token_id,
        eos_token_id=dataset.eos_token_id,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']['type']}  Params: {total_params:,}")

    max_len = config.get("model", {}).get("params", {}).get("max_seq_len", 512)

    # Evaluate
    print(f"\n{'='*60}")
    print(f"  Evaluating ({args.split} set)")
    print(f"{'='*60}")

    metrics = evaluate(
        model, eval_loader, device,
        max_length=max_len,
        max_batches=args.max_batches,
        verbose=args.verbose,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Results ({args.split} set, n={metrics['num_samples']})")
    print(f"{'='*60}")
    print(f"  SER:          {metrics['ser']*100:.2f}%")
    print(f"  CER:          {metrics['cer']*100:.2f}%")
    print(f"  Sequence Acc: {metrics['sequence_acc']*100:.1f}%")
    print(f"{'='*60}")

    # Save JSON
    if args.output_json:
        result = {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "epoch": epoch,
            **metrics,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
