#!/usr/bin/env python3
"""
VRAM profiler for MusicTrOCR configurations.

Profiles a model configuration and reports:
  - Peak VRAM during training and inference
  - VRAM breakdown: parameters, gradients, optimizer states, activations
  - Training and inference throughput
  - Model parameter counts

Results are saved as JSON for downstream analysis (see plot_results.py).

Usage:
    python -m src.profiling.vram_profiler --config configs/backbones/deit_small.yaml
    python -m src.profiling.vram_profiler --config configs/distillation.yaml
    python -m src.profiling.vram_profiler --config configs/backbones/resnet50.yaml --checkpoint networks/checkpoints/resnet50/stage_1/best_checkpoint.pt
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.train import load_config, create_model, setup_data_loaders, setup_optimizer_and_scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def _param_memory_bytes(model: torch.nn.Module, only_trainable: bool = False) -> int:
    """Compute memory occupied by model parameters on the current device."""
    total = 0
    for p in model.parameters():
        if only_trainable and not p.requires_grad:
            continue
        total += p.nelement() * p.element_size()
    return total


def _optimizer_state_memory_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Compute memory occupied by optimizer state tensors."""
    total = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state is None:
                continue
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    total += v.nelement() * v.element_size()
    return total


def _gpu_info(device: torch.device) -> dict:
    """Return basic GPU information."""
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "total_memory_mb": round(_bytes_to_mb(props.total_mem), 1),
    }


# ---------------------------------------------------------------------------
# Core profiling routines
# ---------------------------------------------------------------------------

def profile_model_params(model: torch.nn.Module) -> dict:
    """Count parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    encoder_params = 0
    decoder_params = 0
    if hasattr(model, "vision_encoder"):
        encoder_params = sum(p.numel() for p in model.vision_encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
    elif hasattr(model, "student"):
        # Distillation wrapper — report student only
        encoder_params = sum(p.numel() for p in model.student.vision_encoder.parameters())
        decoder_params = sum(p.numel() for p in model.student.decoder.parameters())

    return {
        "total_params": total,
        "trainable_params": trainable,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
    }


def profile_vram_breakdown(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    device: torch.device,
    config: dict,
    num_warmup: int = 3,
    num_profile: int = 5,
) -> dict:
    """
    Measure VRAM breakdown during training.

    Runs *num_warmup* batches to let the CUDA allocator settle, then
    profiles *num_profile* batches and records peak memory.

    Returns a dict with memory figures in MB.
    """
    accum_steps = config["training"].get("gradient_accumulation_steps", 1)
    use_amp = config.get("hardware", {}).get("mixed_precision", False)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_clip = config["training"].get("grad_clip_norm", None)

    # ------------------------------------------------------------------
    # 1.  Analytical breakdown (device-independent)
    # ------------------------------------------------------------------
    param_bytes = _param_memory_bytes(model)
    trainable_bytes = _param_memory_bytes(model, only_trainable=True)
    # Gradients occupy the same shape as trainable params, always FP32
    gradient_bytes = sum(
        p.numel() * 4 for p in model.parameters() if p.requires_grad
    )

    # ------------------------------------------------------------------
    # 2.  Warmup batches  (populates optimizer states, JIT caches, etc.)
    # ------------------------------------------------------------------
    model.train()
    loader_iter = iter(train_loader)

    for i in range(num_warmup):
        batch = next(loader_iter)
        optimizer.zero_grad()
        if hasattr(model, "training_step"):
            loss = model.training_step(
                batch, device, config, epoch=0, batch_idx=i,
                scaler=scaler, accumulation_steps=accum_steps,
            )
        else:
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            outputs = model(images, targets)
            loss = F.cross_entropy(
                outputs["logits"].reshape(-1, model.vocab_size),
                outputs["decoder_target"].reshape(-1),
                ignore_index=model.PAD_TOKEN_ID,
            )
            loss.backward()

        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    # Measure optimizer state memory after states are populated
    optimizer_bytes = _optimizer_state_memory_bytes(optimizer)

    # ------------------------------------------------------------------
    # 3.  Profiling batches  (measure peak VRAM + throughput)
    # ------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    total_time = 0.0
    total_tokens = 0

    for i in range(num_profile):
        batch = next(loader_iter)
        _, targets = batch
        total_tokens += targets.numel()

        optimizer.zero_grad()

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if hasattr(model, "training_step"):
            loss = model.training_step(
                batch, device, config, epoch=0, batch_idx=i,
                scaler=scaler, accumulation_steps=accum_steps,
            )
        else:
            images, targets_dev = batch[0].to(device), batch[1].to(device)
            outputs = model(images, targets_dev)
            loss = F.cross_entropy(
                outputs["logits"].reshape(-1, model.vocab_size),
                outputs["decoder_target"].reshape(-1),
                ignore_index=model.PAD_TOKEN_ID,
            )
            loss.backward()

        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        torch.cuda.synchronize(device)
        total_time += time.perf_counter() - t0

    peak_training_bytes = torch.cuda.max_memory_allocated(device)
    activation_bytes = max(
        0, peak_training_bytes - param_bytes - gradient_bytes - optimizer_bytes
    )

    # ------------------------------------------------------------------
    # 4.  Inference profiling  (peak VRAM for generate())
    # ------------------------------------------------------------------
    peak_inference_bytes = 0
    inference_time = 0.0
    inference_batches = 0

    model.eval()
    generate_fn = getattr(model, "generate", None)
    if generate_fn is not None:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        loader_iter2 = iter(train_loader)

        with torch.no_grad():
            for i in range(min(num_profile, 3)):
                batch = next(loader_iter2)
                images = batch[0].to(device)

                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                _ = generate_fn(images, max_length=128)
                torch.cuda.synchronize(device)
                inference_time += time.perf_counter() - t0
                inference_batches += 1

        peak_inference_bytes = torch.cuda.max_memory_allocated(device)

    # ------------------------------------------------------------------
    # 5.  Assemble results
    # ------------------------------------------------------------------
    train_it_per_sec = num_profile / total_time if total_time > 0 else 0
    train_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    inf_it_per_sec = inference_batches / inference_time if inference_time > 0 else 0

    return {
        "vram": {
            "peak_training_mb": round(_bytes_to_mb(peak_training_bytes), 1),
            "parameters_mb": round(_bytes_to_mb(param_bytes), 1),
            "gradients_mb": round(_bytes_to_mb(gradient_bytes), 1),
            "optimizer_states_mb": round(_bytes_to_mb(optimizer_bytes), 1),
            "activations_estimated_mb": round(_bytes_to_mb(activation_bytes), 1),
            "peak_inference_mb": round(_bytes_to_mb(peak_inference_bytes), 1),
        },
        "throughput": {
            "train_it_per_sec": round(train_it_per_sec, 3),
            "train_tokens_per_sec": round(train_tokens_per_sec, 1),
            "inference_it_per_sec": round(inf_it_per_sec, 3),
        },
    }


def load_best_val_loss(checkpoint_path: str) -> float | None:
    """Extract best validation loss from a training checkpoint, if it exists."""
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("best_loss", ckpt.get("loss", None))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def profile(config_path: str, checkpoint_path: str | None = None,
            output_dir: str = "profiling_results",
            num_warmup: int = 3, num_profile: int = 5) -> dict:
    """
    Profile a single model configuration.

    Args:
        config_path:     Path to a YAML config file.
        checkpoint_path: Optional path to a trained checkpoint (to read best
                         val loss; the profiler always creates a fresh model
                         for VRAM measurement).
        output_dir:      Directory to save the JSON result.
        num_warmup:      Warmup batches before measurement.
        num_profile:     Batches to average over for throughput.

    Returns:
        The profiling result dict (also saved to disk).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("VRAM profiling requires a CUDA GPU")

    device = torch.device("cuda")
    config = load_config(config_path)

    # ---- data ----
    train_loader, val_loader, _, vocab_size = setup_data_loaders(config)
    train_dataset = train_loader.dataset

    # ---- model ----
    model = create_model(
        config, vocab_size,
        pad_token_id=train_dataset.pad_token_id,
        bos_token_id=train_dataset.bos_token_id,
        eos_token_id=train_dataset.eos_token_id,
    )
    model.to(device)

    # ---- optimizer ----
    optimizer, _ = setup_optimizer_and_scheduler(
        model, config, steps_per_epoch=len(train_loader),
    )

    # ---- profile ----
    param_info = profile_model_params(model)
    vram_info = profile_vram_breakdown(
        model, optimizer, train_loader, device, config,
        num_warmup=num_warmup, num_profile=num_profile,
    )

    # ---- best val loss from checkpoint ----
    best_val_loss = None
    if checkpoint_path:
        best_val_loss = load_best_val_loss(checkpoint_path)
    else:
        # Try auto-detecting from checkpoint_dir in config
        ckpt_dir = config.get("training", {}).get("checkpoint_dir", "")
        auto_best = Path(ckpt_dir) / "stage_1" / "best_checkpoint.pt"
        if auto_best.exists():
            best_val_loss = load_best_val_loss(str(auto_best))

    # ---- backbone name ----
    model_cfg = config["model"]
    if model_cfg["type"] == "Distillation":
        backbone = model_cfg["student"]["params"]["vision_model_name"]
    else:
        backbone = model_cfg.get("params", {}).get("vision_model_name", "unknown")

    # ---- assemble result ----
    result = {
        "config_path": config_path,
        "model_type": model_cfg["type"],
        "backbone": backbone,
        "model": param_info,
        "training": {
            "batch_size": config["training"]["batch_size"],
            "effective_batch_size": (
                config["training"]["batch_size"]
                * config["training"].get("gradient_accumulation_steps", 1)
            ),
            "img_height": config.get("data", {}).get("img_height", 128),
            "mixed_precision": config.get("hardware", {}).get("mixed_precision", False),
        },
        "gpu": _gpu_info(device),
        **vram_info,
    }
    if best_val_loss is not None:
        result["best_val_loss"] = round(best_val_loss, 6)

    # ---- save ----
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(config_path).stem
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nProfile saved to {out_path}")

    # ---- print summary ----
    _print_summary(result)

    return result


def _print_summary(result: dict):
    """Pretty-print the profiling result."""
    v = result["vram"]
    m = result["model"]
    t = result["throughput"]

    print("\n" + "=" * 60)
    print(f"  VRAM Profile: {result['backbone']}")
    print(f"  Model type:   {result['model_type']}")
    print("=" * 60)

    print(f"\n  Parameters:        {m['total_params']:>12,}")
    print(f"    Encoder:         {m['encoder_params']:>12,}")
    print(f"    Decoder:         {m['decoder_params']:>12,}")
    print(f"    Trainable:       {m['trainable_params']:>12,}")

    print(f"\n  VRAM Breakdown (training):")
    print(f"    Parameters:      {v['parameters_mb']:>10.1f} MB")
    print(f"    Gradients:       {v['gradients_mb']:>10.1f} MB")
    print(f"    Optimizer:       {v['optimizer_states_mb']:>10.1f} MB")
    print(f"    Activations:     {v['activations_estimated_mb']:>10.1f} MB")
    print(f"    ─────────────────────────────")
    print(f"    Peak training:   {v['peak_training_mb']:>10.1f} MB")
    print(f"    Peak inference:  {v['peak_inference_mb']:>10.1f} MB")

    print(f"\n  Throughput:")
    print(f"    Training:        {t['train_it_per_sec']:>10.2f} it/s")
    print(f"    Tokens/s:        {t['train_tokens_per_sec']:>10.0f}")
    print(f"    Inference:       {t['inference_it_per_sec']:>10.2f} it/s")

    if "best_val_loss" in result:
        print(f"\n  Best val loss:     {result['best_val_loss']:>10.6f}")

    gpu = result["gpu"]
    util = v["peak_training_mb"] / gpu["total_memory_mb"] * 100
    print(f"\n  GPU: {gpu['name']} ({gpu['total_memory_mb']:.0f} MB)")
    print(f"  VRAM utilisation:  {util:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Profile VRAM usage for a MusicTrOCR configuration"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (for best val loss)")
    parser.add_argument("--output-dir", type=str, default="profiling_results",
                        help="Directory to save JSON results")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup batches")
    parser.add_argument("--batches", type=int, default=5,
                        help="Number of profiling batches")
    args = parser.parse_args()

    profile(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_warmup=args.warmup,
        num_profile=args.batches,
    )


if __name__ == "__main__":
    main()
