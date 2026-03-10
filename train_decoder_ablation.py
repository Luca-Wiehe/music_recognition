#!/usr/bin/env python3
"""
Training script for decoder ablation study (CRNN encoder + CTC loss).

Usage:
    python train_decoder_ablation.py --decoder lstm
    python train_decoder_ablation.py --decoder transformer --wandb-project music-recognition
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.unified_dataset import load_pdmx_synth, create_collate_fn
from networks.decoders import create_ablation_model, DECODERS, SCALE_CONFIGS
import src.utils.utils as utils

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Decoder ablation training (CTC)")
    parser.add_argument("--decoder", type=str, required=True,
                        choices=list(DECODERS.keys()),
                        help="Decoder variant to train")
    parser.add_argument("--scale", type=str, default="1x",
                        choices=list(SCALE_CONFIGS.keys()),
                        help="Decoder parameter scale: 1x (~6.3M), 2x (~12.6M), 4x (~25.2M)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--img-height", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="guangyangmusic/PDMX-Synth")
    parser.add_argument("--tokenizer", type=str, default="guangyangmusic/legato")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                    best_loss, patience_counter, config, ckpt_dir, is_best):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_loss": best_loss,
        "patience_counter": patience_counter,
        "config": config,
    }
    torch.save(ckpt, ckpt_dir / "latest_checkpoint.pt")
    if is_best:
        torch.save(ckpt, ckpt_dir / "best_checkpoint.pt")
        print(f"  Saved best checkpoint (val_loss={best_loss:.6f})")


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    print(f"Loading dataset {args.dataset} ...")
    train_ds, val_ds, _ = load_pdmx_synth(
        dataset_id=args.dataset,
        tokenizer_id=args.tokenizer,
        img_height=args.img_height,
    )

    vocab_size = len(train_ds.vocabulary_to_index)
    pad_token_id = train_ds.pad_token_id
    print(f"Vocab: {vocab_size}, PAD={pad_token_id}")
    print(f"Splits: train={len(train_ds)}, val={len(val_ds)}")

    collate_fn = create_collate_fn(pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=1, pin_memory=True)

    # --- Model ---
    # Images are RGB (3-channel) from UnifiedDataset
    hparams = {"optimizer": {"learning_rate": args.lr, "weight_decay": args.weight_decay}}
    model = create_ablation_model(hparams, vocab_size,
                                  decoder_type=args.decoder, scale=args.scale,
                                  in_channels=3)
    model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    print(f"Decoder: {args.decoder} | Total params: {total_p:,} | Decoder params: {dec_p:,}")

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    optim_steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = optim_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.03)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Scheduler: {warmup_steps} warmup / {total_steps} total steps")

    # --- CTC loss ---
    ctc_loss = nn.CTCLoss(blank=pad_token_id, zero_infinity=True)

    # --- AMP ---
    scaler = torch.amp.GradScaler("cuda")

    # --- Wandb ---
    wandb_run = None
    if args.wandb_project and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"decoder-ablation-{args.decoder}-{args.scale}",
            config=vars(args),
            tags=["decoder-ablation", args.decoder, args.scale, "crnn", "ctc"],
        )

    # --- Checkpointing ---
    ckpt_dir = Path(f"networks/checkpoints/decoder_ablation/{args.decoder}/{args.scale}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0

    # Auto-resume
    resume_path = args.resume
    if not resume_path:
        auto = ckpt_dir / "latest_checkpoint.pt"
        if auto.exists():
            resume_path = str(auto)

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.6f}")

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        train_loop = utils.create_tqdm_bar(
            train_loader, desc=f"Train [{epoch+1}/{args.epochs}]")

        for batch_idx, (images, targets) in train_loop:
            images, targets = images.to(device), targets.to(device)

            with torch.amp.autocast("cuda"):
                preds = model(images).permute(1, 0, 2)  # (T, B, C)
                input_lengths = torch.full(
                    (preds.shape[1],), preds.shape[0],
                    dtype=torch.int32, device=device)
                target_lengths = (targets != pad_token_id).sum(dim=1)
                loss = ctc_loss(preds, targets, input_lengths, target_lengths)

            scaler.scale(loss / args.grad_accum_steps).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0 \
                    or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item()
            train_loop.set_postfix(
                loss=f"{train_loss / (batch_idx + 1):.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_train = train_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_loop = utils.create_tqdm_bar(
            val_loader, desc=f"Val   [{epoch+1}/{args.epochs}]")

        with torch.no_grad():
            for batch_idx, (images, targets) in val_loop:
                images, targets = images.to(device), targets.to(device)

                with torch.amp.autocast("cuda"):
                    preds = model(images).permute(1, 0, 2)
                    input_lengths = torch.full(
                        (preds.shape[1],), preds.shape[0],
                        dtype=torch.int32, device=device)
                    target_lengths = (targets != pad_token_id).sum(dim=1)
                    loss = ctc_loss(preds, targets, input_lengths, target_lengths)

                val_loss += loss.item()
                val_loop.set_postfix(
                    val_loss=f"{val_loss / (batch_idx + 1):.6f}")

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")

        # Checkpoint + early stopping
        is_best = avg_val < best_loss
        if is_best:
            best_loss = avg_val
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                        best_loss, patience_counter, vars(args),
                        ckpt_dir, is_best)

        if wandb_run:
            wandb_run.log({
                "train/loss": avg_train,
                "val/loss": avg_val,
                "best_val_loss": best_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1,
            })

        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete. Best val_loss: {best_loss:.6f}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
