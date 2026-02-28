#!/usr/bin/env python3
"""
Flexible training script for optical music recognition models.

Usage:
    # CLI mode (no config file needed):
    python -m src.train --encoder facebook/deit-small-patch16-224
    python -m src.train --encoder facebook/deit-small-patch16-224 --encoder-mode lora --lora-rank 8
    python -m src.train --encoder facebook/deit-small-patch16-224 --strip-non-pitch --epochs 15

    # Config mode (for distillation, monophonic, or complex setups):
    python -m src.train --config configs/distillation.yaml
    python -m src.train --config configs/monophonic_model.yaml --resume checkpoint.pt

    # Mixed: config as base, CLI overrides specific values:
    python -m src.train --config configs/luca_model.yaml --lr 1e-4 --epochs 20
"""

import argparse
import yaml
import os
import copy
from pathlib import Path

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

# Data imports
from src.data.unified_dataset import UnifiedDataset, create_collate_fn, load_pdmx_synth

# Model imports
from src.networks.luca_model import MusicTrOCR
from src.networks.monophonic_nn import MonophonicModel
from src.networks.distillation import DistillationWrapper
from transformers import AutoModel

# Utils
import src.utils.utils as utils
from src.utils.debug_utils import should_print_debug, print_debug_info
from src.metrics import compute_metrics, aggregate_metrics

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be disabled")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _encoder_short_name(encoder: str) -> str:
    """Derive a short name from a HuggingFace model ID for checkpoint dirs and wandb.

    Examples:
        "facebook/deit-small-patch16-224" -> "deit_small"
        "microsoft/swin-tiny-patch4-window7-224" -> "swin_tiny"
        "google/efficientnet-b0" -> "efficientnet_b0"
    """
    name = encoder.split("/")[-1]           # drop org prefix
    name = name.split("-patch")[0]           # drop patch/window suffixes
    name = name.replace("-", "_")
    return name


def build_config_from_args(args: argparse.Namespace) -> dict:
    """Build a full config dict from CLI arguments.

    Produces the same nested structure that ``load_config()`` returns so all
    downstream code (``create_model``, ``setup_data_loaders``, etc.) works
    unchanged.
    """
    short = _encoder_short_name(args.encoder)
    checkpoint_dir = args.checkpoint_dir or f"networks/checkpoints/{short}"

    config = {
        "model": {
            "type": "MusicTrOCR",
            "encoder_mode": args.encoder_mode,
            "params": {
                "vision_model_name": args.encoder,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_decoder_layers": args.n_decoder_layers,
                "d_ff": args.d_ff,
                "max_seq_len": args.max_seq_len,
                "dropout": args.dropout,
            },
        },
        "data": {
            "dataset_id": args.dataset,
            "tokenizer_id": args.tokenizer,
            "img_height": args.img_height,
            "strip_non_pitch": args.strip_non_pitch,
            "num_workers": 1,
            "pin_memory": True,
        },
        "training": {
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum_steps,
            "epochs": args.epochs,
            "optimizer": {
                "type": "AdamW",
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "betas": [0.9, 0.99],
                "eps": 1e-6,
            },
            "scheduler": {
                "type": "LinearWarmup",
                "warmup_ratio": 0.03,
            },
            "early_stop_patience": args.early_stop_patience,
            "grad_clip_norm": 1.0,
            "checkpoint_dir": checkpoint_dir,
        },
        "hardware": {
            "device": "auto",
            "mixed_precision": True,
        },
    }

    # LoRA / QLoRA settings
    if args.encoder_mode in ("lora", "qlora"):
        config["model"]["lora"] = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "target_modules": ["query", "value"],
            "dropout": 0.05,
        }

    # Wandb (only enabled if --wandb-project is given)
    if args.wandb_project:
        run_name = args.wandb_run_name or f"backbone-{short}"
        tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else ["omr", short]
        config["logging"] = {
            "use_wandb": True,
            "verbose": False,
            "wandb": {
                "project": args.wandb_project,
                "run_name": run_name,
                "tags": tags,
                "notes": f"MusicTrOCR with {args.encoder}",
            },
        }
    else:
        config["logging"] = {"use_wandb": False, "verbose": False}

    return config


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply explicitly-set CLI flags on top of a YAML config.

    Only overrides values for flags the user actually passed on the command
    line (detected via ``argparse`` defaults).
    """
    # Model params
    if args.d_model != 512:
        config.setdefault("model", {}).setdefault("params", {})["d_model"] = args.d_model
    if args.n_heads != 8:
        config.setdefault("model", {}).setdefault("params", {})["n_heads"] = args.n_heads
    if args.n_decoder_layers != 6:
        config.setdefault("model", {}).setdefault("params", {})["n_decoder_layers"] = args.n_decoder_layers
    if args.d_ff != 2048:
        config.setdefault("model", {}).setdefault("params", {})["d_ff"] = args.d_ff
    if args.max_seq_len != 2048:
        config.setdefault("model", {}).setdefault("params", {})["max_seq_len"] = args.max_seq_len
    if args.dropout != 0.1:
        config.setdefault("model", {}).setdefault("params", {})["dropout"] = args.dropout
    if args.encoder_mode != "full":
        config.setdefault("model", {})["encoder_mode"] = args.encoder_mode

    # Training params
    if args.batch_size != 8:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.grad_accum_steps != 4:
        config.setdefault("training", {})["gradient_accumulation_steps"] = args.grad_accum_steps
    if args.epochs != 10:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr != 3e-4:
        config.setdefault("training", {}).setdefault("optimizer", {})["learning_rate"] = args.lr
    if args.weight_decay != 0.01:
        config.setdefault("training", {}).setdefault("optimizer", {})["weight_decay"] = args.weight_decay
    if args.early_stop_patience != 6:
        config.setdefault("training", {})["early_stop_patience"] = args.early_stop_patience
    if args.checkpoint_dir:
        config.setdefault("training", {})["checkpoint_dir"] = args.checkpoint_dir

    # Data params
    if args.dataset != "guangyangmusic/PDMX-Synth":
        config.setdefault("data", {})["dataset_id"] = args.dataset
    if args.tokenizer != "guangyangmusic/legato":
        config.setdefault("data", {})["tokenizer_id"] = args.tokenizer
    if args.img_height != 224:
        config.setdefault("data", {})["img_height"] = args.img_height
    if args.strip_non_pitch:
        config.setdefault("data", {})["strip_non_pitch"] = True

    # Wandb overrides
    if args.wandb_project:
        config.setdefault("logging", {})["use_wandb"] = True
        config["logging"].setdefault("wandb", {})["project"] = args.wandb_project
    if args.wandb_run_name:
        config.setdefault("logging", {}).setdefault("wandb", {})["run_name"] = args.wandb_run_name
    if args.wandb_tags:
        tags = [t.strip() for t in args.wandb_tags.split(",")]
        config.setdefault("logging", {}).setdefault("wandb", {})["tags"] = tags

    return config


def create_model(config: dict, vocab_size: int,
                 pad_token_id: int = None, bos_token_id: int = None,
                 eos_token_id: int = None) -> torch.nn.Module:
    """Create model based on configuration.

    Supports model types: MusicTrOCR, MonophonicModel, Distillation.
    For Distillation the config must contain ``model.teacher.checkpoint``
    and ``model.student.params`` (see configs/distillation.yaml).
    """
    model_type = config['model']['type']

    if model_type == 'Distillation':
        if pad_token_id is None or bos_token_id is None or eos_token_id is None:
            raise ValueError(f"Missing special token IDs: "
                           f"pad={pad_token_id}, bos={bos_token_id}, eos={eos_token_id}")

        token_kwargs = dict(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        # --- teacher ---
        teacher_ckpt_path = config['model']['teacher']['checkpoint']
        print(f"Loading teacher checkpoint from {teacher_ckpt_path}")
        teacher_ckpt = torch.load(teacher_ckpt_path, map_location='cpu')
        teacher_params = teacher_ckpt['config']['model']['params']
        teacher = MusicTrOCR(**token_kwargs, **teacher_params)
        teacher.load_state_dict(teacher_ckpt['model_state_dict'])

        # --- student ---
        student_params = config['model']['student']['params']
        student = MusicTrOCR(**token_kwargs, **student_params)

        # --- wrapper ---
        distill_cfg = config['model'].get('distillation', {})
        model = DistillationWrapper(
            teacher=teacher,
            student=student,
            alpha=distill_cfg.get('alpha', 0.3),
            beta=distill_cfg.get('beta', 0.5),
            gamma=distill_cfg.get('gamma', 0.2),
            temperature=distill_cfg.get('temperature', 4.0),
        )
        print(f"Distillation: teacher {sum(p.numel() for p in teacher.parameters()):,} params "
              f"→ student {model.count_parameters():,} trainable params")
        return model

    model_params = config['model']['params']

    if model_type == 'MusicTrOCR':
        if pad_token_id is None or bos_token_id is None or eos_token_id is None:
            raise ValueError(f"Missing special token IDs: "
                           f"pad={pad_token_id}, bos={bos_token_id}, eos={eos_token_id}")

        print(f"Special tokens: PAD={pad_token_id}, BOS={bos_token_id}, EOS={eos_token_id}")

        model = MusicTrOCR(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **model_params
        )
    elif model_type == 'MonophonicModel':
        model = MonophonicModel(
            hparams=config,
            output_size=vocab_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def setup_data_loaders(config: dict, stage: int = None) -> tuple:
    """Setup train, validation, and test data loaders.

    Args:
        config: Configuration dictionary.
        stage: Kept for interface compatibility (ignored for single-dataset configs).
    """
    data_config = config['data']

    dataset_id = data_config['dataset_id']
    tokenizer_id = data_config['tokenizer_id']
    img_height = data_config.get('img_height', 128)
    max_seq_len = config.get('model', {}).get('params', {}).get('max_seq_len', None)
    # For distillation configs, params live under model.student.params
    if max_seq_len is None:
        max_seq_len = config.get('model', {}).get('student', {}).get('params', {}).get('max_seq_len', None)

    strip_non_pitch = data_config.get('strip_non_pitch', False)

    print(f"Loading dataset '{dataset_id}' with tokenizer '{tokenizer_id}'")
    if max_seq_len:
        print(f"Truncating sequences to max_seq_len={max_seq_len}")
    if strip_non_pitch:
        print("Filtering transcriptions to pitch/rhythm/key only (strip_non_pitch=True)")

    train_dataset, val_dataset, test_dataset = load_pdmx_synth(
        dataset_id=dataset_id,
        tokenizer_id=tokenizer_id,
        img_height=img_height,
        max_seq_len=max_seq_len,
        strip_non_pitch=strip_non_pitch,
    )

    print(f"Dataset splits:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    vocab_size = len(train_dataset.vocabulary_to_index)
    pad_token_id = train_dataset.pad_token_id

    # Create data loaders
    batch_size = config['training']['batch_size']
    collate_fn = create_collate_fn(pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=data_config.get('pin_memory', False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=data_config.get('pin_memory', False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=data_config.get('pin_memory', False)
    )

    return train_loader, val_loader, test_loader, vocab_size


def setup_optimizer_and_scheduler(model: torch.nn.Module, config: dict,
                                  steps_per_epoch: int = None):
    """Setup optimizer and learning rate scheduler"""
    optimizer_config = config['training']['optimizer']
    scheduler_config = config['training']['scheduler']

    # Only optimise trainable parameters (important for distillation / frozen / LoRA runs)
    params = model.trainable_parameters() if hasattr(model, 'trainable_parameters') else [p for p in model.parameters() if p.requires_grad]

    # Create optimizer
    if optimizer_config['type'] == 'Adam':
        optimizer = torch.optim.Adam(
            params,
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config.get('weight_decay', 0.0),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    elif optimizer_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            params,
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-6)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")

    # Create scheduler
    if scheduler_config['type'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            threshold=scheduler_config.get('threshold', 1e-4),
            threshold_mode=scheduler_config.get('threshold_mode', 'rel'),
            cooldown=scheduler_config.get('cooldown', 0),
            eps=scheduler_config.get('eps', 1e-8)
        )
    elif scheduler_config['type'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_config['type'] == 'LinearWarmup':
        accum_steps = config['training'].get('gradient_accumulation_steps', 1)
        optimizer_steps_per_epoch = steps_per_epoch // accum_steps
        total_steps = optimizer_steps_per_epoch * config['training']['epochs']
        warmup_ratio = scheduler_config.get('warmup_ratio', 0.03)
        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Linear decay after warmup
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"Linear warmup: {warmup_steps} warmup steps / {total_steps} total steps")
    else:
        scheduler = None

    return optimizer, scheduler


def setup_wandb(config: dict, model: torch.nn.Module):
    """Setup Weights & Biases logging"""
    if not WANDB_AVAILABLE or not config.get('logging', {}).get('use_wandb', False):
        return None

    wandb_config = config.get('logging', {}).get('wandb', {})

    wandb.init(
        project=wandb_config.get('project', 'music-recognition'),
        name=wandb_config.get('run_name', None),
        config=config,
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes', ''),
    )

    wandb.watch(model, log='all', log_freq=100)

    return wandb


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler,
                   epoch: int,
                   loss: float,
                   config: dict,
                   checkpoint_dir: str,
                   is_best: bool = False,
                   best_loss: float = float('inf'),
                   patience_counter: int = 0,
                   scaler=None):
    """Save model checkpoint with full training state for crash recovery."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
        'best_loss': best_loss,
        'patience_counter': patience_counter,
        'config': config,
        'model_type': config['model']['type']
    }

    checkpoint_path = Path(checkpoint_dir) / 'latest_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = Path(checkpoint_dir) / 'best_checkpoint.pt'
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint at epoch {epoch} (val_loss={loss:.6f})")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler,
                    scaler=None):
    """Load model checkpoint with full training state."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('inf')))
    patience_counter = checkpoint.get('patience_counter', 0)

    print(f"Resumed from epoch {checkpoint['epoch']}, best_loss: {best_loss:.6f}, patience: {patience_counter}")
    return start_epoch, best_loss, patience_counter


def train_epoch(model: torch.nn.Module,
               train_loader: DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int,
               config: dict,
               wandb_run=None,
               scaler=None,
               scheduler=None) -> float:
    """Train for one epoch (supports gradient accumulation and AMP)."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    accum_steps = config['training'].get('gradient_accumulation_steps', 1)
    grad_clip = config['training'].get('grad_clip_norm', None)
    use_amp = scaler is not None

    train_loop = utils.create_tqdm_bar(train_loader, desc=f"Training Epoch {epoch}")

    optimizer.zero_grad()

    for batch_idx, batch in train_loop:
        if hasattr(model, 'training_step'):
            loss = model.training_step(batch, device, config, epoch, batch_idx,
                                       scaler=scaler, accumulation_steps=accum_steps)
        else:
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            (loss / accum_steps).backward()

        # Optimizer step every accum_steps batches (or on last batch)
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
            if use_amp:
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad()

            # Step per-batch schedulers after each optimizer step
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        train_loop.set_postfix({
            'loss': f'{avg_loss:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

        if wandb_run and batch_idx % 50 == 0:
            wandb_run.log({
                'train/batch_loss': loss.item(),
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model: torch.nn.Module,
                  val_loader: DataLoader,
                  device: torch.device,
                  epoch: int,
                  config: dict,
                  wandb_run=None,
                  use_amp=False) -> float:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    if num_batches == 0:
        print(f"Validation Epoch {epoch}: No validation data available (overfitting mode)")
        return 0.0

    val_loop = utils.create_tqdm_bar(val_loader, desc=f"Validation Epoch {epoch}")

    with torch.no_grad():
        for batch_idx, batch in val_loop:
            if hasattr(model, 'validation_step'):
                loss = model.validation_step(batch, device, config, epoch, batch_idx,
                                             use_amp=use_amp)
            else:
                images, targets = batch
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, targets)

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            val_loop.set_postfix({'val_loss': f'{avg_loss:.6f}'})

    avg_loss = total_loss / num_batches

    if wandb_run and num_batches > 0:
        wandb_run.log({
            'val/loss': avg_loss,
            'epoch': epoch
        })

    return avg_loss


def evaluate_metrics(model: torch.nn.Module,
                     val_loader: DataLoader,
                     device: torch.device,
                     config: dict,
                     max_batches: int | None = None) -> dict:
    """
    Run greedy decoding on *val_loader* and compute SER / CER / sequence accuracy.

    This is more expensive than ``validate_epoch`` (which only computes loss
    via teacher forcing) because it calls ``model.generate()`` autoregressively
    for every batch.

    Args:
        model:       Model with a ``generate(images, max_length=…)`` method.
        val_loader:  Validation data loader.
        device:      CUDA / CPU device.
        config:      Training config (used for max_seq_len).
        max_batches: If set, evaluate only this many batches (useful for
                     periodic mid-training checks).

    Returns:
        Aggregated metric dict with keys ``ser``, ``cer``, ``sequence_acc``,
        ``num_samples``.
    """
    generate_fn = getattr(model, "generate", None)
    if generate_fn is None:
        return {"ser": -1, "cer": -1, "sequence_acc": -1, "num_samples": 0}

    dataset = val_loader.dataset
    pad_id = dataset.pad_token_id
    bos_id = dataset.bos_token_id
    eos_id = dataset.eos_token_id
    index_to_vocab = getattr(dataset, "index_to_vocabulary", None)

    max_len = config.get("model", {}).get("params", {}).get("max_seq_len", 512)

    model.eval()
    batch_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, targets = batch
            images = images.to(device)

            predictions = generate_fn(images, max_length=max_len)

            m = compute_metrics(
                predictions.cpu(), targets, pad_id, bos_id, eos_id,
                index_to_vocab=index_to_vocab,
            )
            batch_metrics.append(m)

    if not batch_metrics:
        return {"ser": -1, "cer": -1, "sequence_acc": -1, "num_samples": 0}

    return aggregate_metrics(batch_metrics)


def train_stage(config: dict, stage: int, resume_path: str = None, stage1_checkpoint: str = None):
    """Train for a specific stage"""
    print(f"=== Starting Training Stage {stage} ===")
    print(f"Config: {config}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    stage_checkpoint_dir = checkpoint_dir / f"stage_{stage}"
    stage_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_loader, val_loader, test_loader, vocab_size = setup_data_loaders(config, stage=stage)
    except Exception as e:
        print(f"Error setting up data loaders for stage {stage}: {e}")
        print("Please check your data configuration and ensure:")
        print("  1. Dataset paths exist and contain valid data")
        print("  2. Vocabulary file exists at specified path")
        raise

    train_dataset = train_loader.dataset
    model = create_model(config, vocab_size,
                         pad_token_id=train_dataset.pad_token_id,
                         bos_token_id=train_dataset.bos_token_id,
                         eos_token_id=train_dataset.eos_token_id)

    model.index_to_vocabulary = train_dataset.index_to_vocabulary
    model.vocabulary_to_index = train_dataset.vocabulary_to_index

    # --- Encoder mode: frozen / LoRA / QLoRA ---
    encoder_mode = config['model'].get('encoder_mode', 'full')

    if encoder_mode == 'qlora':
        from peft import get_peft_model, LoraConfig
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        # Reload backbone in 4-bit (bitsandbytes handles device placement)
        model.vision_encoder.backbone = AutoModel.from_pretrained(
            config['model']['params']['vision_model_name'],
            quantization_config=quantization_config,
        )
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

        lora_cfg = config['model'].get('lora', {})
        lora_config = LoraConfig(
            r=lora_cfg.get('rank', 8),
            lora_alpha=lora_cfg.get('alpha', 16),
            target_modules=lora_cfg.get('target_modules', ['query', 'value']),
            lora_dropout=lora_cfg.get('dropout', 0.05),
        )
        model.vision_encoder.backbone = get_peft_model(
            model.vision_encoder.backbone, lora_config
        )
        print(f"QLoRA applied (rank={lora_cfg.get('rank', 8)}, 4-bit quantised encoder)")

    elif encoder_mode == 'lora':
        from peft import get_peft_model, LoraConfig

        for param in model.vision_encoder.parameters():
            param.requires_grad = False

        lora_cfg = config['model'].get('lora', {})
        lora_config = LoraConfig(
            r=lora_cfg.get('rank', 8),
            lora_alpha=lora_cfg.get('alpha', 16),
            target_modules=lora_cfg.get('target_modules', ['query', 'value']),
            lora_dropout=lora_cfg.get('dropout', 0.05),
        )
        model.vision_encoder.backbone = get_peft_model(
            model.vision_encoder.backbone, lora_config
        )
        print(f"LoRA applied (rank={lora_cfg.get('rank', 8)})")

    elif encoder_mode == 'frozen':
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen (no adapters)")

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = model.count_parameters() if hasattr(model, 'count_parameters') else sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model']['type']} | encoder_mode: {encoder_mode}")
    print(f"Total parameters: {total_params:,} | Trainable: {trainable:,} ({100*trainable/total_params:.1f}%)")

    optimizer, scheduler = setup_optimizer_and_scheduler(
        model, config, steps_per_epoch=len(train_loader))

    # Mixed precision
    use_amp = config.get('hardware', {}).get('mixed_precision', False)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled")

    stage_config = copy.deepcopy(config)
    if 'logging' in stage_config and 'wandb' in stage_config['logging']:
        original_name = stage_config['logging']['wandb'].get('run_name', 'omr-training')
        stage_config['logging']['wandb']['run_name'] = f"{original_name}-stage{stage}"

    wandb_run = setup_wandb(stage_config, model)

    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0

    # Auto-resume: check for latest checkpoint if no explicit resume path
    if not resume_path:
        auto_resume_path = stage_checkpoint_dir / 'latest_checkpoint.pt'
        if auto_resume_path.exists():
            print(f"Auto-resuming from {auto_resume_path}")
            resume_path = str(auto_resume_path)

    if stage == 2 and stage1_checkpoint and os.path.exists(stage1_checkpoint):
        print(f"Loading stage 1 checkpoint for stage 2: {stage1_checkpoint}")
        start_epoch, best_loss, patience_counter = load_checkpoint(
            stage1_checkpoint, model, optimizer, scheduler, scaler)
        start_epoch = 0
        best_loss = float('inf')
        patience_counter = 0
    elif resume_path and os.path.exists(resume_path):
        start_epoch, best_loss, patience_counter = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler)

    epochs = config['training']['epochs']
    early_stop_patience = config['training'].get('early_stop_patience', 10)

    for epoch in range(start_epoch, epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1, config,
                                 wandb_run, scaler=scaler, scheduler=scheduler)
        val_loss = validate_epoch(model, val_loader, device, epoch + 1, config,
                                  wandb_run, use_amp=use_amp)

        if val_loss == 0.0 and len(val_loader) == 0:
            print(f"Train Loss: {train_loss:.6f}, Val Loss: N/A (overfitting mode)")
        else:
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # --- SER / CER / sequence accuracy (autoregressive) ---
        eval_every = config.get('evaluation', {}).get('metrics_every_n_epochs', 0)
        metrics_max_batches = config.get('evaluation', {}).get('metrics_max_batches', None)
        is_last_epoch = (epoch + 1 == epochs) or (
            len(val_loader) > 0 and patience_counter + 1 >= early_stop_patience
        )
        should_eval_metrics = (
            len(val_loader) > 0
            and hasattr(model, 'generate')
            and (
                (eval_every > 0 and (epoch + 1) % eval_every == 0)
                or is_last_epoch
            )
        )

        if should_eval_metrics:
            print("  Computing SER / CER / sequence accuracy …")
            metrics = evaluate_metrics(
                model, val_loader, device, config,
                max_batches=metrics_max_batches,
            )
            print(f"  SER: {metrics['ser']*100:.2f}%  "
                  f"CER: {metrics['cer']*100:.2f}%  "
                  f"SeqAcc: {metrics['sequence_acc']*100:.1f}%  "
                  f"(n={metrics['num_samples']})")
            if wandb_run:
                wandb_run.log({
                    'val/ser': metrics['ser'],
                    'val/cer': metrics['cer'],
                    'val/sequence_acc': metrics['sequence_acc'],
                    'epoch': epoch + 1,
                })

        # Only step per-epoch schedulers here (per-batch ones are stepped in train_epoch)
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric_for_scheduler = val_loss if len(val_loader) > 0 else train_loss
            scheduler.step(metric_for_scheduler)

        if len(val_loader) > 0:
            is_best = val_loss < best_loss
            metric_for_comparison = val_loss
            metric_name = "validation"
        else:
            is_best = train_loss < best_loss
            metric_for_comparison = train_loss
            metric_name = "training"

        if is_best:
            best_loss = metric_for_comparison
            patience_counter = 0
            print(f"New best {metric_name} loss: {best_loss:.6f}")
        else:
            patience_counter += 1

        checkpoint_loss = val_loss if len(val_loader) > 0 else train_loss
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_loss, config,
                        stage_checkpoint_dir, is_best, best_loss=best_loss,
                        patience_counter=patience_counter, scaler=scaler)

        if len(val_loader) > 0 and patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break

        if wandb_run:
            wandb_run.log({
                'train/epoch_loss': train_loss,
                'val/epoch_loss': val_loss,
                'epoch': epoch + 1,
                'best_val_loss': best_loss
            })

    print(f"\n=== Training Stage {stage} Complete ===")
    print(f"Best validation loss: {best_loss:.6f}")

    if wandb_run:
        wandb_run.finish()

    best_checkpoint_path = stage_checkpoint_dir / 'best_checkpoint.pt'
    return str(best_checkpoint_path) if best_checkpoint_path.exists() else None


def train(config: dict, resume_path: str = None):
    """Main training function."""
    print("=== Starting Training ===")
    train_stage(config, stage=1, resume_path=resume_path)
    print("=== Training Complete ===")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train OMR models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with a specific encoder (no config file needed):
  python -m src.train --encoder facebook/deit-small-patch16-224

  # LoRA fine-tuning:
  python -m src.train --encoder facebook/deit-small-patch16-224 --encoder-mode lora --lora-rank 8

  # Use a YAML config (for distillation, monophonic, etc.):
  python -m src.train --config configs/distillation.yaml

  # Config + CLI overrides:
  python -m src.train --config configs/luca_model.yaml --lr 1e-4 --epochs 20
""",
    )

    # --- Source of config (one of --config or --encoder required) ---
    source = parser.add_argument_group('config source (one required)')
    source.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    source.add_argument('--encoder', type=str, default=None,
                        help='Pretrained encoder model name (e.g. facebook/deit-small-patch16-224)')

    # --- Model options ---
    model_grp = parser.add_argument_group('model options')
    model_grp.add_argument('--encoder-mode', type=str, default='full',
                           choices=['full', 'frozen', 'lora', 'qlora'],
                           help='Encoder fine-tuning mode (default: full)')
    model_grp.add_argument('--lora-rank', type=int, default=8, help='LoRA rank (default: 8)')
    model_grp.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha (default: 16)')
    model_grp.add_argument('--d-model', type=int, default=512, help='Decoder hidden dim (default: 512)')
    model_grp.add_argument('--n-heads', type=int, default=8, help='Attention heads (default: 8)')
    model_grp.add_argument('--n-decoder-layers', type=int, default=6, help='Decoder layers (default: 6)')
    model_grp.add_argument('--d-ff', type=int, default=2048, help='Feed-forward dim (default: 2048)')
    model_grp.add_argument('--max-seq-len', type=int, default=2048, help='Max sequence length (default: 2048)')
    model_grp.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')

    # --- Training options ---
    train_grp = parser.add_argument_group('training options')
    train_grp.add_argument('--batch-size', type=int, default=8, help='Batch size (default: 8)')
    train_grp.add_argument('--grad-accum-steps', type=int, default=4,
                           help='Gradient accumulation steps (default: 4)')
    train_grp.add_argument('--epochs', type=int, default=10, help='Training epochs (default: 10)')
    train_grp.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    train_grp.add_argument('--weight-decay', type=float, default=0.01,
                           help='Weight decay (default: 0.01)')
    train_grp.add_argument('--early-stop-patience', type=int, default=6,
                           help='Early stopping patience (default: 6)')
    train_grp.add_argument('--checkpoint-dir', type=str, default=None,
                           help='Checkpoint directory (default: auto from encoder name)')

    # --- Data options ---
    data_grp = parser.add_argument_group('data options')
    data_grp.add_argument('--dataset', type=str, default='guangyangmusic/PDMX-Synth',
                          help='Dataset ID (default: guangyangmusic/PDMX-Synth)')
    data_grp.add_argument('--tokenizer', type=str, default='guangyangmusic/legato',
                          help='Tokenizer ID (default: guangyangmusic/legato)')
    data_grp.add_argument('--img-height', type=int, default=224, help='Image height (default: 224)')
    data_grp.add_argument('--strip-non-pitch', action='store_true',
                          help='Filter transcriptions to pitch/rhythm/key only')

    # --- Wandb options ---
    wandb_grp = parser.add_argument_group('wandb options')
    wandb_grp.add_argument('--wandb-project', type=str, default=None,
                           help='Wandb project name (enables wandb logging)')
    wandb_grp.add_argument('--wandb-run-name', type=str, default=None, help='Wandb run name')
    wandb_grp.add_argument('--wandb-tags', type=str, default=None,
                           help='Comma-separated wandb tags')

    # --- Other options ---
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID to use')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=None,
                        help='Run only a specific training stage (1 or 2)')
    parser.add_argument('--stage1-checkpoint', type=str, default=None,
                        help='Path to stage 1 checkpoint for stage 2 training')

    args = parser.parse_args()

    # Validate: need either --config or --encoder
    if args.config is None and args.encoder is None:
        parser.error('one of --config or --encoder is required')

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Build config
    if args.config:
        config = load_config(args.config)
        config = apply_cli_overrides(config, args)
    else:
        config = build_config_from_args(args)

    if args.stage:
        print(f"Running only stage {args.stage}")
        if args.stage == 2 and args.stage1_checkpoint:
            train_stage(config, stage=args.stage, stage1_checkpoint=args.stage1_checkpoint)
        else:
            train_stage(config, stage=args.stage, resume_path=args.resume)
    else:
        train(config, args.resume)


if __name__ == '__main__':
    main()
