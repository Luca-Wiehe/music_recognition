#!/usr/bin/env python3
"""
Flexible training script for optical music recognition models.

Usage:
    python train.py --config configs/luca_model.yaml
    python train.py --config configs/monophonic_model.yaml --resume checkpoint.pt
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
from data.data_loading.unified_dataset import UnifiedDataset, collate_fn, create_split_datasets
from data.data_loading.primus_dataset import split_data  # Keep split_data utility function

# Model imports
from networks.luca_model import MusicTrOCR
from networks.monophonic_nn import MonophonicModel

# Utils
import utils.utils as utils
from utils.debug_utils import should_print_debug, print_debug_info

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


def create_model(config: dict, vocab_size: int) -> torch.nn.Module:
    """Create model based on configuration"""
    model_type = config['model']['type']
    model_params = config['model']['params']
    
    if model_type == 'MusicTrOCR':
        model = MusicTrOCR(
            vocab_size=vocab_size,
            **model_params
        )
    elif model_type == 'MonophonicModel':
        # For backwards compatibility with existing monophonic model
        model = MonophonicModel(
            hparams=config,
            output_size=vocab_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def setup_data_loaders(config: dict) -> tuple:
    """Setup train, validation, and test data loaders"""
    data_config = config['data']
    
    # Extract data paths from new config format
    data_paths = []
    primus_paths = []
    bekern_paths = []
    
    for dataset_info in data_config['datasets']:
        path = dataset_info['path']
        format_type = dataset_info['format']
        
        data_paths.append(path)
        if format_type == 'primus':
            primus_paths.append(path)
        elif format_type == 'bekern':
            bekern_paths.append(path)
        else:
            raise ValueError(f"Unknown dataset format: {format_type}")
    
    print(f"Loading datasets:")
    print(f"  Primus format: {len(primus_paths)} datasets")
    print(f"  BeKern format: {len(bekern_paths)} datasets")
    
    # Create datasets using automatic train/val/test detection
    train_dataset, val_dataset, test_dataset = create_split_datasets(
        data_paths=data_paths,
        bekern_vocab_path=data_config['bekern_vocabulary_path'],
        mapping_file_path=data_config.get('mapping_file_path'),
        transform=None
    )
    
    print(f"Using automatic train/val/test splits:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    vocab_size = len(train_dataset.vocabulary_to_index)
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
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


def setup_optimizer_and_scheduler(model: torch.nn.Module, config: dict):
    """Setup optimizer and learning rate scheduler"""
    optimizer_config = config['training']['optimizer']
    scheduler_config = config['training']['scheduler']
    
    # Create optimizer
    if optimizer_config['type'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config.get('weight_decay', 0.0),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    elif optimizer_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config.get('weight_decay', 0.01)
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
    else:
        scheduler = None
    
    return optimizer, scheduler


def setup_wandb(config: dict, model: torch.nn.Module):
    """Setup Weights & Biases logging"""
    if not WANDB_AVAILABLE or not config.get('logging', {}).get('use_wandb', False):
        return None
    
    wandb_config = config.get('logging', {}).get('wandb', {})
    
    # Initialize wandb
    wandb.init(
        project=wandb_config.get('project', 'music-recognition'),
        name=wandb_config.get('run_name', None),
        config=config,
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes', ''),
    )
    
    # Log model architecture
    wandb.watch(model, log='all', log_freq=100)
    
    return wandb


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   scheduler,
                   epoch: int,
                   loss: float,
                   config: dict,
                   checkpoint_dir: str,
                   is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config,
        'model_type': config['model']['type']
    }
    
    # Save latest checkpoint
    checkpoint_path = Path(checkpoint_dir) / 'latest_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint if applicable
    if is_best:
        best_path = Path(checkpoint_dir) / 'best_checkpoint.pt'
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint at epoch {epoch}")
    
    # Save periodic checkpoint
    if epoch % 100 == 0:
        periodic_path = Path(checkpoint_dir) / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, periodic_path)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('loss', float('inf'))
    
    print(f"Resumed from epoch {checkpoint['epoch']}, loss: {best_loss:.6f}")
    return start_epoch, best_loss


def train_epoch(model: torch.nn.Module,
               train_loader: DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int,
               config: dict,
               wandb_run=None) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    # Create progress bar
    train_loop = utils.create_tqdm_bar(train_loader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, batch in train_loop:
        # Forward pass and compute loss
        if hasattr(model, 'training_step'):
            # For models with custom training_step (like MusicTrOCR)
            loss = model.training_step(batch, device, optimizer, config, epoch, batch_idx)
        else:
            # For models using standard training loop (like MonophonicModel) 
            # This would need to be implemented based on the specific model
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, targets)  # Simplified
            loss.backward()
            
            # Gradient clipping
            if config['training'].get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip_norm'])
            
            optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        train_loop.set_postfix({
            'loss': f'{avg_loss:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Log to wandb
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
                  wandb_run=None) -> float:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    # Handle case where validation dataset is empty (overfitting scenario)
    if num_batches == 0:
        print(f"Validation Epoch {epoch}: No validation data available (overfitting mode)")
        return 0.0
    
    # Create progress bar
    val_loop = utils.create_tqdm_bar(val_loader, desc=f"Validation Epoch {epoch}")
    
    with torch.no_grad():
        for batch_idx, batch in val_loop:
            # Forward pass and compute loss
            if hasattr(model, 'validation_step'):
                # For models with custom validation_step
                loss = model.validation_step(batch, device, config, epoch, batch_idx)
            else:
                # For standard models
                images, targets = batch
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, targets)  # Simplified
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            val_loop.set_postfix({'val_loss': f'{avg_loss:.6f}'})
    
    avg_loss = total_loss / num_batches
    
    # Log to wandb (only if we have validation data)
    if wandb_run and num_batches > 0:
        wandb_run.log({
            'val/loss': avg_loss,
            'epoch': epoch
        })
    
    return avg_loss


def train(config: dict, resume_path: str = None):
    """Main training function"""
    print("=== Starting Training ===")
    print(f"Config: {config}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data loaders
    try:
        train_loader, val_loader, test_loader, vocab_size = setup_data_loaders(config)
    except Exception as e:
        print(f"Error setting up data loaders: {e}")
        print("Please check your data configuration and ensure:")
        print("  1. Dataset paths exist and contain valid data")
        print("  2. BeKern vocabulary file exists at specified path")
        print("  3. Mapping file exists and contains valid mappings")
        raise
    
    # Create model
    model = create_model(config, vocab_size)
    
    model.index_to_vocabulary = train_loader.dataset.index_to_vocabulary
    model.vocabulary_to_index = train_loader.dataset.vocabulary_to_index
    
    model.to(device)

    print(f"Model: {config['model']['type']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config)
    
    # Setup logging
    wandb_run = setup_wandb(config, model)
    
    # Setup checkpointing and resuming
    start_epoch = 0
    best_loss = float('inf')
    
    if resume_path and os.path.exists(resume_path):
        start_epoch, best_loss = load_checkpoint(resume_path, model, optimizer, scheduler)
    
    # Training loop
    epochs = config['training']['epochs']
    early_stop_patience = config['training'].get('early_stop_patience', 10)
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1, config, wandb_run)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device, epoch + 1, config, wandb_run)
        
        # Print losses (handle case where val_loss is 0.0 due to no validation data)
        if val_loss == 0.0 and len(val_loader) == 0:
            print(f"Train Loss: {train_loss:.6f}, Val Loss: N/A (overfitting mode)")
        else:
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use train_loss for scheduling when no validation data is available
                metric_for_scheduler = val_loss if len(val_loader) > 0 else train_loss
                scheduler.step(metric_for_scheduler)
            else:
                scheduler.step()
        
        # Check for best model (use train_loss when no validation data available)
        if len(val_loader) > 0:
            # Normal validation-based early stopping
            is_best = val_loss < best_loss
            metric_for_comparison = val_loss
            metric_name = "validation"
        else:
            # Overfitting mode - use training loss for comparison
            is_best = train_loss < best_loss
            metric_for_comparison = train_loss
            metric_name = "training"
        
        if is_best:
            best_loss = metric_for_comparison
            patience_counter = 0
            print(f"New best {metric_name} loss: {best_loss:.6f}")
        else:
            patience_counter += 1
        
        # Save checkpoint
        checkpoint_loss = val_loss if len(val_loader) > 0 else train_loss
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_loss, config, checkpoint_dir, is_best)
        
        # Early stopping (disabled in overfitting mode to let the model fully overfit)
        if len(val_loader) > 0 and patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        # Log epoch summary
        if wandb_run:
            wandb_run.log({
                'train/epoch_loss': train_loss,
                'val/epoch_loss': val_loss,
                'epoch': epoch + 1,
                'best_val_loss': best_loss
            })
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation loss: {best_loss:.6f}")
    
    # Cleanup wandb
    if wandb_run:
        wandb_run.finish()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train OMR models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID to use')
    
    args = parser.parse_args()
    
    # Set GPU device if specified
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load config
    config = load_config(args.config)
    
    # Start training
    train(config, args.resume)


if __name__ == '__main__':
    main()