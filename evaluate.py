#!/usr/bin/env python3
"""
Evaluation script for optical music recognition models.

Usage:
    python evaluate.py --checkpoint networks/checkpoints/luca_model/best_checkpoint.pt --dataset data/primus/package_aa
    python evaluate.py --checkpoint networks/checkpoints/luca_model/best_checkpoint.pt --dataset data/primus/package_aa --max_samples 10
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Data imports
from data.primus_dataset import PrimusDataset, collate_fn

# Model imports
from networks.luca_model import MusicTrOCR
from networks.monophonic_nn import MonophonicModel

# Utils
import utils.utils as utils


def load_checkpoint_and_config(checkpoint_path: str) -> tuple:
    """Load checkpoint and extract configuration"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract information from checkpoint
    config = checkpoint.get('config', {})
    model_type = checkpoint.get('model_type', config.get('model', {}).get('type', 'unknown'))
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    
    print(f"Checkpoint info:")
    print(f"  Model type: {model_type}")
    print(f"  Epoch: {epoch}")
    print(f"  Loss: {loss}")
    print(f"  Config keys: {list(config.keys())}")
    
    return checkpoint, config


def create_model_from_checkpoint(checkpoint: dict, config: dict, vocab_size: int) -> torch.nn.Module:
    """Create model from checkpoint configuration"""
    model_type = checkpoint.get('model_type', config.get('model', {}).get('type', 'unknown'))
    
    if model_type == 'MusicTrOCR':
        model_params = config['model']['params']
        model = MusicTrOCR(
            vocab_size=vocab_size,
            **model_params
        )
    elif model_type == 'MonophonicModel':
        # For backwards compatibility
        model = MonophonicModel(
            hparams=config,
            output_size=vocab_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def tokens_to_strings(tokens: torch.Tensor, index_to_vocab: dict, remove_special: bool = True) -> List[str]:
    """Convert token tensors to readable strings"""
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    
    batch_strings = []
    for sequence in tokens:
        # Convert to list and filter out padding tokens (0)
        token_list = sequence.tolist()
        
        # Remove padding tokens
        token_list = [t for t in token_list if t != 0]
        
        # Convert to strings
        string_tokens = []
        for token in token_list:
            if token in index_to_vocab:
                token_str = index_to_vocab[token]
                string_tokens.append(token_str)
            else:
                string_tokens.append(f"<UNK_{token}>")
        
        batch_strings.append(' '.join(string_tokens))
    
    return batch_strings


def decode_predictions(model_predictions: torch.Tensor, model, dataset) -> List[str]:
    """Decode model predictions to readable strings"""
    # Handle special tokens properly for MusicTrOCR model
    if hasattr(model, 'decode_model_tokens_to_dataset'):
        # For MusicTrOCR, we need to:
        # 1. Remove START tokens (token ID 1) from the beginning
        # 2. Remove END tokens (token ID 2) and everything after
        # 3. Convert remaining model tokens back to dataset tokens
        
        batch_size = model_predictions.shape[0]
        decoded_sequences = []
        
        for i in range(batch_size):
            sequence = model_predictions[i]
            
            # Remove START token if it's at the beginning
            if len(sequence) > 0 and sequence[0] == model.START_TOKEN_ID:
                sequence = sequence[1:]
            
            # Find END token and truncate there
            end_positions = (sequence == model.END_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                sequence = sequence[:end_positions[0]]  # Keep everything before first END token
            
            # Convert model tokens to dataset tokens
            dataset_tokens = model.decode_model_tokens_to_dataset(sequence.unsqueeze(0))
            decoded_sequences.append(dataset_tokens.squeeze(0))
        
        # Stack back to tensor if we have multiple sequences
        if len(decoded_sequences) > 0:
            # Pad sequences to same length for stacking
            max_len = max(len(seq) for seq in decoded_sequences)
            padded_sequences = []
            for seq in decoded_sequences:
                if len(seq) < max_len:
                    padding = torch.zeros(max_len - len(seq), dtype=seq.dtype, device=seq.device)
                    seq = torch.cat([seq, padding])
                padded_sequences.append(seq)
            dataset_tokens = torch.stack(padded_sequences)
        else:
            dataset_tokens = torch.empty(0, 0, dtype=torch.long)
    else:
        # For other models, use as-is
        dataset_tokens = model_predictions
    
    # Convert to strings using dataset vocabulary
    return tokens_to_strings(dataset_tokens, dataset.index_to_vocabulary, remove_special=True)


def run_inference(model: torch.nn.Module, 
                 data_loader: DataLoader, 
                 device: torch.device,
                 dataset: PrimusDataset,
                 max_samples: Optional[int] = None) -> None:
    """Run inference on dataset and print structured results"""
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Total samples: {len(dataset)}")
    print(f"Max samples to process: {max_samples if max_samples else 'all'}")
    print(f"{'='*80}")
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_samples and sample_count >= max_samples:
                break
                
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            
            batch_size = images.shape[0]
            
            # Generate predictions
            if hasattr(model, 'generate'):
                # For models with generate method (like MusicTrOCR)
                predictions = model.generate(images, max_length=512, do_sample=False)
            else:
                # For models without generate method - use forward pass
                outputs = model(images)
                if isinstance(outputs, dict) and 'logits' in outputs:
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                else:
                    predictions = torch.argmax(outputs, dim=-1)
            
            # Decode predictions and targets to strings
            pred_strings = decode_predictions(predictions, model, dataset)
            target_strings = tokens_to_strings(targets, dataset.index_to_vocabulary, remove_special=True)
            
            # Print results for each sample in batch
            for i in range(batch_size):
                if max_samples and sample_count >= max_samples:
                    break
                    
                sample_count += 1
                
                print(f"\nSample {sample_count}:")
                print(f"  Image shape: {images[i].shape}")
                
                # Calculate some basic metrics
                gt_tokens = target_strings[i].split()
                pred_tokens = pred_strings[i].split()
                
                # Print token-by-token comparison table
                print(f"\n  Token-by-Token Comparison:")
                print(f"  {'Position':<8} {'Ground Truth':<20} {'Prediction':<20} {'Match':<6}")
                print(f"  {'-'*8} {'-'*20} {'-'*20} {'-'*6}")
                
                max_tokens = max(len(gt_tokens), len(pred_tokens))
                matches = 0
                
                for j in range(max_tokens):
                    gt_token = gt_tokens[j] if j < len(gt_tokens) else "<EMPTY>"
                    pred_token = pred_tokens[j] if j < len(pred_tokens) else "<EMPTY>"
                    
                    match_symbol = "✓" if gt_token == pred_token and gt_token != "<EMPTY>" else "✗"
                    if gt_token == pred_token and gt_token != "<EMPTY>":
                        matches += 1
                    
                    # Truncate long tokens for display
                    gt_display = gt_token[:18] + ".." if len(gt_token) > 20 else gt_token
                    pred_display = pred_token[:18] + ".." if len(pred_token) > 20 else pred_token
                    
                    print(f"  {j+1:<8} {gt_display:<20} {pred_display:<20} {match_symbol:<6}")
                
                # Summary metrics
                print(f"\n  Summary:")
                if len(gt_tokens) > 0:
                    token_accuracy = matches / len(gt_tokens) * 100
                    print(f"  Token Accuracy: {token_accuracy:.1f}% ({matches}/{len(gt_tokens)})")
                    
                    # Exact sequence match
                    exact_match = target_strings[i] == pred_strings[i]
                    print(f"  Exact Match: {'✓' if exact_match else '✗'}")
                else:
                    print(f"  Token Accuracy: N/A (empty ground truth)")
                    print(f"  Exact Match: N/A")
                
                print(f"  Length GT/Pred: {len(gt_tokens)}/{len(pred_tokens)}")
                print(f"  {'-'*60}")
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"Processed {sample_count} samples")
    print(f"{'='*80}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Evaluate OMR models')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--vocabulary', type=str, default='data/semantic_labels.txt',
                       help='Path to vocabulary file')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference (default: 1)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Validate paths
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
        
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory not found: {args.dataset}")
        sys.exit(1)
        
    if not os.path.exists(args.vocabulary):
        print(f"Error: Vocabulary file not found: {args.vocabulary}")
        sys.exit(1)
    
    try:
        # Load checkpoint and configuration
        checkpoint, config = load_checkpoint_and_config(args.checkpoint)
        
        # Create dataset
        print(f"\nLoading dataset from: {args.dataset}")
        dataset = PrimusDataset(
            data_path=args.dataset,
            vocabulary_path=args.vocabulary
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Vocabulary size: {len(dataset.vocabulary_to_index)}")
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Keep order for evaluation
            collate_fn=collate_fn,
            num_workers=0,  # Single process for evaluation
            pin_memory=False
        )
        
        # Create model
        print(f"\nCreating model...")
        model = create_model_from_checkpoint(checkpoint, config, len(dataset.vocabulary_to_index))
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Run inference
        run_inference(model, data_loader, device, dataset, args.max_samples)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
