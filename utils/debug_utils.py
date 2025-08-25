"""
Debug utilities for verbose training output.
"""

import torch
from typing import List, Optional, Dict, Any
from prettytable import PrettyTable


def print_prediction_table(predictions: torch.Tensor, 
                          targets: torch.Tensor,
                          vocabulary: Dict[int, str],
                          sample_idx: int = 0,
                          max_tokens: int = 50,
                          pad_token_id: int = 0) -> None:
    """
    Print a structured table comparing predictions vs ground truth token by token.
    
    Args:
        predictions: Model predictions tensor [batch_size, seq_len] or logits [batch_size, seq_len, vocab_size]
        targets: Ground truth tensor [batch_size, seq_len]
        vocabulary: Dictionary mapping token IDs to strings {id: token}
        sample_idx: Which sample from the batch to display (default: 0)
        max_tokens: Maximum number of tokens to display (default: 50)
        pad_token_id: ID of padding token to skip (default: 0)
    """
    
    # Handle logits vs predictions
    if len(predictions.shape) == 3:
        # Convert logits to predictions
        pred_tokens = torch.argmax(predictions, dim=-1)
    else:
        pred_tokens = predictions
    
    # Extract sample
    if sample_idx >= pred_tokens.shape[0]:
        print(f"Warning: sample_idx {sample_idx} >= batch_size {pred_tokens.shape[0]}, using sample 0")
        sample_idx = 0
    
    pred_sample = pred_tokens[sample_idx].cpu().numpy()
    target_sample = targets[sample_idx].cpu().numpy()
    
    # Create table
    table = PrettyTable()
    table.field_names = ["Position", "Predicted Token", "Ground Truth", "Match"]
    table.align["Predicted Token"] = "l"
    table.align["Ground Truth"] = "l"
    table.align["Match"] = "c"
    
    # Track accuracy
    total_tokens = 0
    correct_tokens = 0
    
    # Fill table row by row
    for i, (pred_id, target_id) in enumerate(zip(pred_sample, target_sample)):
        if i >= max_tokens:
            table.add_row(["...", "...", "...", "..."])
            break
            
        # Skip padding tokens in target
        if target_id == pad_token_id:
            continue
            
        # Get token strings
        pred_token = vocabulary.get(pred_id, f"<UNK:{pred_id}>")
        target_token = vocabulary.get(target_id, f"<UNK:{target_id}>")
        
        # Check if match
        is_match = pred_id == target_id
        match_symbol = "✓" if is_match else "✗"
        
        # Add row
        table.add_row([i, pred_token, target_token, match_symbol])
        
        # Update accuracy
        total_tokens += 1
        if is_match:
            correct_tokens += 1
    
    # Calculate accuracy
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    # Print table
    print(f"\n{'='*80}")
    print(f"PREDICTION vs GROUND TRUTH - Sample {sample_idx}")
    print(f"Token Accuracy: {correct_tokens}/{total_tokens} ({accuracy:.2%})")
    print(f"{'='*80}")
    print(table)
    print(f"{'='*80}\n")


def print_sequence_comparison(predictions: torch.Tensor,
                             targets: torch.Tensor, 
                             vocabulary: Dict[int, str],
                             sample_idx: int = 0,
                             pad_token_id: int = 0) -> None:
    """
    Print full sequence comparison in a compact format.
    
    Args:
        predictions: Model predictions tensor [batch_size, seq_len] or logits [batch_size, seq_len, vocab_size]
        targets: Ground truth tensor [batch_size, seq_len] 
        vocabulary: Dictionary mapping token IDs to strings {id: token}
        sample_idx: Which sample from the batch to display (default: 0)
        pad_token_id: ID of padding token to skip (default: 0)
    """
    
    # Handle logits vs predictions
    if len(predictions.shape) == 3:
        pred_tokens = torch.argmax(predictions, dim=-1)
    else:
        pred_tokens = predictions
    
    # Extract sample
    if sample_idx >= pred_tokens.shape[0]:
        print(f"Warning: sample_idx {sample_idx} >= batch_size {pred_tokens.shape[0]}, using sample 0")
        sample_idx = 0
        
    pred_sample = pred_tokens[sample_idx].cpu().numpy()
    target_sample = targets[sample_idx].cpu().numpy()
    
    # Convert to strings
    pred_sequence = []
    target_sequence = []
    
    for pred_id, target_id in zip(pred_sample, target_sample):
        if target_id == pad_token_id:
            continue
            
        pred_token = vocabulary.get(pred_id, f"<UNK:{pred_id}>")
        target_token = vocabulary.get(target_id, f"<UNK:{target_id}>")
        
        pred_sequence.append(pred_token)
        target_sequence.append(target_token)
    
    # Print sequences
    print(f"\nSEQUENCE COMPARISON - Sample {sample_idx}")
    print("-" * 80)
    print(f"PREDICTED : {' '.join(pred_sequence)}")
    print(f"TARGET    : {' '.join(target_sequence)}")
    print("-" * 80)


def should_print_debug(config: Dict[str, Any], 
                      epoch: int, 
                      batch_idx: int, 
                      print_interval: int = 100) -> bool:
    """
    Determine if debug output should be printed based on config and intervals.
    
    Args:
        config: Training configuration dictionary
        epoch: Current epoch number
        batch_idx: Current batch index
        print_interval: Print every N batches (default: 100)
    
    Returns:
        True if debug output should be printed
    """
    verbose = config.get('logging', {}).get('verbose', False)
    
    if not verbose:
        return False
    
    # Print on first batch of each epoch, and then every print_interval batches
    return batch_idx == 0 or (batch_idx % print_interval == 0)


def get_vocabulary_from_model(model) -> Dict[int, str]:
    """
    Extract vocabulary mapping from model for debug printing.
    
    Args:
        model: The model instance
        
    Returns:
        Dictionary mapping token IDs to token strings
    """
    if hasattr(model, 'index_to_vocabulary'):
        return model.index_to_vocabulary
    elif hasattr(model, 'i2w'):
        return model.i2w
    elif hasattr(model, 'vocab'):
        # Handle different vocab formats
        if hasattr(model.vocab, 'i2w'):
            return model.vocab.i2w
        elif isinstance(model.vocab, dict):
            # If it's w2i, invert it
            if all(isinstance(v, int) for v in model.vocab.values()):
                return {v: k for k, v in model.vocab.items()}
            else:
                return model.vocab
    else:
        # No fallback: crash if vocabulary is not found
        raise AttributeError(
            "Could not find vocabulary in model. Expected one of: "
            "'index_to_vocabulary', 'i2w', 'vocab.i2w', or 'vocab' (dict). "
            "Please ensure the model has a proper vocabulary mapping."
        )


def print_debug_info(predictions: torch.Tensor,
                    targets: torch.Tensor,
                    loss: torch.Tensor,
                    model,
                    epoch: int,
                    batch_idx: int,
                    sample_idx: int = 0,
                    phase: str = "TRAINING") -> None:
    """
    Main function to print comprehensive debug information.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        loss: Current loss value
        model: The model instance (to extract vocabulary)
        epoch: Current epoch
        batch_idx: Current batch index
        sample_idx: Which sample to debug (default: 0)
        phase: Training phase (TRAINING/VALIDATION)
    """
    
    vocabulary = get_vocabulary_from_model(model)
    pad_token_id = getattr(model, 'PAD_TOKEN_ID', 0)
    
    print(f"\n{'#'*100}")
    print(f"{phase} DEBUG INFO - Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
    print(f"{'#'*100}")
    
    # Print detailed token-by-token comparison
    print_prediction_table(predictions, targets, vocabulary, sample_idx, pad_token_id=pad_token_id)
    
    # Print compact sequence comparison
    print_sequence_comparison(predictions, targets, vocabulary, sample_idx, pad_token_id=pad_token_id)
    
    print(f"{'#'*100}\n")