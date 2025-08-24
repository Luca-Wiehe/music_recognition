"""
Data module for music recognition project.

This module contains dataset classes and utilities for optical music recognition:

- PrimusDataset: Original Camera Primus dataset loader
- UnifiedDataset: Unified dataset that converts Camera Primus data to BeKern format
- PrimusToBeKernConverter: Format converter for Camera Primus to BeKern tokens
"""

from .primus_dataset import PrimusDataset, split_data, visualize_sample, collate_fn
from .format_converter import PrimusToBeKernConverter, load_primus_labels_from_file
from .unified_dataset import UnifiedDataset, collate_fn as unified_collate_fn

__all__ = [
    # Original Primus dataset
    'PrimusDataset',
    'split_data',
    'visualize_sample', 
    'collate_fn',
    
    # Format conversion
    'PrimusToBeKernConverter',
    'load_primus_labels_from_file',
    
    # Unified dataset
    'UnifiedDataset', 
    'unified_collate_fn'
]