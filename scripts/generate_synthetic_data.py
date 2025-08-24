#!/usr/bin/env python3
"""
Synthetic sheet music data generation script for OMR training.
Adapted from SMT repository's VerovioGenerator.

This script generates synthetic sheet music images using the Verovio library
and stores them in the data/synthetic/ directory.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path to import from data module
sys.path.append(str(Path(__file__).parent.parent))

from data.utils.synthetic_generator import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic sheet music data')
    parser.add_argument('--output_dir', type=str, default='data/datasets/synthetic',
                        help='Output directory for synthetic data')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--format', type=str, choices=['bekern', 'kern', 'ekern'], 
                        default='bekern', help='Output format for ground truth')
    parser.add_argument('--system_level', action='store_true',
                        help='Generate single-system images instead of full pages')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible generation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    # Note: You'll need to provide bekern data sources from your unified dataset
    generator = SyntheticDataGenerator(
        bekern_data_path='data/datasets/primus',  # Adapt to your bekern data location
        output_format=args.format,
        seed=args.seed
    )
    
    print(f"Generating {args.num_samples} synthetic samples...")
    print(f"Output directory: {output_path}")
    print(f"Format: {args.format}")
    print(f"System level: {args.system_level}")
    
    # Generate samples
    generator.generate_dataset(
        num_samples=args.num_samples,
        output_dir=output_path,
        system_level=args.system_level
    )
    
    print("Synthetic data generation completed!")


if __name__ == "__main__":
    main()