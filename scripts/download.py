#!/usr/bin/env python3
"""
Download script for datasets used in the music recognition project.
Handles both Camera Primus dataset and SMT HuggingFace datasets.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

import datasets


# SMT datasets from HuggingFace (for synthetic generation)
SMT_DATASETS = {
    "grandstaff": {
        "id": "antoniorv6/grandstaff",
        "description": "GrandStaff system-level dataset (original format)",
        "format": "original"
    },
    "grandstaff-ekern": {
        "id": "antoniorv6/grandstaff-ekern", 
        "description": "GrandStaff dataset in ekern format",
        "format": "ekern"
    },
    "grandstaff-bekern": {
        "id": "antoniorv6/grandstaff-bekern",
        "description": "GrandStaff dataset in bekern format", 
        "format": "bekern"
    },
    "mozarteum": {
        "id": "antoniorv6/mozarteum",
        "description": "Mozarteum dataset",
        "format": "original"
    },
    "polish-scores": {
        "id": "antoniorv6/polish-scores",
        "description": "Polish Scores dataset",
        "format": "original"
    },
    "string-quartets": {
        "id": "antoniorv6/string-quartets",
        "description": "String Quartets dataset",
        "format": "original"
    }
}

# Camera Primus dataset info
PRIMUS_DATASET = {
    "url": "https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz",
    "filename": "primusCalvoRizoAppliedSciences2018.tgz",
    "output_dir": "data/datasets/primus"
}


def download_primus_dataset():
    """Download and extract the Camera Primus dataset."""
    primus_dir = Path(PRIMUS_DATASET["output_dir"])
    primus_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = primus_dir / PRIMUS_DATASET["filename"]
    
    print(f"Downloading Camera Primus dataset...")
    print(f"URL: {PRIMUS_DATASET['url']}")
    print(f"Output: {primus_dir}")
    
    # Download with wget
    try:
        subprocess.run([
            "wget", 
            "-O", str(archive_path),
            PRIMUS_DATASET["url"]
        ], check=True, cwd=str(primus_dir))
        
        print(f"Download completed: {archive_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading Primus dataset: {e}")
        return False
    except FileNotFoundError:
        print("Error: wget not found. Please install wget or download manually.")
        return False
    
    # Extract archive
    print("Extracting archive...")
    try:
        subprocess.run([
            "tar", "-xzvf", str(archive_path)
        ], check=True, cwd=str(primus_dir))
        
        print("Extraction completed successfully")
        
        # Optional: Remove archive file to save space
        if archive_path.exists():
            archive_path.unlink()
            print(f"Removed archive file: {archive_path}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting archive: {e}")
        return False


def list_smt_datasets():
    """List all available SMT datasets."""
    print("Available SMT datasets (for synthetic generation):")
    print("=" * 60)
    
    for key, info in SMT_DATASETS.items():
        print(f"ID: {key}")
        print(f"  HuggingFace: {info['id']}")
        print(f"  Description: {info['description']}")
        print(f"  Format: {info['format']}")
        print()


def download_smt_dataset(dataset_key: str, output_dir: Path, splits: List[str] = None):
    """Download a specific SMT dataset from HuggingFace."""
    if dataset_key not in SMT_DATASETS:
        print(f"Error: Unknown SMT dataset '{dataset_key}'")
        print("Available datasets:")
        for key in SMT_DATASETS.keys():
            print(f"  - {key}")
        return False
    
    dataset_info = SMT_DATASETS[dataset_key]
    dataset_id = dataset_info["id"]
    
    if splits is None:
        splits = ["train", "validation", "test"]
    
    print(f"Downloading SMT dataset: {dataset_id}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    dataset_output_dir = output_dir / dataset_key
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download dataset
        for split in splits:
            try:
                print(f"Downloading {split} split...")
                dataset = datasets.load_dataset(dataset_id, split=split, trust_remote_code=False)
                
                # Save to disk
                split_path = dataset_output_dir / split
                dataset.save_to_disk(str(split_path))
                
                print(f"  {split}: {len(dataset)} samples saved to {split_path}")
                
            except Exception as e:
                print(f"  Warning: Could not download {split} split: {e}")
                continue
        
        # Save dataset info
        info_file = dataset_output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            import json
            json.dump({
                "dataset_id": dataset_id,
                "dataset_key": dataset_key,
                "description": dataset_info["description"],
                "format": dataset_info["format"],
                "downloaded_splits": splits
            }, f, indent=2)
        
        print(f"Successfully downloaded {dataset_key}")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset {dataset_id}: {e}")
        return False


def download_multiple_smt_datasets(dataset_keys: List[str], output_dir: Path):
    """Download multiple SMT datasets."""
    success_count = 0
    
    for dataset_key in dataset_keys:
        print(f"\n{'='*60}")
        if download_smt_dataset(dataset_key, output_dir):
            success_count += 1
        print(f"{'='*60}")
    
    print(f"\nSMT download summary: {success_count}/{len(dataset_keys)} datasets downloaded successfully")


def main():
    parser = argparse.ArgumentParser(description='Download datasets for music recognition project')
    
    # Dataset type selection
    parser.add_argument('--primus', action='store_true',
                        help='Download Camera Primus dataset')
    parser.add_argument('--smt', type=str, nargs='+', 
                        help='Download specific SMT datasets (space-separated)')
    parser.add_argument('--smt-all', action='store_true',
                        help='Download all SMT datasets')
    
    # General options
    parser.add_argument('--output_dir', type=str, default='data/datasets',
                        help='Base output directory for datasets')
    parser.add_argument('--list-smt', action='store_true',
                        help='List available SMT datasets')
    parser.add_argument('--splits', type=str, nargs='+', 
                        default=['train', 'validation', 'test'],
                        help='Dataset splits to download for SMT datasets')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.list_smt:
        list_smt_datasets()
        return
    
    # Download Camera Primus dataset
    if args.primus:
        print("Downloading Camera Primus dataset...")
        success = download_primus_dataset()
        if success:
            print("Camera Primus dataset downloaded successfully")
        else:
            print("Failed to download Camera Primus dataset")
            sys.exit(1)
    
    # Download SMT datasets
    if args.smt_all:
        smt_output_dir = output_dir / "smt_datasets"
        dataset_keys = list(SMT_DATASETS.keys())
        print(f"Downloading all {len(dataset_keys)} SMT datasets...")
        download_multiple_smt_datasets(dataset_keys, smt_output_dir)
    elif args.smt:
        smt_output_dir = output_dir / "smt_datasets"
        download_multiple_smt_datasets(args.smt, smt_output_dir)
    
    if not (args.primus or args.smt or args.smt_all):
        print("Please specify what to download:")
        print("  --primus          Download Camera Primus dataset")
        print("  --smt <datasets>  Download specific SMT datasets")
        print("  --smt-all         Download all SMT datasets")
        print("  --list-smt        List available SMT datasets")
        print("\nExamples:")
        print("  python scripts/download.py --primus")
        print("  python scripts/download.py --smt grandstaff-ekern")
        print("  python scripts/download.py --smt-all")
        print("  python scripts/download.py --primus --smt grandstaff-ekern")


if __name__ == "__main__":
    main()