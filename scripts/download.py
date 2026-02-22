#!/usr/bin/env python3
"""
Download script for datasets used in the music recognition project.
Handles Camera Primus, SMT HuggingFace datasets, and PDMX-Synth.
"""

import os
import sys
import argparse
import subprocess
import random
import shutil
from pathlib import Path
from typing import List

import datasets


# SMT datasets from HuggingFace (for synthetic generation)
SMT_DATASETS = {
    "grandstaff": {
        "id": "PRAIG/grandstaff",
        "description": "GrandStaff system-level dataset (original format)",
        "format": "original"
    },
    "camera-grandstaff": {
        "id": "PRAIG/camera-grandstaff", 
        "description": "Camera GrandStaff dataset",
        "format": "original"
    },
    "fp-grandstaff": {
        "id": "PRAIG/fp-grandstaff",
        "description": "FP GrandStaff dataset", 
        "format": "original"
    },
    "polish-scores": {
        "id": "PRAIG/polish-scores",
        "description": "Polish Scores dataset",
        "format": "original"
    }
}

# PDMX-Synth dataset from HuggingFace (large-scale OMR with ABC notation labels)
PDMX_SYNTH_DATASET = {
    "id": "guangyangmusic/PDMX-Synth",
    "description": "PDMX-Synth: 216K image-ABC pairs rendered from public domain MusicXML scores (CC-BY-4.0)",
    "output_dir": "data/datasets/pdmx-synth",
    "splits": {"train": 214547, "val": 800, "test": 800},
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
        ], check=True)
        
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
            "tar", "-xzvf", str(archive_path), "-C", str(primus_dir)
        ], check=True)
        
        print("Extraction completed successfully")
        
        # Optional: Remove archive file to save space
        if archive_path.exists():
            archive_path.unlink()
            print(f"Removed archive file: {archive_path}")
        
        # Split extracted data into train/val/test
        success = _split_primus_dataset(primus_dir)
        if not success:
            print("Warning: Failed to split Primus dataset into train/val/test")
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting archive: {e}")
        return False


def _split_primus_dataset(primus_dir: Path):
    """Split Primus dataset from package_aa and package_ab into train/val/test splits."""
    print("Splitting Primus dataset into train/val/test...")
    
    # Find all sample directories from both packages
    all_samples = []
    
    for package_name in ["package_aa", "package_ab"]:
        package_dir = primus_dir / package_name
        if not package_dir.exists():
            print(f"Warning: Package {package_name} not found in {primus_dir}")
            continue
        
        # Find all sample directories in this package
        for sample_dir in package_dir.iterdir():
            if sample_dir.is_dir():
                all_samples.append(sample_dir)
    
    if not all_samples:
        print("Error: No sample directories found in package_aa or package_ab")
        return False
    
    print(f"Found {len(all_samples)} total samples to split")
    
    # Shuffle samples for random split
    random.seed(42)  # For reproducible splits
    random.shuffle(all_samples)
    
    # Calculate split sizes (80% train, 10% val, 10% test)
    total_samples = len(all_samples)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    
    print(f"Split sizes: train={train_size}, val={val_size}, test={test_size}")
    
    # Create split directories
    splits = {
        "train": all_samples[:train_size],
        "val": all_samples[train_size:train_size + val_size],
        "test": all_samples[train_size + val_size:]
    }
    
    # Move samples to split directories
    for split_name, split_samples in splits.items():
        split_dir = primus_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"Moving {len(split_samples)} samples to {split_name} split...")
        
        for i, sample_dir in enumerate(split_samples):
            # Create new sample directory name with index
            new_sample_name = f"sample_{i:06d}"
            new_sample_dir = split_dir / new_sample_name
            
            # Move the sample directory
            shutil.move(str(sample_dir), str(new_sample_dir))
    
    # Remove empty package directories
    for package_name in ["package_aa", "package_ab"]:
        package_dir = primus_dir / package_name
        if package_dir.exists():
            try:
                package_dir.rmdir()  # Remove empty directory
                print(f"Removed empty package directory: {package_name}")
            except OSError:
                # Directory not empty, remove it with contents
                shutil.rmtree(package_dir)
                print(f"Removed package directory with remaining contents: {package_name}")
    
    print("Primus dataset split completed successfully")
    return True


def download_pdmx_synth_dataset(output_dir: Path, splits: List[str] = None):
    """Download PDMX-Synth dataset from HuggingFace and convert to Primus-compatible format.

    Dataset: guangyangmusic/PDMX-Synth (~19GB, 216K image-ABC pairs)
    Requires accepting the dataset terms on HuggingFace and being logged in via `huggingface-cli login`.
    """
    dataset_id = PDMX_SYNTH_DATASET["id"]
    pdmx_dir = output_dir / "pdmx-synth"
    pdmx_dir.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ["train", "val", "test"]

    print(f"Downloading PDMX-Synth dataset: {dataset_id}")
    print(f"Output directory: {pdmx_dir}")
    print(f"Note: This dataset is ~19GB. You must accept the terms at")
    print(f"  https://huggingface.co/datasets/{dataset_id}")
    print(f"  and log in via `huggingface-cli login` before downloading.")
    print()

    total_exported = 0

    for split in splits:
        split_dir = pdmx_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Skip if split already has sample directories
        existing = list(split_dir.glob("sample_*"))
        if existing:
            print(f"  {split}: already has {len(existing)} samples, skipping (delete to re-download)")
            total_exported += len(existing)
            continue

        try:
            print(f"  Downloading {split} split...")
            dataset = datasets.load_dataset(dataset_id, split=split)
            print(f"  Converting {split} split ({len(dataset)} samples) to Primus format...")

            for i, sample in enumerate(dataset):
                sample_dir = split_dir / f"sample_{i:06d}"
                sample_dir.mkdir(exist_ok=True)

                # Save image
                image = sample["image"]
                image.save(sample_dir / f"sample_{i:06d}.png")

                # Save ABC transcription as .semantic file
                transcription = sample["transcription"]
                semantic_path = sample_dir / f"sample_{i:06d}.semantic"
                with open(semantic_path, "w", encoding="utf-8") as f:
                    f.write(transcription)

                if (i + 1) % 5000 == 0:
                    print(f"    Converted {i + 1}/{len(dataset)} samples...")

            total_exported += len(dataset)
            print(f"  Completed {split}: {len(dataset)} samples")

        except Exception as e:
            print(f"  Error downloading/converting {split}: {e}")
            continue

    # Save dataset info
    import json
    info_file = pdmx_dir / "dataset_info.json"
    with open(info_file, "w") as f:
        json.dump({
            "dataset_id": dataset_id,
            "description": PDMX_SYNTH_DATASET["description"],
            "label_format": "abc",
            "downloaded_splits": splits,
            "total_exported": total_exported,
        }, f, indent=2)

    if total_exported > 0:
        print(f"\nSuccessfully downloaded {total_exported} PDMX-Synth samples")
        print(f"Data available in: {pdmx_dir}/{{train,val,test}}/sample_*/")
        return True
    else:
        print("Failed to download any PDMX-Synth splits")
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


def export_smt_to_primus_format(dataset_key: str, output_dir: Path, splits: List[str] = None):
    """Export SMT dataset from HuggingFace format to Primus-compatible format and delete arrow files."""
    if dataset_key not in SMT_DATASETS:
        print(f"Error: Unknown SMT dataset '{dataset_key}'")
        return False
    
    if splits is None:
        splits = ["train", "val", "test"]
    
    smt_dataset_dir = output_dir / "smt_datasets" / dataset_key
    
    if not smt_dataset_dir.exists():
        print(f"Error: SMT dataset not found at {smt_dataset_dir}")
        print("Please download the dataset first using --smt or --smt-all")
        return False
    
    print(f"Converting {dataset_key} to Primus-compatible format...")
    
    # Track arrow files to delete later
    arrow_files_to_delete = []
    
    total_exported = 0
    
    for split in splits:
        split_dir = smt_dataset_dir / split
        if not split_dir.exists():
            print(f"  Warning: Split {split} not found, skipping")
            continue
        
        # Track arrow files in this split for deletion
        arrow_file = split_dir / "data-00000-of-00001.arrow"
        if arrow_file.exists():
            arrow_files_to_delete.append(arrow_file)
        
        try:
            # Load the dataset split
            dataset = datasets.load_from_disk(str(split_dir))
            
            print(f"  Converting {split} split ({len(dataset)} samples)...")
            
            for i, sample in enumerate(dataset):
                sample_dir = split_dir / f"sample_{i:06d}"
                sample_dir.mkdir(exist_ok=True)
                
                # Save image
                image = sample['image']
                image_path = sample_dir / f"sample_{i:06d}.png"
                image.save(image_path)
                
                # Save transcription as .semantic file
                transcription = sample['transcription']
                semantic_path = sample_dir / f"sample_{i:06d}.semantic"
                with open(semantic_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                
                if (i + 1) % 1000 == 0:
                    print(f"    Converted {i + 1} samples...")
            
            total_exported += len(dataset)
            print(f"  Completed {split}: {len(dataset)} samples")
            
        except Exception as e:
            print(f"  Error converting {split}: {e}")
            continue
    
    # Delete arrow files and other HuggingFace metadata after successful conversion
    if total_exported > 0:
        print("  Cleaning up arrow files and metadata...")
        files_deleted = 0
        
        for split in splits:
            split_dir = smt_dataset_dir / split
            if not split_dir.exists():
                continue
                
            # Remove arrow files and metadata
            for file_to_delete in ["data-00000-of-00001.arrow", "dataset_info.json", "state.json"]:
                file_path = split_dir / file_to_delete
                if file_path.exists():
                    try:
                        file_path.unlink()
                        files_deleted += 1
                    except Exception as e:
                        print(f"    Warning: Could not delete {file_path}: {e}")
        
        print(f"  Deleted {files_deleted} arrow/metadata files")
    
    print(f"Successfully converted {total_exported} samples for {dataset_key}")
    print(f"Data available in: {smt_dataset_dir}/*/sample_*/")
    return True


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
        splits = ["train", "val", "test"]
    
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
        
        # Automatically convert to Primus format and cleanup arrow files
        print(f"Converting {dataset_key} to Primus format...")
        export_smt_to_primus_format(dataset_key, output_dir.parent, ["train", "val", "test"])
        
        print(f"Successfully downloaded and converted {dataset_key}")
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
    parser.add_argument('--pdmx-synth', action='store_true',
                        help='Download PDMX-Synth dataset (~19GB, 216K image-ABC pairs)')
    parser.add_argument('--smt', type=str, nargs='+',
                        help='Download specific SMT datasets (space-separated)')
    parser.add_argument('--smt-all', action='store_true',
                        help='Download all SMT datasets')
    parser.add_argument('--export-smt', type=str, nargs='+',
                        help='Convert specific SMT datasets to Primus format and delete arrow files (space-separated)')
    
    # General options
    parser.add_argument('--output_dir', type=str, default='data/datasets',
                        help='Base output directory for datasets')
    parser.add_argument('--list-smt', action='store_true',
                        help='List available SMT datasets')
    parser.add_argument('--splits', type=str, nargs='+', 
                        default=['train', 'val', 'test'],
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

    # Download PDMX-Synth dataset
    if args.pdmx_synth:
        success = download_pdmx_synth_dataset(output_dir, splits=args.splits)
        if not success:
            print("Failed to download PDMX-Synth dataset")
            sys.exit(1)

    # Download SMT datasets (with automatic conversion)
    if args.smt_all:
        smt_output_dir = output_dir / "smt_datasets"
        dataset_keys = list(SMT_DATASETS.keys())
        print(f"Downloading all {len(dataset_keys)} SMT datasets...")
        download_multiple_smt_datasets(dataset_keys, smt_output_dir)
    elif args.smt:
        smt_output_dir = output_dir / "smt_datasets"
        download_multiple_smt_datasets(args.smt, smt_output_dir)
    
    # Convert specific SMT datasets to Primus format (for already downloaded datasets)
    if args.export_smt:
        smt_output_dir = output_dir
        success_count = 0
        for dataset_key in args.export_smt:
            if export_smt_to_primus_format(dataset_key, smt_output_dir):
                success_count += 1
        print(f"Conversion summary: {success_count}/{len(args.export_smt)} datasets converted successfully")
    
    if not (args.primus or args.pdmx_synth or args.smt or args.smt_all or args.export_smt):
        print("Please specify what to download or convert:")
        print("  --primus              Download Camera Primus dataset")
        print("  --pdmx-synth          Download PDMX-Synth dataset (~19GB, 216K image-ABC pairs)")
        print("  --smt <datasets>      Download specific SMT datasets (auto-converts to Primus format)")
        print("  --smt-all             Download all SMT datasets (auto-converts to Primus format)")
        print("  --export-smt <datasets>  Convert existing SMT datasets to Primus format")
        print("  --list-smt            List available SMT datasets")
        print("\nExamples:")
        print("  python scripts/download.py --primus")
        print("  python scripts/download.py --pdmx-synth")
        print("  python scripts/download.py --pdmx-synth --splits train val")
        print("  python scripts/download.py --smt grandstaff")
        print("  python scripts/download.py --smt-all")
        print("  python scripts/download.py --export-smt grandstaff")
        print("  python scripts/download.py --smt grandstaff --export-smt camera-grandstaff")


if __name__ == "__main__":
    main()