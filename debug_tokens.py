#!/usr/bin/env python3
"""
Debug script to understand why BeKern sequences are being reduced to only 3 tokens.
Examines the data processing pipeline step by step.
"""

import torch
import numpy as np
from pathlib import Path
from data.data_loading.unified_dataset import UnifiedDataset
from data.utils.format_converter import PrimusToBeKernConverter, load_primus_labels_from_file

def debug_token_conversion():
    """Debug the token conversion process step by step."""
    
    print("=== DEBUGGING TOKEN CONVERSION PROCESS ===\n")
    
    # Test paths - adjust these based on your actual data structure
    data_path = "data/primus"
    vocab_path = "data/bekern_vocabulary.npy" 
    mapping_path = "data/primus_to_bekern_mapping.json"
    
    # Check if files exist
    if not Path(data_path).exists():
        print(f"❌ Data path doesn't exist: {data_path}")
        return
    if not Path(vocab_path).exists():
        print(f"❌ Vocabulary path doesn't exist: {vocab_path}")
        return
    if not Path(mapping_path).exists():
        print(f"❌ Mapping path doesn't exist: {mapping_path}")
        return
    
    print("✅ All required files exist")
    
    # 1. Load and examine the converter
    print("\n1. EXAMINING FORMAT CONVERTER")
    converter = PrimusToBeKernConverter(mapping_path)
    stats = converter.get_conversion_stats()
    print(f"   Total mappings available: {stats['total_mappings']}")
    print(f"   Missing tokens encountered so far: {stats['missing_tokens_encountered']}")
    
    # Count empty mappings
    empty_mappings = sum(1 for token, bekern in converter.token_mapping.items() if bekern == "")
    print(f"   Empty mappings (tokens that map to empty string): {empty_mappings}")
    print(f"   Valid mappings: {stats['total_mappings'] - empty_mappings}")
    
    # 2. Load BeKern vocabulary
    print("\n2. EXAMINING BEKERN VOCABULARY")
    vocab_data = np.load(vocab_path, allow_pickle=True).item()
    print(f"   Vocabulary type: {type(vocab_data)}")
    print(f"   Vocabulary size: {len(vocab_data)}")
    
    # Show some sample tokens
    if isinstance(vocab_data, dict):
        sample_tokens = list(vocab_data.items())[:10]
        print(f"   Sample tokens: {sample_tokens}")
        w2i = vocab_data
        i2w = {idx: word for word, idx in w2i.items()}
    else:
        print(f"   Unexpected vocabulary format: {type(vocab_data)}")
        return
    
    # 3. Find a sample file
    print("\n3. EXAMINING A SAMPLE FILE")
    
    # Try to find a sample semantic file
    data_dir = Path(data_path)
    semantic_files = list(data_dir.rglob("*.semantic"))
    
    if not semantic_files:
        print("   ❌ No .semantic files found")
        return
    
    sample_file = semantic_files[0]
    print(f"   Using sample file: {sample_file}")
    
    # Load original Primus tokens
    primus_tokens = load_primus_labels_from_file(str(sample_file))
    print(f"   Original Primus tokens ({len(primus_tokens)}): {primus_tokens[:20]}{'...' if len(primus_tokens) > 20 else ''}")
    
    # Convert to BeKern step by step
    print(f"\n4. TOKEN-BY-TOKEN CONVERSION")
    bekern_tokens_manual = []
    skipped_tokens = []
    
    for i, token in enumerate(primus_tokens[:20]):  # Only check first 20 for debugging
        bekern_token = converter.convert_token(token)
        if bekern_token is None:
            skipped_tokens.append((i, token, "no mapping"))
        elif bekern_token == "":
            skipped_tokens.append((i, token, "empty mapping"))
        else:
            bekern_tokens_manual.append(bekern_token)
        
        print(f"   {i:2d}: '{token}' -> '{bekern_token}' {'[SKIPPED]' if bekern_token in [None, ''] else '[KEPT]'}")
    
    # Compare with converter.convert_sequence
    bekern_tokens_auto = converter.convert_sequence(primus_tokens)
    bekern_tokens_with_markers = converter.convert_sequence_with_markers(primus_tokens)
    
    print(f"\n5. CONVERSION RESULTS COMPARISON")
    print(f"   Manual conversion (first 20): {bekern_tokens_manual}")
    print(f"   Automatic conversion (full): {bekern_tokens_auto[:20]}{'...' if len(bekern_tokens_auto) > 20 else ''}")
    print(f"   With markers (full): {bekern_tokens_with_markers[:20]}{'...' if len(bekern_tokens_with_markers) > 20 else ''}")
    print(f"   Skipped tokens: {skipped_tokens}")
    
    # 6. Test vocabulary conversion
    print(f"\n6. VOCABULARY INDEX CONVERSION")
    indices = []
    unknown_tokens = []
    
    for token in bekern_tokens_with_markers[:10]:  # Test first 10
        if token in w2i:
            idx = w2i[token]
            indices.append(idx)
            print(f"   '{token}' -> {idx} ({i2w.get(idx, 'UNKNOWN')})")
        else:
            unknown_tokens.append(token)
            print(f"   '{token}' -> UNKNOWN (not in vocabulary)")
    
    print(f"   Final indices: {indices}")
    print(f"   Unknown tokens: {unknown_tokens}")
    
    # 7. Test with dataset 
    print(f"\n7. TESTING WITH UNIFIED DATASET")
    try:
        dataset = UnifiedDataset(
            data_paths=[data_path],
            bekern_vocab_path=vocab_path,
            mapping_file_path=mapping_path,
            split='train' if Path(data_path + '/train').exists() else None
        )
        
        if len(dataset) > 0:
            sample_image, sample_labels = dataset[0]
            print(f"   Dataset sample 0:")
            print(f"   Image shape: {sample_image.shape}")
            print(f"   Label tensor: {sample_labels}")
            print(f"   Label tensor length: {len(sample_labels)}")
            
            # Convert back to tokens
            label_tokens = [i2w.get(int(idx), f'UNK:{idx}') for idx in sample_labels]
            print(f"   Label tokens: {label_tokens}")
            
            # Get conversion stats
            conv_stats = dataset.get_conversion_stats()
            print(f"   Conversion stats: {conv_stats}")
        else:
            print("   ❌ Dataset is empty")
            
    except Exception as e:
        print(f"   ❌ Error creating dataset: {e}")
    
    print(f"\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    debug_token_conversion()