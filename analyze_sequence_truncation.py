#!/usr/bin/env python3
"""
Analyze where sequence truncation might be happening in the data pipeline.
"""

import numpy as np

# Simulate the UnifiedDataset processing steps
def simulate_dataset_processing():
    print("=== Simulating UnifiedDataset processing ===\n")
    
    # Step 1: Load bekern vocabulary
    print("1. Loading BeKern vocabulary...")
    # Simulate vocabulary loading - using the actual path from the codebase
    vocab_path = "/Users/lucawiehe/projects/music_recognition/data/FP_GrandStaff_BeKernw2i.npy"
    
    try:
        vocab_data = np.load(vocab_path, allow_pickle=True).item()
        print(f"   Vocabulary loaded: {len(vocab_data)} tokens")
        print(f"   Sample tokens: {list(vocab_data.keys())[:10]}")
        w2i = vocab_data
        i2w = {idx: word for word, idx in w2i.items()}
    except Exception as e:
        print(f"   Could not load vocabulary: {e}")
        # Create dummy vocabulary for simulation
        w2i = {'<bos>': 1, '<eos>': 2, '<pad>': 0, '4c': 3, '4d': 4, '=1': 5, '*clefG2': 6}
        i2w = {idx: word for word, idx in w2i.items()}
        print(f"   Using dummy vocabulary: {w2i}")
    
    # Step 2: Simulate loading bekern tokens from file
    print("\n2. Loading BeKern tokens from file...")
    sample_tokens = ['*clefF4', '*clefG2', '*k[]', '=1', '4c', '4d', '4e', '4f', '=2', '2g', '2a', '=3', '4b', '4cc', '2dd']
    print(f"   Loaded tokens: {len(sample_tokens)} tokens")
    print(f"   Tokens: {sample_tokens}")
    
    # Step 3: Convert to indices using vocabulary
    print("\n3. Converting to indices...")
    indices = []
    for token in sample_tokens:
        if token in w2i:
            indices.append(w2i[token])
            print(f"   '{token}' -> {w2i[token]}")
        else:
            # Use padding token for unknown tokens
            print(f"   '{token}' -> UNKNOWN (using pad token 0)")
            indices.append(0)
    
    print(f"   Final indices: {indices}")
    print(f"   Sequence length: {len(indices)}")
    
    # Step 4: Check if there are any limits in the collate function
    print("\n4. Checking collate function behavior...")
    # Simulate batch collation
    batch_sequences = [indices, [1, 2, 3], [4, 5, 6, 7, 8, 9, 10]]
    max_len = max(len(seq) for seq in batch_sequences)
    print(f"   Batch sequences: {batch_sequences}")
    print(f"   Max length in batch: {max_len}")
    
    # Pad sequences
    padded_sequences = []
    for seq in batch_sequences:
        padding_len = max_len - len(seq)
        padded_seq = seq + [0] * padding_len
        padded_sequences.append(padded_seq)
    
    print(f"   Padded sequences: {padded_sequences}")
    
    # Step 5: Check if there are any constraints from SMT compatibility
    print("\n5. Checking SMT processing differences...")
    
    # SMT adds <bos> and <eos> markers and uses different parsing
    smt_style_tokens = ['<bos>'] + sample_tokens + ['<eos>']
    print(f"   SMT-style tokens: {len(smt_style_tokens)} tokens")
    print(f"   SMT tokens: {smt_style_tokens}")
    
    # SMT would also include special separators
    smt_with_separators = []
    for i, token in enumerate(sample_tokens):
        smt_with_separators.append(token)
        if i < len(sample_tokens) - 1:
            smt_with_separators.append('<t>')  # Tab separator
    
    smt_full = ['<bos>'] + smt_with_separators + ['<b>'] + ['<eos>']  # Line break + end
    print(f"   SMT with separators: {len(smt_full)} tokens")
    print(f"   First 10: {smt_full[:10]}")
    print(f"   Last 10: {smt_full[-10:]}")

# Check for any hardcoded limits in the original dataset
def check_for_hardcoded_limits():
    print("\n=== Checking for hardcoded sequence limits ===")
    
    # Look for any MAX_LENGTH or similar constants
    print("Looking for potential sequence length limits...")
    
    # Common places where limits might be set:
    potential_limits = [
        ("SMT max_length from data.py", 4360),  # From the SMT code
        ("Common transformer limit", 512),
        ("BERT-style limit", 1024),
        ("Common LSTM limit", 256),
    ]
    
    for name, limit in potential_limits:
        print(f"   {name}: {limit}")
    
    print("\nIf sequences are being truncated to 7 tokens, this suggests:")
    print("   - Either a hardcoded limit is being applied")
    print("   - Or there's an issue with file loading/parsing")
    print("   - Or the vocabulary conversion is failing for most tokens")

# Analyze the specific issue mentioned
def analyze_7_token_issue():
    print("\n=== Analyzing the 7-token issue ===")
    
    print("The problem states ground truth sequences are only 7 tokens long.")
    print("This is suspiciously short for music sequences.")
    print()
    print("Possible causes:")
    print("1. File parsing issue - only reading first line or few tokens")
    print("2. Vocabulary mismatch - most tokens are unknown and filtered out")
    print("3. Hardcoded truncation in data loading")
    print("4. Bug in the bekern format loader")
    print("5. Issue with tab/line parsing in bekern files")
    print()
    
    print("To debug further, we need to:")
    print("1. Check actual bekern files being loaded")
    print("2. Verify vocabulary coverage")
    print("3. Add debug prints to show sequence lengths at each processing step")
    print("4. Compare with SMT processing to see where they differ")

if __name__ == "__main__":
    simulate_dataset_processing()
    check_for_hardcoded_limits()
    analyze_7_token_issue()