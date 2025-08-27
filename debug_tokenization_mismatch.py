#!/usr/bin/env python3
"""
Debug the tokenization mismatch between bekern file content and vocabulary expectations.
"""

import re

def parse_kern(krn: str, krn_format: str = "bekern"):
    """SMT-style parsing function"""
    # Clean forbidden tokens
    forbidden_tokens = ["*staff2", "*staff1", "*Xped", "*tremolo", "*ped", "*Xtuplet", "*tuplet", "*Xtremolo", "*cue", "*Xcue", "*rscale:1/2", "*rscale:1", "*kcancel", "*below"]
    forbidden_pattern = "(" + "|".join([t.replace("*", r"\*") for t in forbidden_tokens]) + ")"
    krn = re.sub(f".*{forbidden_pattern}.*\n", "", krn) # Remove lines containing any of the forbidden tokens
    krn = re.sub(r"(^|(?<=\n))\*(\s\*)*(\n|$)", "", krn) # Remove lines that only contain "*" tokens
    krn = krn.strip()
    
    krn = re.sub(r"(?<=\=)\d+", "", krn)

    krn = krn.replace(" ", " <s> ")
    krn = krn.replace("\t", " <t> ")
    krn = krn.replace("\n", " <b> ")
    krn = krn.replace(" /", "")
    krn = krn.replace(" \\", "")
    krn = krn.replace("·/", "")
    krn = krn.replace("·\\", "")

    if krn_format == "kern":
        krn = krn.replace("·", "").replace('@', '')
    elif krn_format == "ekern":
        krn = krn.replace("·", " ").replace('@', '')
    elif krn_format == "bekern":
        krn = krn.replace("·", " ").replace("@", " ")

    return krn.strip().split(" ")


def load_bekern_labels_current_method(file_content: str):
    """Current bekern loading method from the UnifiedDataset"""
    lines = file_content.strip().split('\n')
    
    tokens = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip header lines that start with **
        if line.startswith('**'):
            continue
            
        # Split by tabs and add all non-empty tokens
        line_tokens = [token for token in line.split('\t') if token.strip()]
        tokens.extend(line_tokens)
        
    return tokens

# Test with realistic bekern content
bekern_content = '''**bekern	**bekern	**dynam
*part1	*part1	*part1
*staff2	*staff1	*staff1/2
*>[A,A,B]	*>[A,A,B]	*>[A,A,B]
*>A	*>A	*>A
*clefF4	*clefG2	*
*k[]	*k[]	*k[]
*M4/4	*M4/4	*M4/4
*met(c)	*met(c)	*met(c)
=1	=1	=1
4E	4cc	.
4F	4dd	.
4G	4ee	.
4A	4ff	.
=2	=2	=2
2B	2gg	.
2c	2aa	.
=3	=3	=3
4d	4bb	.
4e	4ccc	.
4f	4ddd	.
4g	4eee	.
=4	=4	=4
1A	1fff	.
=5	=5	=5
*>B	*>B	*>B
4BB	4dd	.
4C	4ee	.
4D	4ff	.
4E	4gg	.
=6	=6	=6
2F	2aa	.
2G	2bb	.
=7	=7	=7
4A	4ccc	.
4B	4ddd	.
4c	4eee	.
4d	4fff	.
=8	=8	=8
1E	1ggg	.
*-	*-	*-
'''

print("=== Analyzing tokenization mismatch ===\n")

# Method 1: Current bekern loader
print("1. Current bekern loading method:")
current_tokens = load_bekern_labels_current_method(bekern_content)
print(f"   Tokens produced: {len(current_tokens)}")
print(f"   Sample tokens: {current_tokens[:20]}")
print(f"   Musical note tokens: {[t for t in current_tokens if any(c.isdigit() for c in t) and any(c.isalpha() for c in t)][:10]}")

# Method 2: SMT parsing method
print("\n2. SMT parsing method:")
smt_tokens = parse_kern(bekern_content, krn_format="bekern")
print(f"   Tokens produced: {len(smt_tokens)}")
print(f"   Sample tokens: {smt_tokens[:20]}")
print(f"   Musical note tokens: {[t for t in smt_tokens if any(c.isdigit() for c in t) and any(c.isalpha() for c in t)][:10]}")

# Load actual vocabulary to check compatibility
import numpy as np

vocab_path = '/Users/lucawiehe/projects/music_recognition/data/FP_GrandStaff_BeKernw2i.npy'
try:
    vocab_data = np.load(vocab_path, allow_pickle=True).item()
    print(f"\n3. Vocabulary compatibility check:")
    print(f"   Vocabulary size: {len(vocab_data)}")
    
    # Check current method compatibility
    current_found = 0
    current_missing = []
    for token in current_tokens[:20]:  # Check first 20
        if token in vocab_data:
            current_found += 1
        else:
            current_missing.append(token)
    
    print(f"   Current method - found: {current_found}/20, missing: {current_missing[:10]}")
    
    # Check SMT method compatibility
    smt_found = 0
    smt_missing = []
    for token in smt_tokens[:20]:  # Check first 20
        if token in vocab_data:
            smt_found += 1
        else:
            smt_missing.append(token)
    
    print(f"   SMT method - found: {smt_found}/20, missing: {smt_missing[:10]}")
    
    # Check what types of tokens are in the vocabulary
    print(f"\n4. Vocabulary token analysis:")
    note_tokens = [t for t in vocab_data.keys() if re.match(r'^[a-gA-G]+$', t)]
    number_tokens = [t for t in vocab_data.keys() if re.match(r'^\d+$', t)]
    special_tokens = [t for t in vocab_data.keys() if t.startswith('<') and t.endswith('>')]
    meta_tokens = [t for t in vocab_data.keys() if t.startswith('*')]
    
    print(f"   Note tokens (a-g): {len(note_tokens)} - sample: {note_tokens[:10]}")
    print(f"   Number tokens: {len(number_tokens)} - sample: {number_tokens[:10]}")
    print(f"   Special tokens: {len(special_tokens)} - all: {special_tokens}")
    print(f"   Meta tokens (*): {len(meta_tokens)} - sample: {meta_tokens[:10]}")
    
    # Test a specific note conversion
    print(f"\n5. Note conversion example:")
    sample_note = "4cc"  # From our bekern content
    print(f"   Combined token '{sample_note}' in vocab: {sample_note in vocab_data}")
    
    # Try splitting it
    duration = '4'
    note = 'cc'
    print(f"   Duration '{duration}' in vocab: {duration in vocab_data}")
    print(f"   Note '{note}' in vocab: {note in vocab_data}")
    
    if duration in vocab_data and note in vocab_data:
        print(f"   → This suggests tokens should be split: duration + note")
        print(f"   → Current method produces combined tokens, vocabulary expects split tokens")
        
except Exception as e:
    print(f"Error loading vocabulary: {e}")

print(f"\n=== CONCLUSION ===")
print("The issue is a tokenization mismatch:")
print("1. Current bekern loader produces combined tokens like '4cc', '2g'")
print("2. SMT parser splits content into individual components")
print("3. Vocabulary expects individual components like '4', 'cc', '<t>', '<b>'")
print("4. This explains why sequences are short - most tokens become '<pad>' (0)")
print("5. The 7-token length likely comes from only non-musical metadata tokens")
print("   being found in vocabulary (*clef*, *k[]*, etc.)")