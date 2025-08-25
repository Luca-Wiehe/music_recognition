#!/usr/bin/env python3
"""
Test script to verify BeKern file loading works correctly.
"""

import sys
sys.path.append('/home/stud/wiel/music_recognition')

from data.utils.format_converter import load_bekern_labels_from_file

# Test the BeKern loading function
test_file = "/home/stud/wiel/music_recognition/data/datasets/smt_datasets/camera-grandstaff/train/sample_032271/sample_032271.semantic"

print("Testing BeKern file loading...")
print(f"File: {test_file}")

tokens = load_bekern_labels_from_file(test_file)
print(f"Number of tokens loaded: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")
print(f"Last 10 tokens: {tokens[-10:]}")
