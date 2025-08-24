#!/usr/bin/env python3
"""
Script to generate a JSON template mapping file from semantic_labels.txt.
This creates a template where each Camera Primus token is mapped to a placeholder
that can be manually filled with the corresponding BeKern token.
"""

import json
import os
from pathlib import Path

def parse_semantic_labels(labels_file):
    """Parse semantic_labels.txt and extract all token names."""
    tokens = []
    
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '→' in line:
                # Format: "1→barline" 
                parts = line.split('→', 1)
                if len(parts) == 2:
                    token = parts[1]
                    tokens.append(token)
            elif line and not line.startswith('#'):
                # Direct token format (no line numbers)
                tokens.append(line)
    
    return tokens

def create_mapping_template(tokens, output_file):
    """Create JSON mapping template with Camera Primus tokens as keys."""
    mapping = {}
    
    for token in tokens:
        # Create placeholder based on token type
        if token.startswith('note-'):
            mapping[token] = ""  # To be filled with kern notation like "4c"
        elif token.startswith('rest-'):
            mapping[token] = ""  # To be filled with kern rests like "4r"
        elif token.startswith('gracenote-'):
            mapping[token] = ""  # To be filled with grace note notation
        elif token.startswith('clef-'):
            mapping[token] = ""  # To be filled with "*clef*" format
        elif token.startswith('keySignature-'):
            mapping[token] = ""  # To be filled with "*k[...]" format
        elif token.startswith('timeSignature-'):
            mapping[token] = ""  # To be filled with "*M*" format
        elif token.startswith('multirest-'):
            mapping[token] = ""  # To be filled with multirest notation
        elif token == 'barline':
            mapping[token] = "="  # Common barline representation
        elif token == 'tie':
            mapping[token] = ""  # To be filled with tie notation
        else:
            mapping[token] = ""  # Generic placeholder
    
    # Sort the mapping by token name for better organization
    sorted_mapping = dict(sorted(mapping.items()))
    
    # Write to JSON file with nice formatting
    with open(output_file, 'w') as f:
        json.dump(sorted_mapping, f, indent=2, sort_keys=True)
    
    print(f"Generated mapping template with {len(sorted_mapping)} tokens")
    print(f"Template saved to: {output_file}")
    
    # Print some statistics
    token_types = {}
    for token in tokens:
        token_type = token.split('-')[0] if '-' in token else token
        token_types[token_type] = token_types.get(token_type, 0) + 1
    
    print("\nToken type distribution:")
    for token_type, count in sorted(token_types.items()):
        print(f"  {token_type}: {count}")

def main():
    """Main function to generate the mapping template."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Input and output paths
    labels_file = project_root / "data" / "semantic_labels.txt"
    output_file = project_root / "data" / "primus_to_bekern_mapping.json"
    
    if not labels_file.exists():
        print(f"Error: semantic_labels.txt not found at {labels_file}")
        return
    
    print(f"Reading Camera Primus tokens from: {labels_file}")
    
    # Parse tokens and create template
    tokens = parse_semantic_labels(labels_file)
    create_mapping_template(tokens, output_file)
    
    print(f"\nNext steps:")
    print(f"1. Edit {output_file} to fill in BeKern equivalents")
    print(f"2. Start with the most common tokens (notes, rests, clefs)")
    print(f"3. Use the SMT vocabulary files as reference for valid BeKern tokens")

if __name__ == "__main__":
    main()