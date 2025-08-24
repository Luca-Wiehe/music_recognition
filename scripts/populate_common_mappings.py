#!/usr/bin/env python3
"""
Script to populate common token mappings in the primus_to_bekern_mapping.json file.
This fills in the most common and straightforward conversions based on the analysis
of BeKern vocabulary and music theory knowledge.
"""

import json
import re
from pathlib import Path

def parse_primus_token(token):
    """Parse a Primus token to extract its components."""
    if token.startswith('note-'):
        # Format: note-C4_quarter_fermata or note-A#3_eighth.
        parts = token[5:].split('_')  # Remove 'note-' prefix
        pitch = parts[0]  # e.g., 'C4', 'A#3'
        duration = parts[1] if len(parts) > 1 else 'quarter'
        
        # Extract modifiers (fermata, dots)
        modifiers = []
        if len(parts) > 2:
            for part in parts[2:]:
                if part == 'fermata':
                    modifiers.append('fermata')
                elif part.endswith('.'):
                    modifiers.append('dot')
                elif part == '.':
                    modifiers.append('dot')
                elif part == '..':
                    modifiers.append('double_dot')
                    
        return 'note', pitch, duration, modifiers
        
    elif token.startswith('rest-'):
        # Format: rest-quarter_fermata
        parts = token[5:].split('_')  # Remove 'rest-' prefix
        duration = parts[0]
        modifiers = []
        if len(parts) > 1:
            for part in parts[1:]:
                if part == 'fermata':
                    modifiers.append('fermata')
                elif part.endswith('.'):
                    modifiers.append('dot')
                    
        return 'rest', None, duration, modifiers
        
    elif token.startswith('clef-'):
        return 'clef', token[5:], None, []
        
    elif token.startswith('keySignature-'):
        return 'keysig', token[13:], None, []
        
    elif token.startswith('timeSignature-'):
        return 'timesig', token[14:], None, []
        
    elif token.startswith('multirest-'):
        return 'multirest', token[10:], None, []
        
    return 'other', token, None, []

def pitch_to_kern(pitch_str):
    """Convert Primus pitch notation to Kern notation."""
    # Parse pitch like "C4", "A#3", "Bb5"
    match = re.match(r'([A-G][#b]?)(\d+)', pitch_str)
    if not match:
        return None
        
    note_name, octave = match.groups()
    octave = int(octave)
    
    # Convert note name
    base_note = note_name[0].lower()
    
    # Handle accidentals
    accidental = ""
    if len(note_name) > 1:
        if note_name[1] == '#':
            accidental = "#"
        elif note_name[1] == 'b':
            accidental = "-"
            
    # Convert octave to Kern representation
    # Kern uses: C,, C, C c c' c'' c'''
    # Octave 3 = c, Octave 4 = c', Octave 5 = c''
    if octave <= 2:
        # Very low octaves use uppercase with commas
        kern_note = base_note.upper()
        if octave == 1:
            kern_note += ","
        elif octave == 0:
            kern_note += ",,"
    elif octave == 3:
        # Middle octave uses lowercase
        kern_note = base_note
    else:
        # Higher octaves use lowercase with apostrophes
        kern_note = base_note
        for _ in range(octave - 3):
            kern_note += "'"
            
    return kern_note + accidental

def duration_to_kern(duration_str):
    """Convert Primus duration to Kern duration number."""
    duration_map = {
        'whole': '1',
        'half': '2', 
        'quarter': '4',
        'eighth': '8',
        'sixteenth': '16',
        'thirty_second': '32',
        'sixty_fourth': '64',
        'hundred_twenty_eighth': '128',
        'double_whole': '0',  # Breve
        'quadruple_whole': '00'  # Long
    }
    return duration_map.get(duration_str, '4')  # Default to quarter

def create_bekern_mapping():
    """Create BeKern mappings for common Primus tokens."""
    mappings = {}
    
    # Read existing mapping file
    project_root = Path(__file__).parent.parent
    mapping_file = project_root / "data" / "primus_to_bekern_mapping.json"
    
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
    
    # Fill in common mappings
    for token in mappings.keys():
        if mappings[token]:  # Skip already filled mappings
            continue
            
        token_type, component, duration, modifiers = parse_primus_token(token)
        
        if token_type == 'note':
            # Convert note
            kern_pitch = pitch_to_kern(component)
            kern_duration = duration_to_kern(duration)
            if kern_pitch and kern_duration:
                bekern_token = kern_duration + kern_pitch
                # Add dots for dotted notes
                if 'dot' in modifiers:
                    bekern_token += "."
                mappings[token] = bekern_token
                
        elif token_type == 'rest':
            # Convert rest
            kern_duration = duration_to_kern(duration)
            bekern_token = kern_duration + "r"
            if 'dot' in modifiers:
                bekern_token += "."
            mappings[token] = bekern_token
            
        elif token_type == 'clef':
            # Convert clef
            clef_map = {
                'G1': '*clefG1',
                'G2': '*clefG2', 
                'C1': '*clefC1',
                'C2': '*clefC2',
                'C3': '*clefC3',
                'C4': '*clefC4',
                'C5': '*clefC5',
                'F3': '*clefF3',
                'F4': '*clefF4',
                'F5': '*clefF5'
            }
            if component in clef_map:
                mappings[token] = clef_map[component]
                
        elif token_type == 'keysig':
            # Convert key signature  
            keysig_map = {
                'CM': '*k[]',  # C major (no accidentals)
                'GM': '*k[f#]',  # G major
                'DM': '*k[f#c#]',  # D major
                'AM': '*k[f#c#g#]',  # A major
                'EM': '*k[f#c#g#d#]',  # E major
                'BM': '*k[f#c#g#d#a#]',  # B major
                'F#M': '*k[f#c#g#d#a#e#]',  # F# major
                'C#M': '*k[f#c#g#d#a#e#b#]',  # C# major
                'FM': '*k[b-]',  # F major
                'BbM': '*k[b-e-]',  # Bb major  
                'EbM': '*k[b-e-a-]',  # Eb major
                'AbM': '*k[b-e-a-d-]',  # Ab major
                'DbM': '*k[b-e-a-d-g-]',  # Db major
                'GbM': '*k[b-e-a-d-g-c-]',  # Gb major
            }
            if component in keysig_map:
                mappings[token] = keysig_map[component]
                
        elif token_type == 'timesig':
            # Convert time signature
            mappings[token] = f"*M{component}"
            
        elif token == 'barline':
            mappings[token] = "="
            
        elif token == 'tie':
            mappings[token] = "_"  # Tie character in Kern
            
    # Save updated mappings
    with open(mapping_file, 'w') as f:
        json.dump(mappings, f, indent=2, sort_keys=True)
    
    # Count filled mappings
    filled = sum(1 for v in mappings.values() if v)
    total = len(mappings)
    
    print(f"Updated mapping file: {mapping_file}")
    print(f"Filled mappings: {filled}/{total} ({filled/total*100:.1f}%)")
    
    # Show some examples
    print("\nSample mappings:")
    count = 0
    for token, bekern in mappings.items():
        if bekern and count < 10:
            print(f"  {token} -> {bekern}")
            count += 1

if __name__ == "__main__":
    create_bekern_mapping()