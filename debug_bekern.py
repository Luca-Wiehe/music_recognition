#!/usr/bin/env python3
"""
Debug script to understand bekern file loading issue.
"""
import re

def load_bekern_labels_from_file(file_path: str):
    """
    Load BeKern labels from a .semantic file.
    
    BeKern files have a different format - they start with **ekern header
    and contain tab-separated data organized in columns (staves).
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
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
        
    except FileNotFoundError:
        print(f"BeKern label file not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading BeKern label file {file_path}: {e}")
        return []

def parse_kern(krn: str, krn_format: str = "bekern"):
    """SMT-style parsing function"""
    # Clean forbidden tokens
    forbidden_tokens = ["*staff2", "*staff1", "*Xped", "*tremolo", "*ped", "*Xtuplet", "*tuplet", "*Xtremolo", "*cue", "*Xcue", "*rscale:1/2", "*rscale:1", "*kcancel", "*below"]
    forbidden_pattern = "(" + "|".join([t.replace("*", "\\*") for t in forbidden_tokens]) + ")"
    krn = re.sub(f".*{forbidden_pattern}.*\n", "", krn) # Remove lines containing any of the forbidden tokens
    krn = re.sub("(^|(?<=\n))\*(\s\*)*(\n|$)", "", krn) # Remove lines that only contain "*" tokens
    krn = krn.strip()
    
    krn = re.sub("(?<=\=)\d+", "", krn)

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

# Test with sample bekern content
test_content = '''**bekern
*staff1	*staff2
*clefG2	*clefF4
*M4/4	*M4/4
=1	=1
4c	4C
4d	4D
4e	4E
4f	4F
=2	=2
2g	2G
2a	2A
=	=
*-	*-
'''

print("=== Testing current bekern loading function ===")
with open('/tmp/test_bekern.semantic', 'w') as f:
    f.write(test_content)

tokens = load_bekern_labels_from_file('/tmp/test_bekern.semantic')
print(f'Current loading method - {len(tokens)} tokens: {tokens}')

print("\n=== Testing SMT-style parsing ===")
# Test the SMT way
smt_tokens = parse_kern(test_content, krn_format="bekern")
print(f'SMT parsing method - {len(smt_tokens)} tokens (first 10): {smt_tokens[:10]}')
print(f'SMT parsing method - all tokens: {smt_tokens}')

# Now test with just the raw lines
print("\n=== Testing with longer realistic content ===")
longer_content = '''**bekern	**bekern	**dynam
*part1	*part1	*part1
*staff2	*staff1	*staff1/2
*>[A,A,B]	*>[A,A,B]	*>[A,A,B]
*>A	*>A	*>A
*clefF4	*clefG2	*
*k[]	*k[]	*k[]
*M4/4	*M4/4	*M4/4
*met(c)	*met(c)	*met(c)
=1	=1	=1
[4E	[4cc	.
4F	4dd	.
4G	4ee	.
4A]	4ff]	.
=2	=2	=2
[2B	[2gg	.
2c]	2aa]	.
=3	=3	=3
[4d	[4bb	.
4e	4ccc	.
4f	4ddd	.
4g]	4eee]	.
=4	=4	=4
1A	1fff	.
=5	=5	=5
*>B	*>B	*>B
[4BB	[4dd	.
4C	4ee	.
4D	4ff	.
4E]	4gg]	.
=6	=6	=6
[2F	[2aa	.
2G]	2bb]	.
=7	=7	=7
[4A	[4ccc	.
4B	4ddd	.
4c	4eee	.
4d]	4fff]	.
=8	=8	=8
1E	1ggg	.
*-	*-	*-
'''

with open('/tmp/test_longer.semantic', 'w') as f:
    f.write(longer_content)

tokens_longer = load_bekern_labels_from_file('/tmp/test_longer.semantic')
print(f'Current loading method - {len(tokens_longer)} tokens (first 20): {tokens_longer[:20]}')

smt_tokens_longer = parse_kern(longer_content, krn_format="bekern")
print(f'SMT parsing method - {len(smt_tokens_longer)} tokens (first 20): {smt_tokens_longer[:20]}')

# Test with original line parsing
print("\n=== Testing line by line differences ===")
lines = longer_content.strip().split('\n')
for i, line in enumerate(lines[:10]):  # First 10 lines
    if not line.strip() or line.startswith('**'):
        continue
    print(f"Line {i}: '{line}'")
    line_tokens = [token for token in line.split('\t') if token.strip()]
    print(f"  Tokens: {line_tokens}")