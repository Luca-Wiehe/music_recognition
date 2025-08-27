# BeKern Sequence Loading Analysis

## Problem Summary

The ground truth sequences in the UnifiedDataset are only 7 tokens long instead of containing the full music notation because of a **tokenization mismatch** between the bekern file loading method and the vocabulary expectations.

## Root Cause Analysis

### 1. Current BeKern Loading Method (`load_bekern_labels_from_file`)
- **What it does**: Splits bekern files by lines, then by tabs, producing combined tokens
- **Example output**: `['*clefF4', '*clefG2', '*M4/4', '=1', '4cc', '4dd', '2gg', '=2']`
- **Issue**: Produces compound tokens like `'4cc'` (duration + note combined)

### 2. SMT Method (`parse_kern` in third_party/SMT/utils.py)
- **What it does**: Applies comprehensive text processing with separators
- **Example output**: `['*clefF4', '<t>', '*clefG2', '<b>', '*M4/4', '<t>', '4', 'cc', '<t>', '4', 'dd']`
- **Key difference**: Separates components and adds structural tokens (`<t>` for tabs, `<b>` for line breaks)

### 3. Vocabulary Expectations
The BeKern vocabulary (`FP_GrandStaff_BeKernw2i.npy`) contains **181 tokens** including:
- **Individual note tokens**: `'c'`, `'cc'`, `'ccc'`, `'d'`, `'dd'`, etc.
- **Individual duration tokens**: `'1'`, `'2'`, `'4'`, `'8'`, `'16'`, etc.  
- **Separator tokens**: `'<t>'` (tab), `'<b>'` (line break), `'<s>'` (space)
- **Sequence markers**: `'<bos>'`, `'<eos>'`, `'<pad>'`
- **Metadata tokens**: `'*clefG2'`, `'*M4/4'`, `'*k[]'`, etc.

### 4. The Mismatch
- **Current method** produces: `'4cc'` → **NOT FOUND** in vocabulary → becomes `<pad>` (index 0)
- **SMT method** produces: `'4'`, `'cc'` → **BOTH FOUND** in vocabulary → proper indices
- **Result**: Most musical content becomes padding, leaving only ~7 metadata tokens that happen to match

## Evidence

From the vocabulary analysis:
```python
# Combined tokens (current method output)
'4cc' in vocabulary: False
'2gg' in vocabulary: False

# Split tokens (SMT method output)  
'4' in vocabulary: True -> index 20
'cc' in vocabulary: True -> index 122
'<t>' in vocabulary: True -> index 143
```

## Solution

The `load_bekern_labels_from_file` function in `data/utils/format_converter.py` needs to be updated to match the SMT parsing approach:

### Current Implementation (Lines 160-198):
```python
def load_bekern_labels_from_file(file_path: str) -> List[str]:
    # Simply splits by tabs and extends tokens
    line_tokens = [token for token in line.split('\t') if token.strip()]
    tokens.extend(line_tokens)
```

### Required Fix:
Use the SMT `parse_kern` function or implement equivalent logic that:
1. Preserves tab and line break structure with `<t>` and `<b>` tokens
2. Splits compound musical tokens into components
3. Handles special characters and formatting properly
4. Adds sequence markers (`<bos>`, `<eos>`) consistently

### Code Change Required:
```python
def load_bekern_labels_from_file(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Use SMT-compatible parsing
        tokens = parse_kern(content, krn_format="bekern")
        
        # Remove header tokens that were already filtered by parse_kern
        # parse_kern already handles the **bekern header removal
        return tokens[4:]  # Remove first 4 tokens which are headers
        
    except Exception as e:
        logger.error(f"Error reading BeKern label file {file_path}: {e}")
        return []
```

## Impact

This fix will:
1. **Increase sequence lengths** from ~7 tokens to hundreds/thousands of tokens (typical music pieces)
2. **Preserve musical structure** with proper tab/line break markers
3. **Enable proper vocabulary mapping** for musical notes and durations
4. **Maintain compatibility** with SMT model expectations
5. **Fix the tokenization pipeline** end-to-end

## Files to Modify

1. **Primary**: `data/utils/format_converter.py` - Update `load_bekern_labels_from_file()` 
2. **Secondary**: Import `parse_kern` from SMT or implement equivalent logic
3. **Testing**: Update any tests that depend on the current token format