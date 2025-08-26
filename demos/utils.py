"""
Utility functions for luca_model demo notebook.
Provides functions for model loading, image preprocessing, bekern decoding, and music visualization.
"""

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.luca_model import MusicTrOCR
from data.data_loading.unified_dataset import UnifiedDataset
import torchvision.transforms as transforms

# SMT Verovio imports
sys.path.append('third_party/SMT')
try:
    import verovio
    VEROVIO_AVAILABLE = True
except ImportError:
    VEROVIO_AVAILABLE = False
    print("Warning: Verovio not available. Music visualization will be limited.")


def load_bekern_vocabulary(vocab_path: str = "data/FP_GrandStaff_BeKernw2i.npy") -> Dict:
    """Load bekern vocabulary mapping from numpy file."""
    vocab_dict = np.load(vocab_path, allow_pickle=True).item()
    # Create reverse mapping (id to token)
    id_to_token = {v: k for k, v in vocab_dict.items()}
    return vocab_dict, id_to_token


def load_model_and_vocab(ckpt_path: str, vocab_path: str = "data/FP_GrandStaff_BeKernw2i.npy") -> Tuple[MusicTrOCR, Dict, Dict]:
    """
    Load trained MusicTrOCR model and vocabulary.
    
    Args:
        ckpt_path: Path to model checkpoint
        vocab_path: Path to bekern vocabulary file
        
    Returns:
        Tuple of (model, vocab_dict, id_to_token)
    """
    # Load vocabulary
    vocab_dict, id_to_token = load_bekern_vocabulary(vocab_path)
    vocab_size = len(vocab_dict)
    
    # Initialize model with same config as training
    model = MusicTrOCR(
        vocab_size=vocab_size,
        vision_model_name="facebook/convnext-tiny-224",
        d_model=512,
        n_heads=8,
        n_decoder_layers=6,
        d_ff=2048,
        max_seq_len=1512,
        dropout=0.1
    )
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {ckpt_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Device: {device}")
    
    return model, vocab_dict, id_to_token


def preprocess_image(img_path: str, target_height: int = 128) -> torch.Tensor:
    """
    Preprocess image for model input following training pipeline.
    
    Args:
        img_path: Path to input image
        target_height: Target height for resizing
        
    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """
    # Load image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    # Convert to PIL for transforms
    image = Image.fromarray(image).convert('RGB')
    
    # Calculate target width maintaining aspect ratio
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    target_width = int(target_height * aspect_ratio)
    
    # Define transforms (matching training pipeline)
    transform = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    print(f"Image preprocessed: {img_path}")
    print(f"Original size: {original_width}x{original_height}")
    print(f"Tensor size: {tensor.shape}")
    
    return tensor


def decode_bekern_prediction(token_ids: torch.Tensor, id_to_token: Dict, model: MusicTrOCR) -> str:
    """
    Decode model token predictions to bekern string.
    
    Args:
        token_ids: Predicted token IDs from model
        id_to_token: Token ID to string mapping
        model: Model instance for vocabulary conversion
        
    Returns:
        Decoded bekern string
    """
    print(f"\n=== DEBUGGING TOKEN DECODING ===")
    print(f"Raw model tokens: {token_ids.squeeze().tolist()}")
    
    # Convert model tokens back to dataset vocabulary
    dataset_tokens = model.decode_model_tokens_to_dataset(token_ids)
    print(f"Dataset tokens after conversion: {dataset_tokens.squeeze().tolist()}")
    
    # Check what happens at each step
    raw_tokens = token_ids.squeeze().tolist()
    dataset_tokens_list = dataset_tokens.squeeze().tolist()
    
    print(f"\nToken conversion analysis:")
    print(f"Model vocab: PAD={model.PAD_TOKEN_ID}, START={model.START_TOKEN_ID}, END={model.END_TOKEN_ID}")
    
    # Show what tokens 0, 1, 2 actually map to in bekern vocabulary
    print(f"BeKern vocab mapping for low IDs:")
    for tid in [0, 1, 2, 104, 185, 276]:  # Check both model special tokens and actual bekern special tokens
        if tid in id_to_token:
            print(f"  Dataset ID {tid} -> '{id_to_token[tid]}'")
    
    # Convert to string tokens
    bekern_tokens = []
    for i, (model_tok, dataset_tok) in enumerate(zip(raw_tokens, dataset_tokens_list)):
        print(f"  Step {i}: model_token={model_tok} -> dataset_token={dataset_tok}", end="")
        
        # Skip model's internal special tokens, but process everything else
        if model_tok == model.START_TOKEN_ID:
            print(f" -> START_TOKEN (skipping)")
            continue
        elif model_tok == model.PAD_TOKEN_ID:
            print(f" -> PAD_TOKEN (skipping)")
            continue
        elif dataset_tok in id_to_token:
            token = id_to_token[dataset_tok]
            print(f" -> '{token}'")
            
            # Stop at actual <eos> token, not model's END_TOKEN_ID
            if token == '<eos>':
                print(f"    Found actual <eos> token, stopping")
                break
            # Skip other special tokens but continue processing
            elif token in ['<pad>', '<bos>']:
                print(f"    Skipping special token")
                continue
            else:
                bekern_tokens.append(token)
        else:
            print(f" -> UNKNOWN_TOKEN_ID")
            print(f"Warning: Unknown token ID {dataset_tok} (from model token {model_tok})")
    
    # Join tokens with spaces
    bekern_str = ' '.join(bekern_tokens)
    
    print(f"\nDecoding summary:")
    print(f"  Total model tokens: {len(raw_tokens)}")
    print(f"  Valid bekern tokens: {len(bekern_tokens)}")
    print(f"  Final bekern string length: {len(bekern_str)} chars")
    print(f"  Bekern preview: {bekern_str[:100]}...")
    print(f"=== END DEBUG ===\n")
    
    return bekern_str


def bekern_to_kern(bekern_str: str) -> str:
    """
    Convert bekern format to kern format for Verovio rendering.
    Handles the flattened bekern sequence by parsing and reconstructing proper kern structure.
    
    Args:
        bekern_str: Bekern notation string
        
    Returns:
        Kern format string
    """
    # First, replace bekern special tokens
    kern_str = bekern_str.replace('<b>', '\n')
    kern_str = kern_str.replace('<s>', ' ')  
    kern_str = kern_str.replace('<t>', '\t')
    
    # Split into tokens for analysis
    tokens = bekern_str.split()
    
    # Separate metadata from music content
    metadata_tokens = []
    music_tokens = []
    
    for token in tokens:
        if token.startswith('*') and not token in ['*-']:
            metadata_tokens.append(token)
        elif token not in ['<b>', '<s>', '<t>', '<bos>', '<eos>', '<pad>']:
            music_tokens.append(token)
    
    # Build proper kern structure
    kern_lines = []
    
    # Add header
    kern_lines.append('**kern\t**kern')
    
    # Add metadata (clef, key, meter) - assuming two parts
    clef_tokens = [t for t in metadata_tokens if t.startswith('*clef')]
    key_tokens = [t for t in metadata_tokens if t.startswith('*k')]
    meter_tokens = [t for t in metadata_tokens if t.startswith('*M')]
    met_tokens = [t for t in metadata_tokens if t.startswith('*met')]
    
    # Add clefs
    if clef_tokens:
        if len(clef_tokens) >= 2:
            kern_lines.append(f'{clef_tokens[0]}\t{clef_tokens[1]}')
        else:
            kern_lines.append(f'{clef_tokens[0]}\t{clef_tokens[0]}')
    
    # Add key signatures
    if key_tokens:
        if len(key_tokens) >= 2:
            kern_lines.append(f'{key_tokens[0]}\t{key_tokens[1]}')
        else:
            kern_lines.append(f'{key_tokens[0]}\t{key_tokens[0]}')
    
    # Add time signatures
    if meter_tokens:
        if len(meter_tokens) >= 2:
            kern_lines.append(f'{meter_tokens[0]}\t{meter_tokens[1]}')
        else:
            kern_lines.append(f'{meter_tokens[0]}\t{meter_tokens[0]}')
    
    # Add meter markings
    if met_tokens:
        if len(met_tokens) >= 2:
            kern_lines.append(f'{met_tokens[0]}\t{met_tokens[1]}')
        else:
            kern_lines.append(f'{met_tokens[0]}\t{met_tokens[0]}')
    
    # Add music content - group tokens in pairs for two parts
    if music_tokens:
        # Simple approach: alternate tokens between left and right hand
        left_hand = []
        right_hand = []
        
        for i, token in enumerate(music_tokens):
            if i % 2 == 0:
                left_hand.append(token)
            else:
                right_hand.append(token)
        
        # Pad shorter voice with rests
        max_len = max(len(left_hand), len(right_hand))
        while len(left_hand) < max_len:
            left_hand.append('r')
        while len(right_hand) < max_len:
            right_hand.append('r')
        
        # Add paired music lines
        for left, right in zip(left_hand, right_hand):
            kern_lines.append(f'{left}\t{right}')
    
    # Add ending marker
    kern_lines.append('*-\t*-')
    
    result = '\n'.join(kern_lines)
    
    # Debug output
    print("Parsed tokens:")
    print(f"  Metadata: {metadata_tokens}")
    print(f"  Music: {music_tokens}")
    print("Generated Kern structure:")
    for i, line in enumerate(kern_lines[:5]):
        print(f"  {i+1}: {repr(line)}")
    
    return result


def render_music_score(kern_str: str, output_path: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Render kern notation to image using Verovio.
    
    Args:
        kern_str: Kern format notation
        output_path: Optional path to save rendered image
        
    Returns:
        Rendered image as numpy array, or None if Verovio unavailable
    """
    if not VEROVIO_AVAILABLE:
        print("Verovio not available. Cannot render music score.")
        return None
    
    try:
        # Initialize Verovio toolkit
        verovio.enableLog(verovio.LOG_OFF)  # Disable verbose logging
        tk = verovio.toolkit()
        
        # Set rendering options
        tk.setOptions({
            "pageWidth": 2100,
            "footer": "none",
            "header": "none",
            "scale": 40,
            "adjustPageHeight": True
        })
        
        # Debug: print kern string for analysis
        print("Kern string to be rendered:")
        print("-" * 40)
        print(kern_str)
        print("-" * 40)
        
        # Validate kern format before loading
        if not kern_str.strip():
            print("Error: Empty kern string")
            return None
        
        if not '**kern' in kern_str:
            print("Error: Invalid kern format - missing **kern header")
            return None
        
        # Load data and render
        try:
            tk.loadData(kern_str)
        except Exception as load_error:
            print(f"Error loading kern data into Verovio: {load_error}")
            print("This usually indicates invalid kern syntax")
            return None
        
        try:
            svg = tk.renderToSVG()
        except Exception as render_error:
            print(f"Error rendering to SVG: {render_error}")
            return None
        
        if not svg or len(svg.strip()) == 0:
            print("Error: Verovio returned empty SVG")
            return None
        
        # Convert SVG to PNG using cairosvg
        try:
            from cairosvg import svg2png
            png_data = svg2png(bytestring=svg.encode('utf-8'), background_color='white')
            
            # Convert to numpy array
            image_array = cv2.imdecode(np.frombuffer(png_data, np.uint8), -1)
            
            if image_array is None:
                print("Error: Failed to decode PNG data")
                return None
                
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Save if path provided
            if output_path:
                cv2.imwrite(output_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
                print(f"Rendered score saved to: {output_path}")
            
            print("✅ Music score rendered successfully")
            return image_array
            
        except ImportError:
            print("cairosvg not available. Cannot convert SVG to PNG.")
            return None
        except Exception as svg_error:
            print(f"Error converting SVG to PNG: {svg_error}")
            return None
            
    except Exception as e:
        print(f"Unexpected error rendering music score: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_inference(model: MusicTrOCR, image_tensor: torch.Tensor, vocab_dict: Dict, max_length: int = 512) -> torch.Tensor:
    """
    Run inference on preprocessed image with corrected END token detection.
    
    Args:
        model: Trained MusicTrOCR model
        image_tensor: Preprocessed image tensor
        vocab_dict: BeKern vocabulary dictionary
        max_length: Maximum generation length
        
    Returns:
        Generated token sequence
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Find the actual EOS token ID in model vocabulary space
    bekern_eos_id = vocab_dict.get('<eos>', None)
    if bekern_eos_id is not None:
        # Convert bekern EOS ID to model vocabulary space
        model_eos_id = bekern_eos_id + (model.FIRST_MUSIC_TOKEN_ID - 1)  # +2 offset
        print(f"Corrected EOS: bekern_id={bekern_eos_id} -> model_id={model_eos_id}")
    else:
        model_eos_id = model.END_TOKEN_ID  # Fallback to original
        print(f"Warning: <eos> not found in vocabulary, using model END_TOKEN_ID={model_eos_id}")
    
    print("Running inference with corrected generation logic...")
    
    # Custom generation loop with correct END token detection
    model.eval()
    batch_size = image_tensor.shape[0]
    
    # Encode images once
    memory = model.encode_image(image_tensor)
    
    # Initialize with start tokens  
    sequences = torch.full((batch_size, 1), model.START_TOKEN_ID, device=device)
    
    # Generate tokens one by one
    for step in range(max_length - 1):
        # Create padding mask
        tgt_key_padding_mask = torch.zeros(batch_size, sequences.shape[1], 
                                         device=device, dtype=torch.bool)
        
        # Forward pass
        logits = model.decoder(sequences, memory, tgt_key_padding_mask)
        
        # Get logits for the last position
        next_token_logits = logits[:, -1, :] / 1.0  # temperature
        
        # Greedy decoding
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Append to sequences
        sequences = torch.cat([sequences, next_tokens], dim=1)
        
        print(f"Step {step}: Generated token {next_tokens.item()}")
        
        # Check for correct END token (not model.END_TOKEN_ID!)
        if (next_tokens == model_eos_id).all():
            print(f"Found actual EOS token {model_eos_id}, stopping generation")
            break
        
        # Safety check - also stop on model's internal END_TOKEN_ID but warn
        if (next_tokens == model.END_TOKEN_ID).all():
            print(f"WARNING: Hit model END_TOKEN_ID {model.END_TOKEN_ID} (incorrect), stopping")
            break
    
    print(f"Generated sequence length: {sequences.shape[1]} (including START token)")
    return sequences


def visualize_results(original_img_path: str, rendered_score: Optional[np.ndarray], bekern_str: str):
    """
    Display original image and rendered score side by side.
    
    Args:
        original_img_path: Path to original input image
        rendered_score: Rendered score image array (or None)
        bekern_str: Decoded bekern string for display
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Load and display original image
    original_img = cv2.imread(original_img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Input Image", fontsize=14)
    axes[0].axis('off')
    
    # Display rendered score or placeholder
    if rendered_score is not None:
        axes[1].imshow(rendered_score)
        axes[1].set_title("Predicted Music Score", fontsize=14)
    else:
        axes[1].text(0.5, 0.5, "Music rendering\nnot available\n(Verovio required)", 
                    ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
        axes[1].set_title("Predicted Music Score (Unavailable)", fontsize=14)
    
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display bekern text preview
    print("\nPredicted BeKern Notation (first 200 characters):")
    print("-" * 50)
    print(bekern_str[:200] + "..." if len(bekern_str) > 200 else bekern_str)


def demo_pipeline(img_path: str, ckpt_path: str, vocab_path: str = "data/FP_GrandStaff_BeKernw2i.npy") -> Dict:
    """
    Complete demo pipeline from image to visualization.
    
    Args:
        img_path: Path to input image
        ckpt_path: Path to model checkpoint
        vocab_path: Path to vocabulary file
        
    Returns:
        Dictionary with all pipeline outputs
    """
    print("=== MusicTrOCR Demo Pipeline ===\n")
    
    # Step 1: Load model and vocabulary
    print("1. Loading model and vocabulary...")
    model, vocab_dict, id_to_token = load_model_and_vocab(ckpt_path, vocab_path)
    
    # Step 2: Preprocess image
    print("\n2. Preprocessing image...")
    image_tensor = preprocess_image(img_path)
    
    # Step 3: Run inference
    print("\n3. Running inference...")
    predictions = run_inference(model, image_tensor, vocab_dict)
    
    # Step 4: Decode predictions
    print("\n4. Decoding predictions...")
    bekern_str = decode_bekern_prediction(predictions, id_to_token, model)
    
    # Step 5: Convert to kern format
    print("\n5. Converting to kern format...")
    kern_str = bekern_to_kern(bekern_str)
    
    # Step 6: Render music score
    print("\n6. Rendering music score...")
    rendered_score = render_music_score(kern_str, "demos/output_score.png")
    
    # Step 7: Visualize results
    print("\n7. Displaying results...")
    visualize_results(img_path, rendered_score, bekern_str)
    
    return {
        'model': model,
        'predictions': predictions,
        'bekern_str': bekern_str,
        'kern_str': kern_str,
        'rendered_score': rendered_score,
        'vocab_dict': vocab_dict,
        'id_to_token': id_to_token
    }