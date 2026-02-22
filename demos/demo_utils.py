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
    import cairosvg
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
    
    # Get special token IDs from BeKern vocabulary (verified from .npy file)
    pad_token_id = vocab_dict.get('<pad>')  # Should be 0
    bos_token_id = vocab_dict.get('<bos>')  # Should be 169
    eos_token_id = vocab_dict.get('<eos>')  # Should be 72
    
    # Verify we found all special tokens
    if pad_token_id is None or bos_token_id is None or eos_token_id is None:
        raise ValueError(f"Missing special tokens in vocabulary: PAD={pad_token_id}, BOS={bos_token_id}, EOS={eos_token_id}")
        
    # Verify expected values
    expected = {'<pad>': 0, '<bos>': 169, '<eos>': 72}
    actual = {'<pad>': pad_token_id, '<bos>': bos_token_id, '<eos>': eos_token_id}
    if actual != expected:
        print(f"WARNING: Special token IDs differ from expected:")
        for token in expected:
            if actual[token] != expected[token]:
                print(f"  {token}: expected {expected[token]}, got {actual[token]}")
    
    print(f"Special token IDs: PAD={pad_token_id}, BOS={bos_token_id}, EOS={eos_token_id}")
    
    # Initialize model with BeKern vocabulary token IDs
    model = MusicTrOCR(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        vision_model_name="facebook/convnext-tiny-224",
        d_model=512,
        n_heads=8,
        n_decoder_layers=6,
        d_ff=2048,
        max_seq_len=4353,
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
    Now simplified since model uses BeKern vocabulary directly.
    
    Args:
        token_ids: Predicted token IDs from model (BeKern vocabulary IDs)
        id_to_token: BeKern token ID to string mapping
        model: Model instance (for special token IDs)
        
    Returns:
        Decoded bekern string
    """
    print(f"\n=== DECODING BEKERN TOKENS ===")
    raw_tokens = token_ids.squeeze().tolist()
    print(f"Model output tokens: {raw_tokens}")
    print(f"Model special tokens: BOS={model.START_TOKEN_ID}, EOS={model.END_TOKEN_ID}, PAD={model.PAD_TOKEN_ID}")
    
    # No conversion needed - model already uses BeKern vocabulary
    # Sequences from generation include BOS at start, so we need to handle that
    bekern_tokens = []
    
    for i, token_id in enumerate(raw_tokens):
        print(f"  Step {i}: token_id={token_id}", end="")
        
        # Skip special tokens (BOS at start, EOS/PAD for stopping)
        if token_id == model.START_TOKEN_ID:
            print(f" -> BOS (skipping)")
            continue
        elif token_id == model.END_TOKEN_ID:
            print(f" -> EOS (stopping)")
            break
        elif token_id == model.PAD_TOKEN_ID:
            print(f" -> PAD (skipping)")
            continue
        elif token_id in id_to_token:
            token_str = id_to_token[token_id]
            print(f" -> '{token_str}'")
            # Skip dataset-level special tokens (these are structural, not music content)
            if token_str not in ['<bos>', '<eos>', '<pad>']:
                bekern_tokens.append(token_str)
        else:
            print(f" -> UNKNOWN_TOKEN")
            print(f"    Warning: Unknown token ID {token_id}")
    
    # Join tokens with spaces
    bekern_str = ' '.join(bekern_tokens)
    
    print(f"\nDecoding summary:")
    print(f"  Input tokens: {len(raw_tokens)}")
    print(f"  Valid music tokens: {len(bekern_tokens)}")
    print(f"  BeKern string: '{bekern_str[:100]}{'...' if len(bekern_str) > 100 else ''}'")
    print(f"=== END DECODING ===\n")
    
    return bekern_str


def bekern_to_kern(bekern_str: str) -> str:
    """
    Convert bekern/ekern format to kern format using SMT's exact conversion approach.
    This matches the conversion used in third_party/SMT/SynthGenerator.py
    
    Args:
        bekern_str: Bekern/ekern notation string 
        
    Returns:
        Standard kern format string for Verovio rendering
    """
    # Apply SMT's exact conversion approach from SynthGenerator.py lines 175-176, 223-224
    # Convert structural tokens back to their actual characters
    kern_str = bekern_str.replace('<b>', '\n')
    kern_str = kern_str.replace('<s>', ' ')
    kern_str = kern_str.replace('<t>', '\t')
    
    # Remove bekern-specific symbols (following SMT's approach)
    kern_str = kern_str.replace('@', '')   # Remove @ symbols
    kern_str = kern_str.replace('·', '')   # Remove middle dots (key fix!)
    
    # Ensure we have proper kern header if input has **ekern
    if '**ekern' in kern_str:
        kern_str = kern_str.replace('**ekern', '**kern')
    elif not kern_str.startswith('**kern'):
        # Add kern header if missing
        kern_str = '**kern\t**kern\n' + kern_str
    
    return kern_str


def render_with_transposition(kern_str: str, transpose_semitones: int = 0) -> Optional[np.ndarray]:
    """
    Render kern notation with optional transposition using Verovio.
    
    Args:
        kern_str: Kern format notation string
        transpose_semitones: Number of semitones to transpose (positive = up, negative = down)
        
    Returns:
        Rendered image as numpy array, or None if Verovio unavailable
    """
    if not VEROVIO_AVAILABLE:
        print("Verovio not available. Cannot render music score.")
        return None
    
    try:
        tk = verovio.toolkit()
        
        # Set options with transposition if specified
        options = {
            "pageWidth": 2100,
            "scale": 40,
            "adjustPageHeight": True,
            "footer": "none",
            "header": "none"
        }
        
        if transpose_semitones != 0:
            options["transpose"] = str(transpose_semitones)
        
        tk.setOptions(options)
        tk.loadData(kern_str)
        svg = tk.renderToSVG()
        
        if not svg or len(svg.strip()) == 0:
            print("Error: Verovio returned empty SVG")
            return None
        
        # Convert SVG to PNG
        png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'), background_color='white')
        image_array = cv2.imdecode(np.frombuffer(png_data, np.uint8), -1)
        
        if image_array is None:
            print("Error: Failed to decode PNG data")
            return None
            
        return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"Error rendering music score: {e}")
        return None


def display_comparison(original_img, transposed_img, transpose_amount):
    """
    Display side-by-side comparison of original and transposed notation.
    
    Args:
        original_img: Original notation image array
        transposed_img: Transposed notation image array  
        transpose_amount: Number of semitones transposed
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    if original_img is not None:
        axes[0].imshow(original_img)
        axes[0].set_title("Original Prediction", fontsize=14)
    else:
        axes[0].text(0.5, 0.5, 'Rendering\nFailed', ha='center', va='center', 
                    transform=axes[0].transAxes)
        axes[0].set_title("Original Prediction (Failed)", fontsize=14)
    
    if transposed_img is not None:
        axes[1].imshow(transposed_img)
        if transpose_amount == 0:
            axes[1].set_title("Same Prediction (No Transposition)", fontsize=14)
        else:
            direction = "Up" if transpose_amount > 0 else "Down"
            axes[1].set_title(f"Transposed {abs(transpose_amount)} Semitones {direction}", fontsize=14)
    else:
        axes[1].text(0.5, 0.5, 'Rendering\nFailed', ha='center', va='center', 
                    transform=axes[1].transAxes)
        if transpose_amount == 0:
            axes[1].set_title("Same Prediction (Failed)", fontsize=14)
        else:
            axes[1].set_title(f"Transposed {abs(transpose_amount)} Semitones (Failed)", fontsize=14)
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def display_image_and_prediction(original_img_path: str, rendered_score: Optional[np.ndarray]):
    """
    Display original input image and predicted score side by side.
    
    Args:
        original_img_path: Path to original input image
        rendered_score: Rendered score image array (or None)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Load and display original image
    original_img = cv2.imread(original_img_path)
    if original_img is not None:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_img)
        axes[0].set_title("Original Input Image", fontsize=14)
    else:
        axes[0].text(0.5, 0.5, 'Image\nLoad Failed', ha='center', va='center', 
                    transform=axes[0].transAxes)
        axes[0].set_title("Original Input Image (Failed)", fontsize=14)
    
    # Display rendered score or placeholder
    if rendered_score is not None:
        axes[1].imshow(rendered_score)
        axes[1].set_title("Predicted Music Score", fontsize=14)
    else:
        axes[1].text(0.5, 0.5, "Music rendering\nnot available\n(Verovio required)", 
                    ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
        axes[1].set_title("Predicted Music Score (Unavailable)", fontsize=14)
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def run_inference(model: MusicTrOCR, image_tensor: torch.Tensor, vocab_dict: Dict, max_length: int = 512) -> torch.Tensor:
    """
    Run inference on preprocessed image using BeKern vocabulary directly.
    
    Args:
        model: Trained MusicTrOCR model  
        image_tensor: Preprocessed image tensor
        vocab_dict: BeKern vocabulary dictionary (for reference)
        max_length: Maximum generation length
        
    Returns:
        Generated token sequence
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    print(f"Running inference with BeKern vocabulary...")
    print(f"Model special tokens: PAD={model.PAD_TOKEN_ID}, BOS={model.START_TOKEN_ID}, EOS={model.END_TOKEN_ID}")
    
    with torch.no_grad():
        # Use the model's built-in generation method
        predictions = model.generate(
            image_tensor,
            max_length=max_length,
            temperature=1.0,
            do_sample=False  # Greedy decoding
        )
    
    print(f"Generated sequence length: {predictions.shape[1]} (including BOS token)")
    return predictions


def demo_pipeline(img_path: str, ckpt_path: str, vocab_path: str = "data/FP_GrandStaff_BeKernw2i.npy", transpose_semitones: int = 0) -> Dict:
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
    original_score = render_with_transposition(kern_str, 0)
    transposed_score = render_with_transposition(kern_str, transpose_semitones) if transpose_semitones != 0 else original_score
    
    # Step 7: Visualize results
    print("\n7. Displaying results...")
    display_image_and_prediction(img_path, original_score)
    
    if transpose_semitones != 0:
        print(f"\n8. Displaying transposition comparison ({transpose_semitones} semitones)...")
        display_comparison(original_score, transposed_score, transpose_semitones)
    
    return {
        'model': model,
        'predictions': predictions,
        'bekern_str': bekern_str,
        'kern_str': kern_str,
        'original_score': original_score,
        'transposed_score': transposed_score,
        'vocab_dict': vocab_dict,
        'id_to_token': id_to_token
    }