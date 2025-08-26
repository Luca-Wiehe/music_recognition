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
        max_seq_len=512,
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
    # Convert model tokens back to dataset vocabulary
    dataset_tokens = model.decode_model_tokens_to_dataset(token_ids)
    
    # Convert to string tokens
    bekern_tokens = []
    for token_id in dataset_tokens.squeeze().tolist():
        if token_id in id_to_token:
            token = id_to_token[token_id]
            # Skip special tokens for bekern output
            if token not in ['<pad>', '<bos>', '<eos>']:
                bekern_tokens.append(token)
        else:
            print(f"Warning: Unknown token ID {token_id}")
    
    # Join tokens with spaces
    bekern_str = ' '.join(bekern_tokens)
    
    print(f"Decoded {len(bekern_tokens)} tokens")
    print(f"Bekern preview: {bekern_str[:100]}...")
    
    return bekern_str


def bekern_to_kern(bekern_str: str) -> str:
    """
    Convert bekern format to kern format for Verovio rendering.
    
    Args:
        bekern_str: Bekern notation string
        
    Returns:
        Kern format string
    """
    # Replace bekern special tokens with kern equivalents
    kern_str = bekern_str.replace('<b>', '\n')
    kern_str = kern_str.replace('<s>', ' ')  
    kern_str = kern_str.replace('<t>', '\t')
    
    # Clean up extra spaces and newlines
    lines = [line.strip() for line in kern_str.split('\n') if line.strip()]
    kern_str = '\n'.join(lines)
    
    # Add kern header if not present
    if not kern_str.startswith('**kern'):
        kern_str = '**kern\t**kern\n' + kern_str
    
    # Add ending marker
    if not kern_str.endswith('*-'):
        kern_str += '\n*-\t*-'
    
    return kern_str


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
        
        # Load data and render
        tk.loadData(kern_str)
        svg = tk.renderToSVG()
        
        # Convert SVG to PNG using cairosvg
        try:
            from cairosvg import svg2png
            png_data = svg2png(bytestring=svg.encode('utf-8'), background_color='white')
            
            # Convert to numpy array
            image_array = cv2.imdecode(np.frombuffer(png_data, np.uint8), -1)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Save if path provided
            if output_path:
                cv2.imwrite(output_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
                print(f"Rendered score saved to: {output_path}")
            
            return image_array
            
        except ImportError:
            print("cairosvg not available. Cannot convert SVG to PNG.")
            return None
            
    except Exception as e:
        print(f"Error rendering music score: {e}")
        return None


def run_inference(model: MusicTrOCR, image_tensor: torch.Tensor, max_length: int = 512) -> torch.Tensor:
    """
    Run inference on preprocessed image.
    
    Args:
        model: Trained MusicTrOCR model
        image_tensor: Preprocessed image tensor
        max_length: Maximum generation length
        
    Returns:
        Generated token sequence
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    print("Running inference...")
    with torch.no_grad():
        # Generate predictions
        predictions = model.generate(
            image_tensor,
            max_length=max_length,
            temperature=1.0,
            do_sample=False  # Greedy decoding
        )
    
    print(f"Generated sequence length: {predictions.shape[1]}")
    return predictions


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
    predictions = run_inference(model, image_tensor)
    
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