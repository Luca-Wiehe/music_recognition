# MusicTrOCR Demo

This folder contains demonstration materials for the MusicTrOCR (Luca Model) optical music recognition system.

## Files

- `luca_model_demo.ipynb` - Main demonstration notebook
- `utils.py` - Utility functions for model loading, preprocessing, and visualization
- `demo.png` - Input sheet music image (to be provided)
- `output_score.png` - Generated music score visualization (created during demo)
- `README.md` - This file

## Setup

### 1. Prerequisites

Ensure you have the main project dependencies installed:
```bash
conda activate music_recognition
# or
pip install torch torchvision matplotlib pillow numpy opencv-python
```

### 2. Optional: Verovio for Music Rendering

For music score visualization, install Verovio:
```bash
pip install verovio cairosvg
```

Without Verovio, the demo will still work but won't generate visual music scores.

### 3. Demo Image

Place a sheet music image at `demos/demo.png`. The image should be:
- Clear sheet music notation
- PNG, JPG, or similar format
- Reasonable resolution (will be resized to height 128px)

### 4. Model Checkpoint

Ensure your trained model checkpoint is available at:
```
networks/checkpoints/luca_model/best_model.pth
```

Or update the path in the notebook configuration cell.

## Running the Demo

1. Open the notebook:
```bash
jupyter notebook luca_model_demo.ipynb
```

2. Run cells sequentially to see the complete pipeline:
   - Model loading
   - Image preprocessing
   - Neural inference
   - BeKern decoding
   - Music score rendering
   - Results visualization

3. Alternatively, uncomment the complete pipeline cell for one-step execution.

## Expected Output

The demo will show:
1. Original input sheet music image
2. Predicted BeKern notation (symbolic representation)
3. Rendered music score (if Verovio available)
4. Side-by-side comparison of input and output

## Troubleshooting

**Model loading errors:**
- Check checkpoint path exists
- Verify model was trained with same architecture parameters

**Image loading errors:**
- Ensure demo.png exists and is readable
- Try different image formats (JPG, PNG)

**Verovio rendering errors:**
- Install verovio and cairosvg packages
- Check that BeKern notation is valid format

**Memory issues:**
- Reduce max_length parameter in inference
- Use CPU device if GPU memory insufficient

## Architecture Details

**MusicTrOCR Model:**
- Vision Encoder: ConvNeXt-Tiny (pre-trained)
- Text Decoder: 6-layer Transformer with cross-attention
- Vocabulary: ~300 BeKern music symbols
- Output: Autoregressive sequence generation

**Processing Pipeline:**
1. Image → Tensor (128px height, aspect ratio preserved)
2. Vision Encoder → Spatial features
3. Transformer Decoder → Token sequence
4. Token Decoding → BeKern notation
5. Format Conversion → Kern notation
6. Verovio Rendering → Visual score