# Model Configuration Files

This directory contains YAML configuration files for training different OMR models.

## Usage

Train any model using:
```bash
python train.py --config configs/<model_name>.yaml
```

Resume training from a checkpoint:
```bash
python train.py --config configs/<model_name>.yaml --resume checkpoints/<model_name>/best_checkpoint.pt
```

## Available Models

### Music-TrOCR (`luca_model.yaml`)
- **Architecture**: Vision Encoder + Autoregressive Transformer Decoder
- **Vision Backbone**: ConvNeXt-Tiny (~28M parameters)
- **Decoder**: 6-layer Transformer with cross-attention
- **Total Parameters**: ~300-400M (well under 1B limit)
- **Key Features**:
  - Region-focused attention via cross-attention
  - Token-by-token autoregressive generation
  - Pre-trained HuggingFace backbone
  - Teacher forcing during training

### Monophonic CRNN (`monophonic_model.yaml`)  
- **Architecture**: CNN + Bidirectional LSTM + CTC Loss
- **Key Features**:
  - Convolutional feature extraction
  - Recurrent sequence modeling
  - CTC loss for alignment-free training
  - Baseline model for comparison

## Configuration Parameters

### Model Parameters
- `vision_model_name`: HuggingFace model for vision encoder
- `d_model`: Hidden dimension of transformer
- `n_heads`: Number of attention heads
- `n_decoder_layers`: Number of transformer layers

### Training Parameters
- `batch_size`: Training batch size
- `learning_rate`: Initial learning rate
- `epochs`: Maximum training epochs
- `early_stop_patience`: Early stopping patience

### Data Parameters
- `dataset_path`: Path to Primus dataset
- `vocabulary_path`: Path to vocabulary file
- `split_ratio`: Train/validation/test split ratios

### Logging Parameters
- `use_wandb`: Enable Weights & Biases logging
- `project`: W&B project name
- `run_name`: W&B run name
- `tags`: W&B tags for organization