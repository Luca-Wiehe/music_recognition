# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based optical music recognition (OMR) system that converts images of musical scores into MIDI files. The main architecture implemented is a Convolutional Recurrent Neural Network (CRNN) based on the Camera Primus paper, with plans for Transformer architectures (TrOMR).

## Environment Setup

Use conda environment:
```bash
conda env create -f environment.yaml
conda activate music_recognition
```

Or pip install dependencies:
```bash
pip install torch torchvision ipykernel prettytable matplotlib transformers
```

## Dataset Setup

Download and extract the Primus dataset:
```bash
cd data/primus
wget https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz
tar -xzvf primusCalvoRizoAppliedSciences2018.tgz
```

## Development Workflow

The main development workflow uses Jupyter notebooks:
- `data_loader.ipynb` contains the complete training pipeline
- Training is performed using the notebook interface with interactive monitoring
- Model checkpoints are saved to `networks/checkpoints/`

Training workflow:
1. Load dataset using `PrimusDataset` from `data/primus_dataset.py`
2. Apply train/val/test split using `split_data()`
3. Initialize `MonophonicModel` from `networks/monophonic_nn.py`
4. Train using `train_model()` function with CTC loss
5. Evaluate model performance and visualize results

## Architecture

### Data Pipeline
- `data/primus_dataset.py`: Custom dataset class handling Primus format
  - Loads `.png` images and `.semantic` label files
  - Applies normalization and resizing (fixed height 128px)
  - Converts semantic labels to integer sequences using vocabulary mapping
- `data/semantic_labels.txt`: Vocabulary mapping for music symbols

### Models
- `networks/monophonic_nn.py`: Main CRNN implementation
  - Convolutional feature extraction (4 conv layers with pooling)
  - Bidirectional LSTM layers (2 layers, 256 hidden units each)
  - CTC loss for sequence prediction without alignment
  - Supports variable-length sequences
- `networks/jannis_model.py`: Placeholder for future model implementation
- `networks/luca_model.py`: Placeholder for future model implementation  
- `networks/niklas_model.py`: Placeholder for future model implementation

### Training Details
- Uses CTC (Connectionist Temporal Classification) loss with blank token at index 0
- Early stopping with patience mechanism
- Learning rate scheduling with ReduceLROnPlateau
- Model checkpoints saved every 100 epochs
- Gradient clipping (norm=1.0) for stability

## Key Functions

### Dataset
- `PrimusDataset.__getitem__()`: Returns normalized image tensor and label sequence
- `collate_fn()`: Pads batches to handle variable image widths and sequence lengths
- `split_data()`: 60/20/20 train/val/test split with fixed seed

### Training
- `MonophonicModel.training_step()`: Single training iteration with CTC loss
- `MonophonicModel.validation_step()`: Validation without gradient computation
- `train_model()`: Complete training loop with early stopping and checkpointing

### Utils
- `utils.create_tqdm_bar()`: Progress bar creation for training loops