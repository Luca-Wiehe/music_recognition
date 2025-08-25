# Optical Music Recognition

This repository is a PyTorch implementation of several optical music recognition techniques. The goal is to take an image of a music score as input and produce a MIDI file as output. Currently, the main implemented architecture is a CRNN. A Transformer architecture will follow soon.

## Preview 

## Setup
```bash
conda env create -f environment.yaml
conda activate music_recognition
```

### Data Download and Preprocessing
In `scripts/download.py`, we provide a script to download all relevant datasets for Optical Music Recognition. The script handles two types of datasets:

1. **Camera Primus Dataset**: Real sheet music images with semantic labels (for training)
2. **SMT HuggingFace Datasets**: Bekern sequences (for synthetic data generation)

#### Quick Start
```bash
python scripts/download.py --primus

python scripts/download.py --smt-all
```

<details>
<summary>Available Command Line Arguments</summary>

| Argument | Description | Source |
|----------|-------------|---------|
| `--primus` | Download Camera Primus dataset | https://grfia.dlsi.ua.es/primus/ |
| `--smt <dataset>` | Download specific SMT dataset(s) | HuggingFace (antoniorv6/*) |
| `--smt-all` | Download all SMT datasets | HuggingFace (antoniorv6/*) |
| `--list-smt` | List available SMT datasets | - |
| `--output_dir` | Base output directory | Default: `data/datasets` |
| `--splits` | Dataset splits to download | Default: train, validation, test |

**Available SMT Datasets:**
- `grandstaff`: GrandStaff system-level (original format)
- `grandstaff-ekern`: GrandStaff in ekern format  
- `grandstaff-bekern`: GrandStaff in bekern format
- `mozarteum`: Mozarteum dataset
- `polish-scores`: Polish Scores dataset
- `string-quartets`: String Quartets dataset

</details>

The data will be stored in the `/data/datasets/` folder organized as:
- `/data/datasets/primus/`: Camera Primus dataset (images + semantic labels)
- `/data/datasets/smt_datasets/`: SMT bekern datasets (for synthetic generation)


### Repository Structure
```
├── data/
│   ├── datasets/                    # Downloaded datasets
│   │   ├── primus/                  # Camera Primus dataset (images + semantic labels)
│   │   ├── smt_datasets/            # SMT bekern datasets (for synthetic generation)
│   │   └── synthetic/               # Generated synthetic data
│   └── utils/                       # Data processing utilities
│       ├── format_converter.py      # Primus to bekern format conversion
│       └── synthetic_generator.py   # Synthetic image generation using Verovio
├── networks/                        # Neural networks for OMR tasks
├── scripts/                         # Utility scripts
│   ├── download.py                  # Dataset download script
│   └── generate_synthetic_data.py   # Synthetic data generation script
└── ...
```

## Implemented Networks
### CRNN
The first implemented neural network is a CRNN that I reimplemented from the Camera Primus Paper. It uses the a Convolutional Recurrent Neural Network (CRNN) architecture which is characterized by a set of convolutional layers followed by several BiLSTMs and linear layers. Before each activation, batch normalization is performed to make sure that gradients are in an active regime. 

The implementation of this stage is almost completed. However, the training in the original paper had 64,000 epochs which is infeasible in terms of available compute power at this point.

### TrOMR
The second architecture is a transformer architecture reimplemented from the TrOMR Paper. It uses Transfer Learning with a pretrained Vision Transformer to predict sequences of music symbols.
