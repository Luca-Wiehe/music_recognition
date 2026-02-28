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

#### Training

MusicTrOCR can be trained entirely from CLI flags — no config file needed:

```bash
# Full fine-tuning with DeiT-Small encoder:
python -m src.train --encoder facebook/deit-small-patch16-224 \
    --wandb-project music-recognition

# LoRA fine-tuning (rank 8):
python -m src.train --encoder facebook/deit-small-patch16-224 \
    --encoder-mode lora --lora-rank 8

# Pitch-only ablation:
python -m src.train --encoder facebook/deit-small-patch16-224 --strip-non-pitch
```

For complex setups (distillation, monophonic model), use a YAML config:
```bash
python -m src.train --config configs/distillation.yaml
```

Run `python -m src.train --help` for all available options.

#### Choosing the Vision Encoder

All encoders were benchmarked using the same MusicTrOCR decoder (6-layer Transformer, d_model=512) with full finetuning on the PDMX-Synth dataset for 10 epochs.

| Encoder | Pretrained Model | Encoder Params | Total Params | Best Val Loss |
|---|---|---|---|---|
| DeiT-Small | `facebook/deit-small-patch16-224` | 22M | 58.9M | **0.3205** |
| ConvNeXt-Tiny | `facebook/convnext-tiny-224` | 28M | 65.1M | 0.3371 |
| Swin-Tiny | `microsoft/swin-tiny-patch4-window7-224` | 28M | 64.8M | 0.3555 |
| ViT-Small | `WinKawaks/vit-small-patch16-224` | 22M | 58.9M | 0.3922 |
| ResNet-50 | `microsoft/resnet-50` | 25M | 61.5M | 0.4529 |
| MobileViT-Small | `apple/mobilevit-small` | 6M | 42.2M | 0.4857 |
| EfficientNet-B0 | `google/efficientnet-b0` | 5M | 41.6M | 0.5119 |

<details>
<summary>Reproduction commands</summary>

```bash
# DeiT-Small (best)
python -m src.train --encoder facebook/deit-small-patch16-224 \
    --wandb-project music-recognition --wandb-run-name backbone-deit-small

# ConvNeXt-Tiny
python -m src.train --encoder facebook/convnext-tiny-224 \
    --wandb-project music-recognition --wandb-run-name backbone-convnext-tiny

# Swin-Tiny
python -m src.train --encoder microsoft/swin-tiny-patch4-window7-224 \
    --wandb-project music-recognition --wandb-run-name backbone-swin-tiny

# ViT-Small
python -m src.train --encoder WinKawaks/vit-small-patch16-224 \
    --wandb-project music-recognition --wandb-run-name backbone-vit-small

# ResNet-50
python -m src.train --encoder microsoft/resnet-50 \
    --wandb-project music-recognition --wandb-run-name backbone-resnet50

# MobileViT-Small
python -m src.train --encoder apple/mobilevit-small \
    --wandb-project music-recognition --wandb-run-name backbone-mobilevit-small

# EfficientNet-B0
python -m src.train --encoder google/efficientnet-b0 \
    --wandb-project music-recognition --wandb-run-name backbone-efficientnet-b0
```

</details>
