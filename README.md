# Optical Music Recognition

This repository is a PyTorch implementation of several optical music recognition architectures. The goal is to take an image of a music score as input and produce a MIDI file as output. We want to provide practical guidance for anyone planning to train or run OMR systems, ablating several architecture decisions. In particular, we look at settings with limited compute availability (< 32GB VRAM).  

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

### Training

### Overview
An OMR system has two core components (analogous to eyes and brain). A **vision encoder** extracts visual features from a score image, and a **decoder** translates those features into a sequence of music symbols. The encoder choice matters because these models are typically pretrained on natural images (ImageNet) and then finetuned on sheet music, and different architectures transfer differently to the structured, horizontal layout of musical notation.

#### Choosing the Vision Encoder

We benchmark seven pretrained encoders under a limited compute budget (<32 GB VRAM), keeping the decoder fixed (6-layer Transformer, d_model=512) and fully finetuning each encoder on the PDMX-Synth dataset for 10 epochs.

| Encoder | Pretrained Model | Encoder Params | Total Params | Best Val Loss |
|---|---|---|---|---|
| DeiT-Small | `facebook/deit-small-patch16-224` | 22M | 58.9M | **0.3205** |
| ConvNeXt-Tiny | `facebook/convnext-tiny-224` | 28M | 65.1M | 0.3371 |
| Swin-Tiny | `microsoft/swin-tiny-patch4-window7-224` | 28M | 64.8M | 0.3555 |
| ViT-Small | `WinKawaks/vit-small-patch16-224` | 22M | 58.9M | 0.3922 |
| ResNet-50 | `microsoft/resnet-50` | 25M | 61.5M | 0.4529 |
| MobileViT-Small | `apple/mobilevit-small` | 6M | 42.2M | 0.4857 |
| EfficientNet-B0 | `google/efficientnet-b0` | 5M | 41.6M | 0.5119 |

Patch-based vision transformers (DeiT, Swin) consistently outperform CNNs, with DeiT-Small achieving the best loss while being among the smallest encoders.

<details>
<summary>Reproduction commands</summary>

```bash
# DeiT-Small (best)
python -m src.train \
    --encoder facebook/deit-small-patch16-224 \
    --wandb-project music-recognition \
    --wandb-run-name backbone-deit-small

# ConvNeXt-Tiny
python -m src.train \
    --encoder facebook/convnext-tiny-224 \
    --wandb-project music-recognition \
    --wandb-run-name backbone-convnext-tiny

# Swin-Tiny
python -m src.train \
    --encoder microsoft/swin-tiny-patch4-window7-224 \
    --wandb-project music-recognition \
    --wandb-run-name backbone-swin-tiny

# ViT-Small
python -m src.train \
    --encoder WinKawaks/vit-small-patch16-224 \
    --wandb-project music-recognition \
    --wandb-run-name backbone-vit-small

# ResNet-50
python -m src.train \
    --encoder microsoft/resnet-50 \
    --wandb-project music-recognition \
    --wandb-run-name backbone-resnet50

# MobileViT-Small
python -m src.train \
    --encoder apple/mobilevit-small \
    --wandb-project music-recognition \
    --wandb-run-name backbone-mobilevit-small

# EfficientNet-B0
python -m src.train \
    --encoder google/efficientnet-b0 \
    --wandb-project music-recognition \
    --wandb-run-name backbone-efficientnet-b0
```

</details>

#### Choosing the Decoder

With the encoder fixed to a lightweight CNN (SharedEncoder), we compare four decoder architectures under CTC loss, matching each to ~6.3M decoder parameters (1x) and ~25.2M decoder parameters (4x). All runs use lr=3e-4, AdamW, 3% linear warmup, and 10 epochs on PDMX-Synth.

**1x scale (~6M decoder params)**

| Decoder | Total Params | Decoder Params | Best Val Loss |
|---|---|---|---|
| **GRU** | 9.1M | 6.2M | **0.3021** |
| Transformer | 8.9M | 6.4M | 0.3129 |
| LSTM | 8.8M | 6.3M | 0.3206 |
| RNN | 12.1M | 6.3M | 0.3304 |

**4x scale (~25M decoder params)**

| Decoder | Total Params | Decoder Params | Best Val Loss |
|---|---|---|---|
| **GRU** | 32.0M | 25.1M | **0.3008** |
| LSTM | 31.0M | 25.1M | 0.3093 |
| RNN | 38.2M | 25.2M | 0.3340 |
| Transformer | 28.6M | 25.1M | 0.4476 |

Scaling the decoder from 6M to 25M parameters does not meaningfully improve results. GRU gains only 0.0013 in val loss, LSTM gains 0.0113, and RNN slightly degrades. The Transformer collapses entirely at 4x — its training loss plateaus at ~0.47 from epoch 1 (vs. 0.39→0.32 at 1x), indicating an optimization failure rather than overfitting (train and val losses are nearly identical). This is likely caused by the fixed lr=3e-4 being too aggressive for the larger Transformer (6 layers, d_model=768). Overall, the lightweight GRU at 6M parameters is the best choice — additional decoder capacity is better spent elsewhere in the pipeline.

<details>
<summary>Reproduction commands</summary>

```bash
# 1x scale (~6M decoder params)
bash launch_decoder_ablation.sh

# 4x scale (~25M decoder params)
# Edit launch_decoder_scaling.sh to run only 4x, or run directly:
for decoder in lstm rnn gru transformer; do
    python train_decoder_ablation.py \
        --decoder $decoder --scale 4x \
        --wandb-project music-recognition
done
```

</details>
