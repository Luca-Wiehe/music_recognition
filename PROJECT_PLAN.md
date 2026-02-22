# Practical OMR with Vision Backbones: Project Plan

## Project Vision

A comprehensive educational blog post that teaches readers how Optical Music Recognition works from the ground up. Rather than chasing state-of-the-art benchmarks, the goal is to build intuition: *why* do certain architectures work for music, *how* do design choices affect real-world performance, and *what* trade-offs matter when deploying OMR on limited hardware?

The post uses a hands-on backbone ablation study as its central teaching device, comparing 7 vision encoders on the PDMX-Synth dataset to make architectural trade-offs concrete and tangible.

**Format**: Comprehensive blog post on personal website
**Audience**: ML practitioners and students curious about OMR, multimodal sequence models, or vision-language architectures
**Tone**: Educational and opinionated — explain the "why" behind every decision, not just the "what"

---

## Blog Post Structure

### Section 1: What Makes Music Hard to Read (for Machines)?
- What is Optical Music Recognition and why it remains an open problem
- Key differences from text OCR that make OMR uniquely challenging:
  - **2D spatial structure**: Notes stack vertically (chords), flow horizontally (melody), and interact diagonally (beams, slurs) — unlike left-to-right text
  - **Dense symbol layout**: A single measure packs multiple overlapping symbols (noteheads, stems, flags, accidentals, dynamics) into a small area
  - **Long output sequences**: A single page of music can produce thousands of tokens
  - **Context dependence**: The meaning of a symbol depends on its spatial neighbors (e.g., a dot next to a note vs. a staccato dot above it)
- Brief history: rule-based systems -> CNN classifiers -> end-to-end seq2seq (where we are now)
- The PDMX-Synth dataset: 216K synthetic image-ABC pairs, why synthetic data is useful, limitations vs. real-world scans

### Section 2: How OMR Models Work (Architecture Walkthrough)
- **The encoder-decoder paradigm**: Explain visually with diagrams
  - Vision encoder: "looks at the image and extracts spatial features"
  - Cross-attention: "the decoder asks the encoder where to look for the next symbol" — this is the key insight that makes autoregressive decoding powerful for OMR
  - Autoregressive decoder: "generates one token at a time, conditioning on both the image and everything predicted so far"
- **Why not CTC?** Concrete example showing how CTC struggles with 2D layout and variable-length alignment, while cross-attention handles it naturally
- **Why not just use a Vision-Language Model (VLM)?** Cost and accessibility — LEGATO uses Llama-3.2-11B-Vision (80GB VRAM). We explore what smaller backbones can achieve.
- **Our architecture in detail**: Walk through MusicTrOCR step by step
  - Image preprocessing: height normalization, aspect-ratio preservation, RGB conversion
  - Feature extraction: how CNNs vs Transformers encode spatial information differently
  - Feature projection: bridging backbone output dimensions to the decoder
  - Token prediction: embedding, positional encoding, masked self-attention, cross-attention, output projection
- Code snippets showing key components (keep it readable, not a full dump)

### Section 3: The Experiment — Swapping Vision Backbones
- **Experimental design**: The encoder is the independent variable; everything else is controlled
  - Shared decoder: 6-layer Transformer, d_model=512, 8 heads, SwiGLU FFN, max_seq_len=2048
  - Shared training recipe: AdamW (lr=3e-4), linear warmup (3%), AMP, effective batch=32
  - Same dataset, tokenizer, and evaluation protocol
- **The 7 backbones and why we chose them**:
  - **CNN family**: ResNet-50 (the workhorse), EfficientNet-B0 (mobile-optimized), ConvNeXt-Tiny (modern CNN that borrows from Transformers)
  - **Transformer family**: ViT-Small (vanilla vision transformer), DeiT-Small (data-efficient variant with distillation pretraining), Swin-Tiny (hierarchical windows)
  - **Hybrid**: MobileViT-Small (combines depthwise convolutions with transformer blocks)
- Explain what each architecture does differently and what hypothesis it tests for OMR:
  - Do CNNs' inductive biases (locality, translation invariance) help with dense music symbols?
  - Do Transformers' global attention capture long-range spatial dependencies better?
  - Are "efficient" architectures actually efficient on GPUs? (Spoiler: no — explain why depthwise separable convolutions are memory-bound on GPUs despite fewer FLOPs)

### Section 4: Results and Analysis
- **Training dynamics**: Loss curves for all 7 backbones (from WandB), discuss convergence patterns
- **Results table**: Validation loss, Symbol Error Rate (SER), throughput (samples/sec), peak VRAM, total params
- **Key findings** (present as lessons, not just numbers):
  - Which backbone family works best for OMR, and why?
  - Parameter count vs accuracy: is bigger always better?
  - The GPU efficiency paradox: mobile-optimized models are the slowest to train
  - Swin-Tiny's quirks: window-based attention and the padding problem we encountered
- **Failure analysis**: Example predictions showing where models struggle (complex polyphony, dense passages, rare symbols)

### Section 5: Making It Practical — Efficiency Techniques
- **Motivation**: The best backbone from Phase 1 might not fit on your GPU. How do we close the gap?
- **LoRA / QLoRA**: Low-rank adaptation explained intuitively
  - What it does: freeze the backbone, train tiny adapter matrices instead
  - Compare: frozen backbone vs full fine-tune vs LoRA (rank 4/8/16) on 2-3 selected backbones
  - QLoRA: 4-bit quantized backbone + LoRA for extreme VRAM savings
  - When does LoRA match full fine-tuning? When does it fall short?
- **Knowledge Distillation**: Teaching a small model using a large one
  - Teacher: best-performing backbone from Phase 1
  - Student: MobileViT-Small or EfficientNet-B0
  - Does distillation close the accuracy gap for on-device deployment?
- **Quantization**: Post-training INT8 quantization — free lunch or accuracy trap?
- **Practical takeaway**: Decision tree for "given my GPU, what should I do?"

### Section 6: VRAM Budget Guide
- Concrete recommendations for three hardware tiers:
  - **16GB** (consumer GPU: RTX 4060/4070): Best backbone + batch size + efficiency tricks
  - **24GB** (prosumer: RTX 3090/4090): Optimal backbone, when to use LoRA vs full fine-tune
  - **48GB** (workstation: A40/A6000): Which backbones become feasible, diminishing returns
- VRAM breakdown: where does the memory go? (model params, activations, gradients, optimizer states)
- Cost-performance Pareto frontier: which backbone gives the best accuracy per GB
- **On-device deployment considerations**: If your goal is a mobile app (scanning sheet music with a phone), mobile-optimized backbones + distillation may beat the "best" backbone

### Section 7: Conclusion and What's Next
- Summary of lessons learned
- Limitations: synthetic data, monophonic focus, single dataset
- Future directions: real-world scans, polyphonic scores, multimodal pretraining
- Links to code, configs, and WandB dashboards for reproducibility

---

## Experimental Plan

### Phase 1: Backbone Sweep (Baselines) — IN PROGRESS

Train all backbone variants with identical hyperparameters. Everything except the vision encoder is held constant.

| Config | Backbone | Params (encoder) | Total Params | Pretrained Source | Status |
|--------|----------|-------------------|--------------|-------------------|--------|
| `convnext_tiny` | ConvNeXt-Tiny | 28M | 65M | `facebook/convnext-tiny-224` | **Training** — epoch 2/10 |
| `resnet50` | ResNet-50 | 25M | 61M | `microsoft/resnet-50` | **Training** — epoch 2/10 |
| `efficientnet_b0` | EfficientNet-B0 | 5M | 42M | `google/efficientnet-b0` | **Training** — epoch 2/10 |
| `vit_small` | ViT-Small | 22M | 59M | `WinKawaks/vit-small-patch16-224` | **Training** — epoch 3/10 |
| `deit_small` | DeiT-Small | 22M | 59M | `facebook/deit-small-patch16-224` | **Training** — epoch 3/10 |
| `swin_tiny` | Swin-Tiny | 28M | ~65M | `microsoft/swin-tiny-patch4-window7-224` | **Training** — epoch 1/10 (relaunched after padding fix) |
| `mobilevit_small` | MobileViT-Small | 6M | 42M | `apple/mobilevit-small` | **Training** — epoch 2/10 |

**Early observations (epoch 2-3, as of 2026-02-22):**

| Backbone | Best Val Loss | Train Loss | Speed | Notes |
|----------|---------------|------------|-------|-------|
| DeiT-Small | 0.663 | 0.703 | 2.39 it/s | Early leader |
| ResNet-50 | 0.800 | 0.999 | 2.43 it/s | Solid CNN baseline |
| ConvNeXt-Tiny | 0.996 | 0.863 | 2.26 it/s | Still improving |
| ViT-Small | — | 0.940 | 2.91 it/s | Fastest throughput |
| EfficientNet-B0 | 1.125 | 1.073 | 1.89 it/s | Slow despite small size |
| MobileViT-Small | — | 1.105 | 1.47 it/s | Slowest (depthwise convs) |
| Swin-Tiny | — | — | — | Just relaunched |

All jobs tracked via WandB (project: `music-recognition`, tag: `backbone-sweep`).

**Deliverables:**
- Validation loss and SER/CER for each backbone
- Training throughput (samples/sec, tokens/sec)
- Peak VRAM usage per backbone
- Training time per epoch
- Loss curve comparisons (from WandB)

### Phase 2: Efficiency Ablations

Run on 2-3 selected backbones from Phase 1 (best-performing, most efficient, and one mid-range).

| Experiment | Description | What We Learn |
|------------|-------------|---------------|
| LoRA rank sweep | LoRA with rank 4, 8, 16 on frozen backbone | How much accuracy do adapters recover vs full fine-tuning? |
| QLoRA | 4-bit quantized backbone + LoRA | How far can we push VRAM savings before accuracy collapses? |
| Knowledge distillation | Best teacher -> smallest student | Can a small model match a large one with the right training signal? |
| Batch size scaling | Vary batch size (4, 8, 16, 32) with gradient accumulation | How does memory-throughput trade-off affect training? |
| Image height ablation | 128px vs 224px vs 320px | How much resolution does OMR actually need? |

**Deliverables:**
- LoRA vs full fine-tune accuracy/VRAM comparison table
- Distillation learning curves
- VRAM profiling across configurations

### Phase 3: VRAM Profiling and Recommendations

- Profile peak VRAM for each Phase 1 + Phase 2 configuration using `torch.cuda.max_memory_allocated()`
- Break down VRAM into: model parameters, activations, gradients, optimizer states
- Generate Pareto frontier plots: accuracy vs VRAM, accuracy vs throughput
- Write concrete recommendations for 16GB / 24GB / 48GB budgets

---

## Current State (2026-02-22)

### Phase 1 Backbone Sweep: Running
- **7 SLURM jobs** running across nodes 12-14 (32GB VRAM GPUs)
- All jobs on epoch 2-3 of 10, ~5h into a 48h time limit
- Estimated completion: all jobs should finish within 24-38 hours
- **Swin-Tiny** was relaunched (job 1495928) after fixing a padding bug — the original `_swin_stride` only accounted for patch embedding, not the 3 patch-merge stages that halve spatial dimensions

### Key Technical Decisions
1. **Dataset**: PDMX-Synth loaded via HuggingFace `datasets` (216K samples)
2. **Tokenizer**: LEGATO BPE tokenizer (guangyangmusic/legato, 4096 tokens + special tokens)
3. **Sequence truncation**: max_seq_len=2048 with EOS preservation
4. **Training recipe**: Matches LEGATO paper (AdamW, lr=3e-4, linear warmup 3%, AMP, effective batch=32)
5. **Decoder kept constant**: All experiments share the same 6-layer Transformer decoder with SwiGLU FFN
6. **Backbone integration**: HuggingFace `AutoModel` with automatic output dimension detection and family-specific handling (CNN spatial output vs Transformer sequence output)

### Architecture
```
MusicTrOCR (~42-65M params depending on backbone)
+-- Vision Encoder: [swappable backbone] (pretrained, fine-tuned)
|   +-- CNN family: outputs (B, C, H, W) -> reshape to (B, H*W, C)
|   +-- Transformer family: outputs (B, seq_len, D) -> strip CLS token
|   +-- Swin: requires input padded to multiples of patch_size * 2^(num_stages-1) * window_size
+-- Feature Projection: Linear(backbone_dim -> 512)
+-- 2D Positional Encoding (for CNN spatial features)
+-- Transformer Decoder (6 layers, ~30M params)
    +-- Token Embedding + Learned Positional Encoding
    +-- 6x Pre-norm DecoderLayer:
    |   +-- Masked Self-Attention (8 heads)
    |   +-- Cross-Attention (8 heads) -- region focusing
    |   +-- SwiGLU Feed-Forward (512 -> 2048 -> 512)
    +-- Output Projection: Linear(512, vocab_size)
```

### Files Structure
```
configs/
+-- luca_model.yaml              # ConvNeXt-Tiny standalone config
+-- backbones/
    +-- convnext_tiny.yaml
    +-- resnet50.yaml
    +-- efficientnet_b0.yaml
    +-- vit_small.yaml
    +-- deit_small.yaml
    +-- swin_tiny.yaml
    +-- mobilevit_small.yaml

src/
+-- data/
|   +-- unified_dataset.py       # HuggingFace dataset + BPE tokenizer loader
+-- networks/
|   +-- luca_model.py            # MusicTrOCR (VisionEncoder + TransformerDecoder)
+-- train.py                     # Training loop with AMP + gradient accumulation
+-- utils/
    +-- debug_utils.py           # Training diagnostics

train_backbone.sbatch            # SLURM job template for backbone sweep
launch_sweep.sh                  # Submit all 7 backbone jobs in parallel
```

---

## Reference: LEGATO Paper Comparison

The LEGATO paper (arxiv:2506.19065) is the state-of-the-art on PDMX-Synth:

| Aspect | LEGATO | Our Project |
|--------|--------|-------------|
| Encoder | Llama-3.2-11B-Vision (frozen, 836M) | 7 smaller backbones (5-28M) |
| Decoder | 18 layers, d=1024, 16 heads | 6 layers, d=512, 8 heads |
| Total params | ~1.2B | ~42-65M |
| Tokenizer | BPE (4097 tokens) | Same |
| Dataset | PDMX-Synth (216K) | Same |
| Optimizer | AdamW (lr=3e-4) | Same |
| Batch size | 32 | 32 (effective, via accumulation) |
| GPU | A100 80GB | 32GB (SLURM) |

We don't aim to match LEGATO's accuracy. Instead, the project asks: **how far can practical, accessible backbones go?** This is more useful to practitioners who don't have A100s.

---

## Next Steps

1. **Monitor Phase 1 sweep** — all 7 jobs should complete within ~36h
2. **Implement evaluation metrics** — SER, CER, and sequence accuracy via greedy decoding on test set
3. **Add VRAM profiling** — log `torch.cuda.max_memory_allocated()` per backbone
4. **Analyze Phase 1 results** — loss curves, accuracy table, throughput comparison
5. **Select 2-3 backbones for Phase 2** — best, smallest, and one mid-range
6. **Implement LoRA/QLoRA** — add PEFT adapter support to MusicTrOCR
7. **Implement knowledge distillation** — teacher-student training mode
8. **Run Phase 2 efficiency experiments**
9. **Profile VRAM and generate Pareto plots** (Phase 3)
10. **Write the blog post** — synthesize everything into the educational narrative
