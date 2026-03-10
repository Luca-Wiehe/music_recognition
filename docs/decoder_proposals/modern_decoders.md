# Modern Decoder Architecture Proposals for OMR

**Author**: Architecture Researcher
**Date**: 2026-03-01
**Task**: Proposals for Tasks #3 and #4

---

## Overview

This document proposes two modern decoder architectures as alternatives to the existing 6-layer Transformer decoder (~37M params). Both target the 15–40M parameter range, fit within 32GB VRAM at batch_size=8 with AMP, and accept the DeiT-Small encoder output of shape `(B, src_len, 512)`.

---

## Proposal 1: Mamba (Selective State Space Model) Decoder

### Motivation

Transformers scale O(n²) in sequence length due to self-attention. For OMR with sequences up to 2048 tokens, this creates computational bottlenecks. Mamba (Gu & Dao, 2023) uses selective state space models (S6) achieving **O(n) complexity** with competitive or superior performance on long sequences. This is particularly relevant for complex musical scores that produce long ABC notation outputs.

### Architecture Overview

```
                         ABC Tokens (tgt_len, B)
                                  │
                         Token Embedding
                        (tgt_len, B, d_model=512)
                                  │
                         Positional Encoding
                                  │
                    ┌─────────────▼─────────────┐
                    │       Mamba-CA Block ×8    │
                    │  ┌─────────────────────┐   │
                    │  │  Layer Norm          │   │
                    │  │  Mamba SSM Block     │   │  ← causal self-context
                    │  │  (d_state=16,        │   │
                    │  │   d_conv=4,          │   │
                    │  │   expand=2)          │   │
                    │  ├─────────────────────┤   │
                    │  │  Residual            │   │
                    │  ├─────────────────────┤   │
                    │  │  Layer Norm          │   │
                    │  │  Cross-Attention     │   │  ← attends to encoder
                    │  │  (8 heads, d=512)    │   │     memory (B, src_len, 512)
                    │  ├─────────────────────┤   │
                    │  │  Residual            │   │
                    │  ├─────────────────────┤   │
                    │  │  Layer Norm          │   │
                    │  │  SwiGLU FFN          │   │
                    │  │  (d_ff=1024)         │   │
                    │  └─────────────────────┘   │
                    └─────────────┬─────────────┘
                                  │
                            Layer Norm
                                  │
                         Linear Projection
                        (d_model → vocab_size=4096)
                                  │
                              Logits
                        (B, tgt_len, 4096)
```

### Design Decision: Mamba with Explicit Cross-Attention (Option C)

We adopt **Option C**: replace self-attention with Mamba SSM blocks, but retain explicit multi-head cross-attention layers for encoder-decoder information flow.

**Rationale for Option C over alternatives:**
- **Option A (interleaved)**: Best expressiveness; cross-attention every layer gives rich encoder access. We adopt this.
- **Option B (prefix concatenation)**: Simpler but encoder tokens must be processed by Mamba causally — breaks the non-causal nature of encoder features and wastes state capacity.
- **Option C selected**: Mamba handles sequential decoding context efficiently (O(n)); cross-attention provides full, unmasked access to the 2D encoder memory at each layer. This cleanly separates the two roles.

### Layer-by-Layer Specification

| Component | Input Shape | Output Shape | Details |
|-----------|-------------|--------------|---------|
| Token Embedding | `(B, tgt_len)` | `(B, tgt_len, 512)` | Learned, vocab_size=4096 |
| Sinusoidal Pos Enc | `(B, tgt_len, 512)` | `(B, tgt_len, 512)` | Fixed, max_len=2048 |
| **Mamba-CA Block ×8** | | | |
| ↳ LayerNorm | `(B, T, 512)` | `(B, T, 512)` | Pre-norm |
| ↳ Mamba SSM | `(B, T, 512)` | `(B, T, 512)` | d_state=16, d_conv=4, expand=2; causal |
| ↳ Residual | — | `(B, T, 512)` | |
| ↳ LayerNorm | `(B, T, 512)` | `(B, T, 512)` | Pre-norm |
| ↳ Cross-Attention | Q:`(B,T,512)`, KV:`(B,S,512)` | `(B, T, 512)` | 8 heads, d_k=64; unmasked over encoder |
| ↳ Residual | — | `(B, T, 512)` | |
| ↳ LayerNorm | `(B, T, 512)` | `(B, T, 512)` | Pre-norm |
| ↳ SwiGLU FFN | `(B, T, 512)` | `(B, T, 512)` | d_ff=1024 (×2 for gate + value) |
| ↳ Residual | — | `(B, T, 512)` | |
| Final LayerNorm | `(B, T, 512)` | `(B, T, 512)` | |
| Linear Proj | `(B, T, 512)` | `(B, T, 4096)` | Output logits |

### Mamba SSM Block Internal Details

The S6 (selective scan) mechanism in each Mamba block:

```
Input x: (B, T, d_model=512)
        │
    Linear expand: d_model → d_inner = d_model × expand = 1024
        │
   ┌────┴────────────────────────┐
   │                             │
   ▼                             ▼
 Conv1d(d_conv=4)           Linear → ∆, B, C
   │                       (selective params, input-dependent)
   SiLU                         │
   │                             │
   SSM(A, B, C, ∆) ◄────────────┘
   (discretized, parallel scan)
   │
   × (element-wise gate)
   │
Linear project: d_inner → d_model
```

**Key parameters:**
- `d_model = 512` (matches encoder output)
- `d_state = 16` (SSM state dimension, controls memory capacity)
- `d_conv = 4` (local convolution kernel, captures immediate context)
- `expand = 2` (inner dimension multiplier: d_inner = 1024)
- Causal convolution: enforces autoregressive constraint

### Parameter Count Breakdown

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| Token Embedding | 2.10M | 4096 × 512 |
| Per Mamba-CA Block: | | |
| ↳ Mamba SSM | ~2.10M | proj_in: 512→1024×2; conv1d: 4×1024; SSM params: ~small; proj_out: 1024→512 |
| ↳ Cross-Attention | 1.05M | 4 × (512×512) = 1,048,576 (Q,K,V,O projections) |
| ↳ SwiGLU FFN | 1.57M | 512→1024 (×2 for SwiGLU) + 1024→512 = 3×512×1024 |
| ↳ LayerNorms (×3) | ~3K | 3 × 2 × 512 |
| **Per block total** | ~4.72M | |
| 8 blocks total | 37.76M | |
| Final LN + Proj | 2.10M | 512×4096 |
| **Grand Total** | **~42M** | |

> **Note**: To hit the 15–40M target, reduce to **6 layers** → ~30M params, or reduce d_model to 384 → ~24M params. Recommended: 6 layers, d_model=512 → **~30M params**.

**Revised 6-layer breakdown:**
- Token Embedding: 2.10M
- 6 × Mamba-CA blocks: 28.3M
- Output projection: 2.10M
- **Total: ~32.5M params**

### VRAM Usage Estimate (6-layer, d_model=512, batch=8, tgt_len=512 avg)

| Component | VRAM |
|-----------|------|
| Model parameters (FP16 with AMP) | ~65MB |
| Activations (SSM + cross-attn, B=8, T=512) | ~3.2GB |
| Optimizer states (AdamW, FP32 master copy) | ~260MB |
| KV cache for cross-attention (inference only) | ~0.8GB |
| **Estimated total (training)** | **~6–8GB** |

Well within 32GB budget. Leaves ample room for longer sequences or larger batches.

### Teacher Forcing and Autoregressive Generation

**Training (teacher forcing):**
```python
# x: (B, tgt_len) — shifted-right ground truth tokens
emb = self.embedding(x)  # (B, T, 512)
for block in self.blocks:
    emb = block(emb, encoder_memory)  # Mamba causal + cross-attn
logits = self.proj(self.norm(emb))  # (B, T, 4096)
loss = F.cross_entropy(logits.view(-1, 4096), targets.view(-1))
```

**Inference (autoregressive):**
```python
# Mamba supports recurrent inference mode: O(1) per step
# Each SSM maintains a hidden state h: (B, d_state, d_inner)
generated = [bos_token]
h = [None] * num_layers  # SSM states
for step in range(max_len):
    token = torch.tensor([generated[-1]])
    emb, h = self.step(token, encoder_memory, h)  # single-step forward
    next_token = logits.argmax(-1)
    generated.append(next_token)
    if next_token == eos_token: break
```

The Mamba recurrent mode processes each new token in **O(1)** time (vs. O(n) for Transformer with KV cache), making inference significantly faster for long sequences.

### Practical Considerations: mamba-ssm Package

```bash
# Installation (requires CUDA >= 11.6, PyTorch >= 1.12)
pip install mamba-ssm causal-conv1d

# Verify hardware compatibility
python -c "import mamba_ssm; print(mamba_ssm.__version__)"
```

**Hardware requirements:**
- CUDA capability ≥ 7.0 (V100, A100, H100 — all supported)
- The `mamba-ssm` package uses custom CUDA kernels; requires compilation
- Fallback pure-PyTorch implementation available but ~10× slower
- On A100/H100 GPUs (likely in a 32GB VRAM setup), expect full kernel support

**Alternative if mamba-ssm unavailable:**
```python
# Pure PyTorch S4-style SSM (no custom CUDA):
from torch.nn.utils.rnn import pad_packed_sequence
# Or use: pip install s4torch  # pure-PyTorch S4
```

### Strengths for OMR

1. **O(n) complexity**: Long ABC sequences (2048 tokens) processed linearly; no quadratic attention cost
2. **Efficient inference**: Recurrent mode is O(1) per decoding step — critical for beam search
3. **Long-range memory**: SSM state space captures dependencies across entire score sections
4. **Low memory**: No KV cache growth during autoregressive decoding
5. **Musical repetition**: State space models naturally handle repeated motifs and structures

### Weaknesses for OMR

1. **Cross-attention overhead**: Adding explicit cross-attention layers partially negates O(n) benefit for encoder interaction
2. **Less interpretable**: No attention maps to visualize what encoder regions the decoder attends to
3. **Dependency**: Requires `mamba-ssm` with custom CUDA kernels; build issues possible
4. **Less battle-tested**: Fewer OMR/OCR applications compared to Transformers
5. **Positional encoding**: Mamba's implicit positional bias may need augmentation for music notation ordering

### Key Hyperparameters to Tune

| Hyperparameter | Default | Search Range | Impact |
|----------------|---------|--------------|--------|
| `num_layers` | 6 | 4–10 | Model capacity, VRAM |
| `d_model` | 512 | 256–512 | Matches encoder dim |
| `d_state` | 16 | 8–64 | SSM memory capacity |
| `d_conv` | 4 | 2–8 | Local context window |
| `expand` | 2 | 1–4 | Inner dim multiplier |
| Cross-attn heads | 8 | 4–16 | Encoder-decoder alignment |
| `d_ff` (SwiGLU) | 1024 | 512–2048 | FFN capacity |
| Dropout | 0.1 | 0.0–0.3 | Regularization |

### Required Python Packages

```
mamba-ssm>=1.2.0       # Core SSM implementation (CUDA required)
causal-conv1d>=1.1.0   # Dependency of mamba-ssm
# Standard (already available):
torch>=2.0
einops>=0.6            # Tensor rearrangement utilities
```

---

## Proposal 2: Conformer Decoder (Convolution + Attention Hybrid)

### Motivation

The Conformer architecture (Gulati et al., 2020) was developed for **Automatic Speech Recognition (ASR)** and achieves state-of-the-art results by combining:
- **Multi-head attention** for global, long-range dependencies
- **Depthwise separable convolution** for local, fine-grained patterns

This dual nature directly parallels the structure of **musical notation**:
- *Local patterns*: Note beams, ties, slurs, tuplets span 2–8 adjacent tokens
- *Global context*: Key signatures, time signatures, repeat bars affect the entire score
- *Sequential structure*: Music unfolds in time like speech, with local articulation and global phrase structure

ASR and OMR share structural similarities: both convert a 2D feature sequence (spectrogram / image features) into a 1D symbol sequence with both local phoneme/symbol patterns and global phrase/section structure.

### Architecture Overview

```
                         ABC Tokens (B, tgt_len)
                                  │
                         Token Embedding
                        (B, tgt_len, d_model=512)
                                  │
                    Positional Encoding (sinusoidal)
                                  │
                    ┌─────────────▼─────────────┐
                    │    Conformer-Dec Block ×6  │
                    │                            │
                    │  ┌──────────────────────┐  │
                    │  │ FF Module (1/2 scale) │  │  ← MacaronNet design
                    │  │ LN → Linear(512→2048) │  │
                    │  │ SiLU → Linear(2048→512│  │
                    │  │ ×0.5 residual         │  │
                    │  ├──────────────────────┤  │
                    │  │ Causal Self-Attention │  │  ← masked, relative pos
                    │  │ LN → MHA(8 heads)    │  │
                    │  │ + residual           │  │
                    │  ├──────────────────────┤  │
                    │  │ Cross-Attention       │  │  ← attends to encoder
                    │  │ LN → MHA(8 heads)    │  │     (B, src_len, 512)
                    │  │ + residual           │  │
                    │  ├──────────────────────┤  │
                    │  │ Causal Conv Module   │  │  ← local patterns
                    │  │ LN → Pointwise(×2)  │  │
                    │  │ GLU → DepthwiseConv  │  │
                    │  │ BN → Swish → PW      │  │
                    │  │ Dropout → residual   │  │
                    │  ├──────────────────────┤  │
                    │  │ FF Module (1/2 scale) │  │  ← Macaron second half
                    │  │ LN → Linear(512→2048) │  │
                    │  │ SiLU → Linear(2048→512│  │
                    │  │ ×0.5 residual         │  │
                    │  └──────────────────────┘  │
                    └─────────────┬─────────────┘
                                  │
                            Layer Norm
                                  │
                         Linear (512 → 4096)
                                  │
                              Logits
                        (B, tgt_len, 4096)
```

### Conformer Block Adapted for Decoder

The original Conformer was encoder-only (ASR). We adapt it for **decoder** use with three modifications:

1. **Causal masking** in self-attention (standard decoder causal mask)
2. **Causal convolution** using left-padded depthwise conv (no future token leakage)
3. **Cross-attention module** inserted between self-attention and convolution

**Causal Convolution Implementation:**
```python
class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.padding = kernel_size - 1  # left-only padding
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              groups=channels, bias=False)

    def forward(self, x):
        # x: (B, T, C) → (B, C, T)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))  # pad left only
        x = self.conv(x)
        return x.transpose(1, 2)  # (B, T, C)
```

### Layer-by-Layer Specification

| Module | Sub-component | Input → Output | Parameters |
|--------|---------------|----------------|------------|
| Token Embedding | Lookup | `(B,T)→(B,T,512)` | 4096×512 = 2.1M |
| Pos Encoding | Sinusoidal | `(B,T,512)→(B,T,512)` | 0 (fixed) |
| **Conformer Block ×6** | | | |
| FF₁ (Macaron 1st half) | LN + Linear + SiLU + Linear | `(B,T,512)→(B,T,512)` | 2×(512×2048+2048×512) = 4.2M |
| Causal Self-Attention | LN + MHA (mask) + proj | `(B,T,512)→(B,T,512)` | 4×512² = 1.05M |
| Relative Pos Bias | Sinusoidal rel. encoding | integrated | ~negligible |
| Cross-Attention | LN + MHA (unmasked) + proj | `(B,T,512)→(B,T,512)` | 4×512² = 1.05M |
| Causal Conv Module | LN + PW(×2) + GLU + DW-Conv + BN + Swish + PW | `(B,T,512)→(B,T,512)` | see below |
| ↳ Pointwise expand | Conv1d(512→1024) | `(B,T,512)→(B,T,1024)` | 512×1024 = 0.52M |
| ↳ GLU gate | split 1024 → 512 | in-place | 0 |
| ↳ Depthwise Conv | Conv1d(512, k=31, groups=512) | `(B,T,512)→(B,T,512)` | 512×31 = 15.9K |
| ↳ BatchNorm | BN(512) | `(B,T,512)→(B,T,512)` | 2×512 = 1K |
| ↳ Pointwise contract | Conv1d(512→512) | `(B,T,512)→(B,T,512)` | 512² = 0.26M |
| FF₂ (Macaron 2nd half) | same as FF₁ | `(B,T,512)→(B,T,512)` | 4.2M |
| **Per block total** | | | **~10.3M** |
| **6 blocks** | | | **~61.8M** |

> **Too large!** Reduce to reduce d_model or layers. With **d_model=384**, **6 layers**:
> - FF: 2×(384×1536×2) = 2.36M/block
> - Self-Attn: 4×384² = 0.59M
> - Cross-Attn: 4×384² = 0.59M
> - Conv: ≈0.4M
> - **~4.0M/block × 6 = 24M + 2.36M embedding/proj ≈ 26M total** ✓

**Recommended configuration: d_model=384, 6 layers → ~26M params**

### Parameter Count Breakdown (d_model=384)

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Token Embedding | 1.57M | 4096 × 384 |
| Per Conformer-Dec Block | | |
| ↳ FF₁ (half-step) | 1.18M | 384→1536→384 |
| ↳ Self-Attention | 0.59M | 4 × 384² |
| ↳ Cross-Attention | 0.59M | 4 × 384² |
| ↳ Conv Module | 0.60M | PW_expand + DW + PW_contract |
| ↳ FF₂ (half-step) | 1.18M | 384→1536→384 |
| ↳ LayerNorms (×5) | 3.8K | |
| **Per block total** | **~4.14M** | |
| 6 blocks | 24.84M | |
| Final LN + Proj | 1.57M | 384 → 4096 |
| **Grand Total** | **~28M params** | within 15–40M ✓ |

### VRAM Usage Estimate (d_model=384, 6 layers, batch=8, tgt_len=512)

| Component | VRAM (FP16 AMP) |
|-----------|----------------|
| Model parameters | ~56MB |
| Self-attention activations (causal mask, B=8, T=512) | ~1.2GB |
| Cross-attention activations (B=8, T=512, S=196) | ~0.8GB |
| Conv module activations | ~0.6GB |
| FFN activations | ~1.5GB |
| Optimizer (AdamW FP32 master) | ~224MB |
| Gradient checkpointing (optional) | saves ~40% activation |
| **Estimated total (training)** | **~5–7GB** |

Well within 32GB budget.

### Integration of Cross-Attention to Encoder

The cross-attention module in each Conformer-Dec block:
```python
class ConformerDecoderBlock(nn.Module):
    def forward(self, tgt, memory):
        # memory: (B, src_len, 512) from DeiT-Small
        # tgt: (B, tgt_len, d_model)

        # Macaron FF1
        x = tgt + 0.5 * self.ff1(self.norm_ff1(tgt))

        # Causal self-attention (autoregressive mask)
        causal_mask = self._causal_mask(tgt.size(1))
        x = x + self.self_attn(self.norm_sa(x), causal_mask)

        # Cross-attention to encoder (no mask — full access)
        # Projects memory from 512 → d_model if needed
        x = x + self.cross_attn(self.norm_ca(x), memory)

        # Causal convolution (local patterns)
        x = x + self.conv_module(self.norm_conv(x))

        # Macaron FF2
        x = x + 0.5 * self.ff2(self.norm_ff2(x))

        return self.final_norm(x)
```

**Encoder dimension projection** (DeiT outputs 512, d_model=384):
```python
# In __init__: project encoder memory to d_model
self.encoder_proj = nn.Linear(512, 384)
# In cross_attn: K and V come from projected memory
```

### Teacher Forcing and Autoregressive Generation

**Training:**
```python
tgt_emb = self.embed(tgt_input)          # (B, T, 384)
tgt_emb = self.pos_enc(tgt_emb)
for block in self.conformer_blocks:
    tgt_emb = block(tgt_emb, encoder_memory)
logits = self.proj(self.final_norm(tgt_emb))  # (B, T, 4096)
```

**Inference (autoregressive with KV cache):**
```python
# Standard Transformer-style KV caching applies to both
# self-attention and cross-attention layers
# Cross-attention KV is computed once from encoder_memory
# Self-attention KV grows by 1 per step
```

### Why Conformer Is Particularly Suitable for OMR

The parallels between ASR and OMR make Conformer an especially well-motivated choice:

| ASR Task | OMR Task | Conformer Module |
|----------|----------|-----------------|
| Phoneme boundaries | Note/rest boundaries | Depthwise Conv (k=31) |
| Pitch accent (local) | Accidentals, ornaments | Depthwise Conv |
| Phrase-level prosody | Key/time signature effects | Self-attention |
| Speaker-level style | Score-level style (meter) | Self-attention |
| Coarticulation (adjacent phones) | Beamed note groups | Conv kernel |
| Long pauses / sentence breaks | Double barlines, repeats | Self-attention |

**The depthwise convolution kernel size k=31** spans ~31 adjacent tokens. In music notation, a typical phrase spans 8–16 notes; with BPE tokenization, this covers 20–50 tokens. A k=31 kernel captures fine local notation patterns (beam groups, slurs, ties) while attention captures global structure (key, meter, form).

**ASR precedent**: Conformer achieves 1.9% WER on LibriSpeech (vs. 2.1% for RNN-T and 2.4% for Transformer). The OMR analogy suggests similar gains over pure-attention or pure-recurrent baselines.

### Convolution Kernel Size Analysis for OMR

| Kernel Size | Token Span | Covers | Notes |
|-------------|-----------|--------|-------|
| k=7 | 7 tokens | Single note w/ modifiers | Too small |
| k=15 | 15 tokens | 2–4 note beam group | Small, fast |
| k=31 | 31 tokens | Full measure (~8 notes + BPE) | **Recommended** |
| k=63 | 63 tokens | Multi-measure phrase | Risk of overfit |

### Strengths for OMR

1. **Dual inductive bias**: Both local conv and global attention match musical structure
2. **ASR-proven**: Directly validated on sequential symbol prediction from audio/image features
3. **No new dependencies**: Pure PyTorch, uses standard `nn.Conv1d` and `nn.MultiheadAttention`
4. **Interpretable**: Attention maps show which image regions correspond to which output tokens
5. **Causal conv correctness**: Left-padding ensures no future leakage during training
6. **MacaronNet FF**: Two half-scale FFN modules (one before, one after) improves gradient flow
7. **Relative positional encoding**: Naturally captures note ordering without absolute position limits

### Weaknesses for OMR

1. **O(n²) self-attention**: Still quadratic in tgt_len; for 2048-token sequences, this is 2048²/2 = 2M attention values per layer
2. **Batch normalization**: BN in conv module is sensitive to batch size; consider replacing with GroupNorm or LayerNorm for small batches
3. **Causal conv padding**: Requires careful left-padding to avoid sequence-length growth
4. **Slightly more complex**: More modules per block than standard Transformer
5. **Cross-attention dimension mismatch**: DeiT outputs 512, d_model=384 requires a projection layer

### Key Hyperparameters to Tune

| Hyperparameter | Recommended | Search Range | Impact |
|----------------|-------------|--------------|--------|
| `d_model` | 384 | 256–512 | Capacity, VRAM |
| `num_layers` | 6 | 4–10 | Depth |
| `conv_kernel_size` | 31 | 7, 15, 31, 63 | Local context span |
| `n_heads` (self-attn) | 6 (=384/64) | 4–8 | Global attention resolution |
| `n_heads` (cross-attn) | 8 | 4–16 | Encoder alignment |
| `d_ff` | 1536 (=4×384) | 1024–3072 | FFN capacity |
| `ff_scale` | 0.5 | 0.25–1.0 | MacaronNet FF contribution |
| `conv_dropout` | 0.1 | 0.0–0.3 | Conv regularization |
| `attn_dropout` | 0.1 | 0.0–0.2 | Attention regularization |
| Relative pos encoding | True | True/False | Positional bias type |

### Required Python Packages

```
# All standard — no new dependencies beyond existing codebase:
torch>=2.0
torchvision>=0.15
# Optional for relative positional encoding:
einops>=0.6  # likely already installed
```

---

## Comparison Summary

| Property | Mamba Decoder | Conformer Decoder | Existing Transformer |
|----------|--------------|-------------------|---------------------|
| **Self-context** | Mamba SSM (O(n)) | Causal MHA (O(n²)) | Causal MHA (O(n²)) |
| **Encoder access** | Cross-attention | Cross-attention | Cross-attention |
| **Local patterns** | SSM d_conv=4 | DW-Conv k=31 | None (attention only) |
| **Complexity** | O(n) | O(n²) | O(n²) |
| **Inference mode** | Recurrent (O(1)/step) | KV cache (O(n)/step) | KV cache (O(n)/step) |
| **Params** | ~32M (6L, d=512) | ~28M (6L, d=384) | ~37M (6L, d=512) |
| **VRAM (train)** | ~6–8GB | ~5–7GB | ~8–10GB |
| **Extra deps** | `mamba-ssm` (CUDA) | None | None |
| **OMR suitability** | Good (long seqs) | Excellent (local+global) | Good (proven) |
| **Implementation risk** | Medium (CUDA dep) | Low (pure PyTorch) | — |
| **ASR/OCR precedent** | Limited | Strong | Strong |

## Recommendation

For **immediate integration** with lowest risk: **Conformer Decoder** (no new dependencies, pure PyTorch, strong ASR precedent directly applicable to OMR).

For **long sequences and fast inference** (if beam search speed is critical): **Mamba Decoder** (superior scaling for 2048-token sequences, O(1) per-step inference).

Both architectures can be trained with the existing AdamW + gradient accumulation + AMP setup. The Conformer requires only adding `einops` (if not present); the Mamba decoder requires installing `mamba-ssm` with CUDA kernels.

---

*Document prepared by Architecture Researcher for OMR decoder comparison study.*
