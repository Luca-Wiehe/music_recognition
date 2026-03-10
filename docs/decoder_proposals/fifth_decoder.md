# Fifth Decoder Architecture Proposal: Hybrid CTC + Attention Decoder

**Proposed by**: Architecture Researcher (Ablation Study, Decoder #5)
**Date**: 2026-03-01
**Status**: Proposed

---

## 1. Architecture Selection and Justification

### Chosen Architecture: Hybrid CTC + Attention Decoder

From the four candidate options, I recommend **Option 3: Hybrid CTC + Attention decoder**.

### Why This Completes the Ablation Study

The existing four decoders vary primarily along the **architectural axis** (RNN → LSTM → Mamba → Conformer). The Hybrid CTC+Attention decoder is unique because it introduces variation along the **training objective axis** — it uses a *dual loss function* that coordinates CTC alignment signals with cross-entropy attention training.

| Decoder | Axis of Variation | Core Idea |
|---------|------------------|-----------|
| Vanilla RNN + Attention | Architecture | Simplest recurrent cell |
| LSTM + Attention | Architecture | Gated recurrent with memory |
| Mamba/SSM | Architecture | Linear-complexity state space |
| Conformer | Architecture | Conv + self-attention hybrid |
| **Hybrid CTC + Attention** | **Training objective** | Dual-loss: CTC alignment + attention decoding |

The Hybrid CTC+Attention decoder is **the only architecture that changes what the model is trained to do**, not just how it processes sequences. This provides a uniquely different scientific insight: *does forcing the encoder to produce monotonically aligned representations (via CTC) improve music notation recognition?*

**Why CTC fits OMR well**:
- Music notation is strictly left-to-right (unlike natural language with long-range re-ordering)
- CTC's monotonic alignment assumption matches the sequential reading of sheet music
- CTC provides a strong inductive bias that complements the more flexible attention mechanism
- This architecture is the gold standard for hybrid decoding in speech recognition (ESPnet, Whisper-adjacent), which shares structural similarity with OMR

**Why not the other options**:
- GRU + Attention: A third recurrent variant that is predictable; expected to land between RNN and LSTM, providing limited new insight
- RWKV: Interesting, but largely overlaps with Mamba in the "RNN-that-trains-like-a-transformer" niche — two architectures in the same conceptual niche weakens the ablation
- Perceiver: Interesting bottleneck design, but architecturally adjacent to the existing Transformer decoder with cross-attention; the bottleneck latents are similar in spirit to learned positional encodings in Transformers

---

## 2. Architecture Diagram (ASCII Art)

```
                         INPUT IMAGE
                             │
                    ┌────────▼────────┐
                    │   DeiT-Small    │  (frozen, 22M params)
                    │ Vision Encoder  │
                    └────────┬────────┘
                             │ (B, src_len, 512)
                    ┌────────▼────────┐
                    │ Linear Proj.    │  (optional, if enc dim ≠ 512)
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              │  CTC BRANCH                 │  ATTENTION BRANCH
              │                             │
     ┌────────▼────────┐          ┌─────────▼─────────┐
     │ LayerNorm(512)  │          │ Token Embedding    │
     └────────┬────────┘          │ (vocab × 512)     │
              │                   └─────────┬─────────┘
     ┌────────▼────────┐                    │
     │ Linear(512→4096)│          ┌─────────▼─────────────────────┐
     │  (CTC head)     │          │  Decoder Layer ×6              │
     └────────┬────────┘          │  ┌─────────────────────────┐  │
              │                   │  │ Masked Multi-Head        │  │
              │ log_softmax        │  │ Self-Attention (8 heads) │  │
              │                   │  └────────────┬────────────┘  │
     ┌────────▼────────┐          │               │               │
     │  CTC Loss       │◄──────── │  ┌────────────▼────────────┐  │
     │  (training)     │          │  │ Cross-Attention          │  │
     └─────────────────┘          │  │ (8 heads, Q←tgt,        │◄─┼── encoder memory
                                  │  │  KV←encoder)            │  │   (B, src_len, 512)
                                  │  └────────────┬────────────┘  │
                                  │               │               │
                                  │  ┌────────────▼────────────┐  │
                                  │  │ Feed-Forward (SwiGLU)   │  │
                                  │  │ d_model=512, d_ff=2048  │  │
                                  │  └────────────┬────────────┘  │
                                  └───────────────┼───────────────┘
                                                  │
                                       ┌──────────▼──────────┐
                                       │ Linear(512 → 4096)  │
                                       │ (output projection) │
                                       └──────────┬──────────┘
                                                  │
                                                  │ (B, tgt_len, 4096)
                                       ┌──────────▼──────────┐
                                       │  Cross-Entropy Loss │
                                       │  (training)         │
                                       └─────────────────────┘

              ─────────────────────────────────────────────────
              JOINT LOSS (training):
              L_total = λ·L_CTC + (1-λ)·L_CE       (λ ≈ 0.3)
              ─────────────────────────────────────────────────

              INFERENCE (beam search):
              Score = α·log P_att(y|x) + (1-α)·log P_ctc(y|x)
              (α ≈ 0.7, CTC prefix score for shallow fusion)
```

---

## 3. Layer-by-Layer Specification

### 3.1 CTC Branch

| Layer | Input Shape | Output Shape | Notes |
|-------|-------------|-------------|-------|
| Encoder output (shared) | `(B, src_len, 512)` | `(B, src_len, 512)` | From DeiT-Small |
| LayerNorm | `(B, src_len, 512)` | `(B, src_len, 512)` | Stabilises CTC projection |
| Linear (CTC head) | `(B, src_len, 512)` | `(B, src_len, 4096)` | Maps each encoder frame to vocab logits |
| Log-softmax | `(B, src_len, 4096)` | `(B, src_len, 4096)` | Required by `torch.nn.CTCLoss` |

**CTC assumption**: The encoder sequence `src_len` must be ≥ target sequence length. With DeiT-Small patch size 16 and typical sheet music image width ~2000px, `src_len` ≈ 125–250 frames. Max target length is 2048 tokens. CTC requires `src_len ≥ 2 * tgt_len + 1`. **Note**: this constrains input image width at long sequence lengths — see Section 7.2 for mitigation.

### 3.2 Attention Branch (Transformer Decoder)

| Layer | Input Shape | Output Shape | Notes |
|-------|-------------|-------------|-------|
| Token Embedding | `(B, tgt_len)` | `(B, tgt_len, 512)` | Learnable, shared with output proj. |
| Positional Encoding | `(B, tgt_len, 512)` | `(B, tgt_len, 512)` | Sinusoidal or learned |
| — Decoder Layer × 6 — | | | |
| Masked Multi-Head Self-Attn | `(B, tgt_len, 512)` | `(B, tgt_len, 512)` | 8 heads, head_dim=64, causal mask |
| Add & LayerNorm | `(B, tgt_len, 512)` | `(B, tgt_len, 512)` | Pre-norm convention |
| Multi-Head Cross-Attention | Q: `(B, tgt, 512)`, KV: `(B, src, 512)` | `(B, tgt_len, 512)` | 8 heads, attends to encoder memory |
| Add & LayerNorm | `(B, tgt_len, 512)` | `(B, tgt_len, 512)` | |
| SwiGLU FFN | `(B, tgt_len, 512)` | `(B, tgt_len, 512)` | d_ff=2048; W1,W2,W3 |
| Add & LayerNorm | `(B, tgt_len, 512)` | `(B, tgt_len, 512)` | |
| — End of 6 layers — | | | |
| Final LayerNorm | `(B, tgt_len, 512)` | `(B, tgt_len, 512)` | |
| Linear (output proj) | `(B, tgt_len, 512)` | `(B, tgt_len, 4096)` | Weight-tied with embedding |

### 3.3 Training Objective

```
L_total = λ · L_CTC + (1 - λ) · L_CE

where:
  L_CTC = torch.nn.CTCLoss(blank=0)(log_probs_ctc, targets, input_lengths, target_lengths)
  L_CE  = F.cross_entropy(logits_att.view(-1, vocab), targets_shifted.view(-1))
  λ     = 0.3  (CTC weight; tunable hyperparameter)
```

### 3.4 Inference: CTC-Guided Beam Search

At inference time, CTC prefix scores augment attention beam search:

```python
# Pseudocode
beam = CTCPrefixBeamSearch(ctc_log_probs, beam_size=10)
for each step:
    att_scores = attention_decoder.step(beam.hypotheses, encoder_memory)
    combined   = alpha * att_scores + (1 - alpha) * beam.ctc_prefix_scores
    beam.update(combined)
```

This ensures that hypotheses consistent with the CTC-aligned path are preferred, reducing hallucinations and off-track generation — a known weakness of pure attention decoders on long sequences.

---

## 4. Parameter Count Breakdown

### 4.1 CTC Branch

| Component | Formula | Parameters |
|-----------|---------|-----------|
| CTC LayerNorm | 2 × 512 | 1,024 |
| CTC Linear head | 512 × 4,096 + 4,096 | 2,101,248 |
| **CTC Branch Total** | | **~2.10M** |

### 4.2 Attention Decoder

Per decoder layer:
| Component | Formula | Parameters |
|-----------|---------|-----------|
| Self-Attn (Q,K,V,O) | 4 × 512 × 512 | 1,048,576 |
| Self-Attn LayerNorm | 2 × 512 | 1,024 |
| Cross-Attn (Q,K,V,O) | 4 × 512 × 512 | 1,048,576 |
| Cross-Attn LayerNorm | 2 × 512 | 1,024 |
| SwiGLU FFN (W1,W2,W3) | 3 × 512 × 2,048 | 3,145,728 |
| FFN LayerNorm | 2 × 512 | 1,024 |
| **Per Layer Total** | | **~5.25M** |

Full decoder:
| Component | Formula | Parameters |
|-----------|---------|-----------|
| 6 Decoder Layers | 6 × 5,245,952 | 31,475,712 |
| Token Embedding | 4,096 × 512 | 2,097,152 |
| Final LayerNorm | 2 × 512 | 1,024 |
| Output Projection | weight-tied with embedding | 0 (shared) |
| **Attention Branch Total** | | **~33.57M** |

### 4.3 Summary

| Component | Parameters |
|-----------|-----------|
| CTC Branch | ~2.10M |
| Attention Decoder | ~33.57M |
| **Grand Total** | **~35.67M** |

**Within the 10M–40M target range. ✓**

Note: If the existing Transformer decoder (37M) is being replaced rather than added alongside, this architecture is roughly equivalent in capacity at ~35.7M. The CTC head adds only ~2.1M overhead to a standard 6-layer transformer decoder.

---

## 5. Estimated VRAM Usage

With `batch_size=8`, `AMP (fp16)`, `max_tgt_len=2048`, `src_len=256`:

| Component | Estimated VRAM |
|-----------|---------------|
| Model parameters (fp32 master) | ~143 MB |
| Model parameters (fp16 working) | ~72 MB |
| Attention activations (6 layers, B=8, T=2048) | ~4.5 GB |
| Cross-attention KV cache (inference) | ~0.8 GB |
| CTC log-probs tensor `(B, src_len, vocab)` | 8 × 256 × 4096 × 2 bytes ≈ 16 MB |
| Gradients (fp32) | ~143 MB |
| Optimizer states (AdamW, fp32) | ~286 MB |
| DeiT-Small encoder activations | ~1.5 GB |
| **Estimated Total (training)** | **~8–10 GB** |

With gradient accumulation (effective batch=32, micro-batch=8) and gradient checkpointing on the decoder, peak VRAM remains well under 32GB. **Fits comfortably in budget. ✓**

---

## 6. Strengths and Weaknesses for OMR

### Strengths

1. **Monotonic alignment prior**: Sheet music is strictly left-to-right. CTC's monotonic constraint is not a limitation here — it's a *correct* inductive bias. Unlike language translation, music notation does not require non-monotonic alignments.

2. **Hallucination resistance**: Pure attention decoders can "skip ahead" or repeat symbols when attention becomes diffuse over long sequences. CTC prefix scoring forces generated tokens to stay aligned with the encoder's temporal output, reducing hallucinations in long pieces.

3. **Distinct training paradigm**: The CTC loss trains the encoder to produce frame-level predictions, effectively acting as auxiliary supervision that benefits the shared encoder features. Even a frozen DeiT encoder benefits from CTC gradients flowing through the CTC projection head.

4. **Faster convergence**: CTC provides dense gradient signal at every encoder frame. This is especially helpful early in training when the attention mechanism hasn't learned to focus yet. The model bootstraps faster.

5. **Better length extrapolation**: The CTC branch constrains beam search to consistent sequence lengths, reducing the length bias that pure attention decoders exhibit.

6. **Established benchmark**: CTC+Attention hybrid is the backbone of ESPnet (speech recognition) and has been studied extensively. Results are interpretable and comparable to prior literature.

### Weaknesses

1. **CTC sequence length constraint**: CTC requires `src_len ≥ 2 * tgt_len + 1`. For very long pieces with tgt_len → 2048, the encoder must produce ≥ 4097 frames. With DeiT-Small's patch size 16, this requires images ~65536px wide — infeasible. **Mitigation**: apply CTC only on shorter subsequences, or use a convolutional downsampler before CTC.

2. **Two-hyperparameter balance**: λ (CTC weight) and α (inference fusion weight) must both be tuned. Suboptimal values can hurt overall performance.

3. **CTC independence assumption**: CTC assumes conditional independence of output tokens given the encoder. This is violated in ABC notation (barlines, key signatures have strong context dependence). The attention branch corrects this, but it means CTC alone would perform poorly.

4. **Slightly more complex training loop**: Requires computing `input_lengths` and `target_lengths` tensors for CTC. Masked padding must be tracked carefully.

5. **Blank token management**: CTC requires a dedicated blank token (index 0). This must be consistent with the BPE vocabulary and the attention decoder's `<pad>` token usage.

---

## 7. Key Hyperparameters to Tune

| Hyperparameter | Recommended Range | Rationale |
|----------------|------------------|-----------|
| `lambda_ctc` | 0.1 – 0.5 | CTC weight in joint loss; start at 0.3 |
| `alpha_ctc` | 0.3 – 0.7 | CTC weight in inference fusion; start at 0.3 |
| `beam_size` | 5 – 20 | Beam search width; 10 is a good default |
| `n_decoder_layers` | 4 – 6 | Fewer layers → lower VRAM but weaker attention |
| `d_model` | 256, 512 | Match encoder output dimension (512) |
| `n_heads` | 8 | Standard for d_model=512 |
| `d_ff` | 1024 – 2048 | FFN expansion ratio |
| `dropout` | 0.1 – 0.3 | Apply in attention and FFN |
| `ctc_blank_idx` | 0 | Must match BPE vocab padding token |
| `label_smoothing` | 0.0 – 0.1 | On cross-entropy loss only |

### 7.1 λ Tuning Strategy

A practical approach for tuning λ:
1. Start with λ=0.3 (empirically strong in ASR literature)
2. Train a short run (5 epochs) and compare CTC character error rate vs. attention CER
3. If CTC CER >> attention CER: decrease λ (CTC is a drag)
4. If CTC CER ≈ attention CER: increase λ (CTC can contribute more)

### 7.2 CTC Sequence Length Constraint Mitigation

For OMR with long sequences, add a **temporal downsampling** step before the CTC head:

```
encoder output (B, src_len, 512)
       │
  Conv1d(512, 512, kernel=3, stride=2) × 2  → (B, src_len/4, 512)
       │
  CTC Linear Head (512 → 4096)
```

This allows CTC to operate on `src_len/4` frames, relaxing the constraint:
Required: `src_len/4 ≥ 2 * tgt_len + 1`
With src_len=1000, tgt_len ≤ 249. Sufficient for typical music excerpts.

---

## 8. Required Python Packages

All packages are already available in standard PyTorch + Hugging Face environments:

```python
# Core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss  # Built-in PyTorch

# No additional packages required beyond existing environment:
# torch >= 1.10 (CTCLoss with blank_first=True)
# transformers (for DeiT encoder)
```

**No new dependencies needed.** `torch.nn.CTCLoss` is part of core PyTorch since version 1.0.

Optional, for faster CTC beam search:
```
pip install ctcdecode   # GPU-accelerated CTC beam search with LM fusion
```
`ctcdecode` is optional — a pure Python CTC prefix beam search is sufficient for this ablation.

---

## 9. Implementation Sketch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCTCAttentionDecoder(nn.Module):
    """
    Hybrid CTC + Attention decoder for OMR.

    CTC branch: linear projection from encoder frames → vocab logits
    Attention branch: standard Transformer decoder with cross-attention
    Joint loss: L = lambda_ctc * L_CTC + (1 - lambda_ctc) * L_CE
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        lambda_ctc: float = 0.3,
        ctc_blank_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.lambda_ctc = lambda_ctc
        self.ctc_blank_idx = ctc_blank_idx

        # ── CTC Branch ────────────────────────────────────────────────
        self.ctc_norm = nn.LayerNorm(d_model)
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # ── Attention Branch ──────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = nn.Embedding(max_seq_len, d_model)  # learned
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",  # swap for SwiGLU manually if desired
            batch_first=True,
            norm_first=True,    # pre-norm (more stable)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )
        self.output_norm = nn.LayerNorm(d_model)
        # Weight-tied output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

        # ── Loss functions ─────────────────────────────────────────────
        self.ctc_loss_fn = nn.CTCLoss(blank=ctc_blank_idx, reduction="mean",
                                       zero_infinity=True)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self,
        encoder_memory: torch.Tensor,          # (B, src_len, 512)
        tgt_tokens: torch.Tensor,              # (B, tgt_len) — teacher forcing
        tgt_key_padding_mask: torch.Tensor,    # (B, tgt_len) bool
        memory_key_padding_mask: torch.Tensor, # (B, src_len) bool
        input_lengths: torch.Tensor,           # (B,) CTC input lengths
        target_lengths: torch.Tensor,          # (B,) CTC target lengths
        labels: torch.Tensor,                  # (B, tgt_len) ground truth
    ):
        # ── CTC Branch ────────────────────────────────────────────────
        ctc_input = self.ctc_head(self.ctc_norm(encoder_memory))  # (B, src, V)
        ctc_log_probs = F.log_softmax(ctc_input, dim=-1)          # (B, src, V)
        # CTCLoss expects (T, B, C)
        ctc_log_probs_t = ctc_log_probs.permute(1, 0, 2)
        # Flatten targets for CTCLoss
        targets_flat = labels.masked_select(labels != 0)
        l_ctc = self.ctc_loss_fn(
            ctc_log_probs_t, targets_flat, input_lengths, target_lengths
        )

        # ── Attention Branch ──────────────────────────────────────────
        B, T = tgt_tokens.shape
        positions = torch.arange(T, device=tgt_tokens.device).unsqueeze(0)
        tgt_emb = self.embedding(tgt_tokens) + self.pos_enc(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt_tokens.device
        )
        dec_out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=encoder_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.output_proj(self.output_norm(dec_out))  # (B, T, V)
        l_ce = self.ce_loss_fn(
            logits[:, :-1].reshape(-1, self.vocab_size),
            labels[:, 1:].reshape(-1),
        )

        # ── Joint Loss ────────────────────────────────────────────────
        loss = self.lambda_ctc * l_ctc + (1 - self.lambda_ctc) * l_ce

        return loss, logits, {"l_ctc": l_ctc.item(), "l_ce": l_ce.item()}

    @torch.no_grad()
    def greedy_decode(self, encoder_memory, max_len=512, bos_token=1, eos_token=2):
        """Simple greedy decode (attention branch only) for evaluation."""
        B = encoder_memory.size(0)
        device = encoder_memory.device
        tokens = torch.full((B, 1), bos_token, dtype=torch.long, device=device)
        for _ in range(max_len):
            T = tokens.size(1)
            positions = torch.arange(T, device=device).unsqueeze(0)
            tgt_emb = self.embedding(tokens) + self.pos_enc(positions)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            dec_out = self.transformer_decoder(tgt_emb, encoder_memory,
                                               tgt_mask=causal_mask)
            logits = self.output_proj(self.output_norm(dec_out))  # (B, T, V)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == eos_token).all():
                break
        return tokens
```

---

## 10. Ablation Study Positioning

### What Unique Scientific Question Does This Architecture Answer?

> **"Does auxiliary CTC alignment supervision improve the accuracy and reliability of autoregressive music notation decoding?"**

This is a fundamentally different question from "which architecture processes sequences best?" — it asks whether the *training signal* matters, independently of decoder capacity.

### Expected Findings and Their Interpretations

| Outcome | Interpretation |
|---------|---------------|
| CTC+Att > All others | CTC alignment is genuinely beneficial for OMR; monotonic constraint is a correct inductive bias |
| CTC+Att ≈ Transformer | Architecture dominates; training objective matters less |
| CTC+Att < Mamba/Conformer | Alignment constraint is too rigid; music notation has non-local dependencies that need freer attention |
| CTC+Att > Att alone (ablation) | CTC auxiliary loss improves attention training, even if the inference fusion doesn't help |

### Comparison Summary Table

| Decoder | Params | Inductive Bias | Training Signal | Inference |
|---------|--------|---------------|----------------|-----------|
| RNN + Att | ~12M | Sequential recurrence | CE only | Autoregressive |
| LSTM + Att | ~15M | Gated recurrence | CE only | Autoregressive |
| Mamba | ~20M | Linear SSM | CE only | Autoregressive |
| Conformer | ~25M | Conv + self-attn | CE only | Autoregressive |
| **CTC + Att** | **~35.7M** | **Monotonic align** | **CTC + CE** | **CTC-guided beam** |

---

## 11. References

1. Watanabe et al. (2017). *Hybrid CTC/Attention Architecture for End-to-End Speech Recognition*. IEEE JSTSP. — The foundational paper establishing the CTC+Attention joint training paradigm.

2. Kim et al. (2017). *Joint CTC-Attention based End-to-End Speech Recognition using Multi-task Learning*. ICASSP.

3. Karita et al. (2019). *A Comparative Study on Transformer vs RNN in Speech Applications*. ASRU. — Shows CTC+Attention consistently outperforms pure attention in low-resource settings.

4. Tuggener et al. (2021). *OMR survey*. — Notes that music notation is strictly sequential, supporting monotonic alignment assumptions.

5. ESPnet: https://github.com/espnet/espnet — Reference implementation of CTC+Attention hybrid for ASR.
