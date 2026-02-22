# Teaching Machines to Read Music: A Practical Guide to Optical Music Recognition

> *What happens when you swap the "eyes" of a music-reading AI? We trained 7 vision
> encoders on 216,000 synthetic sheet music images to find out which architectures
> actually work -- and which are all hype.*

---

## Table of Contents

0. [A 2-Minute Music Notation Crash Course](#section-0)
1. [What Makes Music Hard to Read (for Machines)?](#section-1)
2. [How OMR Models Work: An Architecture Walkthrough](#section-2)
3. [The Experiment: Swapping Vision Backbones](#section-3)
4. [Results and Analysis](#section-4)
5. [Making It Practical: Efficiency Techniques](#section-5)
6. [The VRAM Budget Guide](#section-6)
7. [Conclusion and What's Next](#section-7)

---

## 0. A 2-Minute Music Notation Crash Course {#section-0}

You don't need to read music to follow this post, but a 2-minute primer will make everything click. [...]

### The Staff and Clef

A musical staff is five horizontal lines. A clef symbol at the start tells you which line corresponds to which pitch. [...]

> **Figure 0a: Annotated Staff**
>
> *[A single staff with treble clef, annotated with: staff lines labeled with note names (E, G, B, D, F),
> spaces labeled (F, A, C, E), a time signature (4/4), and a key signature (one sharp).
> Each element has a colored label with a one-line explanation.]*
>
> Caption: "The five lines and four spaces of a staff encode pitch. The clef, time signature,
> and key signature set the context for everything that follows."

### Notes, Rests, and Durations

A note has up to three parts: a notehead (filled or hollow), a stem (the vertical line), and optional flags or beams. [...]

> **Figure 0b: Note Anatomy**
>
> *[A single measure showing: whole note, half note, quarter note, eighth note, sixteenth note,
> each labeled with its duration. Below: the corresponding rest symbols. Annotations point out
> notehead, stem, flag, and beam.]*
>
> Caption: "Duration is encoded in the shape: filled vs. hollow noteheads, presence of stems,
> and the number of flags or beams."

### What Is ABC Notation?

ABC notation is a text-based format for representing music -- think of it as "source code" for sheet music. [...]

```
X:1
T:Example
M:4/4
K:C
C D E F | G A B c |
```

> **Figure 0c: ABC Notation Side-by-Side**
>
> *[Left: A rendered sheet music image of the ABC snippet above.
> Right: The ABC text with color-coded mapping -- each note in the text is connected
> by a colored line to the corresponding note in the image.]*
>
> Caption: "ABC notation maps directly to visual notation. Our model learns to produce
> the text on the right from the image on the left."

### What the Tokenizer Does

Our model doesn't predict raw ABC characters -- it uses a BPE (Byte-Pair Encoding) tokenizer that learns common patterns. [...]

```
Raw ABC:  "C D E F | G A B c |"
BPE tokens: ["C D", "E F", "|", "G A", "B c", "|"]
Token IDs:  [142, 307, 5, 891, 1204, 5]
```

> [!TIP]
> **Why BPE?** Character-level prediction would require the model to output every space,
> pipe, and letter individually. BPE groups common patterns into single tokens -- "C D"
> becomes one prediction instead of three. This shortens sequences and lets the model
> learn musical patterns (common intervals, chord voicings) as atomic units.

---

## 1. What Makes Music Hard to Read (for Machines)? {#section-1}

### 1.1 What Is Optical Music Recognition?

If OCR is teaching a computer to read text, then OMR is teaching it to read *music* -- taking a photograph of a musical score and converting it into a machine-readable format like ABC notation or MIDI. [...]

> **Figure 1: The OMR Pipeline at a Glance**
>
> *[Diagram: Sheet music image (input) -> Vision Encoder -> Decoder -> Token sequence -> Rendered notation / Audio.
> Include a real example: a few bars on the left, ABC tokens in the middle, and a piano roll on the right.]*
>
> Caption: "OMR converts a visual score into a symbolic representation that computers can process, play back, or edit."

> [!NOTE]
> **OMR is not audio transcription.** Audio-to-MIDI (e.g., "Shazam for sheet music") works from
> sound waves. OMR works from *images*. They are completely different problems -- we never hear
> a single note. Our input is pixels, not waveforms.

### 1.2 Why Is Music Harder Than Text?

At first glance, music should be *easier* than handwritten text -- scores use standardized symbols and clean layouts. But four properties make it uniquely challenging: [...]

#### 2D Spatial Structure

Unlike text, which flows left-to-right on a single baseline, music is fundamentally two-dimensional. [...]

> **Figure 2: Music Is Not Just Left-to-Right**
>
> *[Annotated sheet music showing three spatial relationships:
> (1) Horizontal arrows: melody flowing left to right
> (2) Vertical arrows: chord notes stacked on top of each other
> (3) Diagonal arrows: beams and slurs connecting notes across both dimensions
> Below: a line of English text with a single horizontal arrow for comparison.]*
>
> Caption: "Music encodes information in at least three spatial directions.
> Text OCR only needs to handle one."

#### Dense Symbol Layout

A single measure can pack an astonishing number of overlapping symbols: noteheads, stems, flags, beams, accidentals, dynamics, articulations. [...]

> **Figure 3: Symbol Density Comparison**
>
> *[Side-by-side: (Left) A line of text with bounding boxes around each character.
> (Right) A measure of piano music with bounding boxes around every symbol.
> Highlight overlapping boxes in the music.]*
>
> Caption: "A single measure of piano music can contain more distinct symbols than
> a paragraph of text, and many of them overlap."

<!-- INTERACTIVE: "Spot the Difference" Quiz
     Show 3-4 pairs of nearly identical music snippets differing by one symbol.
     Reader tries to find the difference. Teaches the density problem viscerally.
     After struggling, reveal: "This is what the model does for every symbol, on every page." -->

#### Long Output Sequences

A single page of orchestral music can produce thousands of tokens -- far longer than typical OCR output. [...]

#### Context Dependence

The meaning of a dot depends entirely on *where* it appears: next to a notehead it means "dotted note" (50% longer), above a note it means "staccato" (play short). [...]

> **Figure 4: The Same Symbol, Two Meanings**
>
> *[Two annotated music snippets side-by-side:
> (Left) A dotted quarter note -- dot circled in red, label: "Augmentation dot (duration x 1.5)"
> (Right) A note with staccato dot above -- same dot circled in red, label: "Staccato (play short)"]*
>
> Caption: "The same black dot means completely different things depending on its position.
> The model must learn spatial context, not just symbol identity."

### 1.3 A Brief History of OMR

The earliest OMR systems (1960s) used hand-crafted rules: find staff lines, segment symbols, classify each one. [...]

> **Figure 5: The Evolution of OMR**
>
> *[Timeline: Rule-based (1960s-2000s) -> CNN classifiers (2010s) -> End-to-end seq2seq (2020s, "You are here")]*
>
> Caption: "OMR evolved from brittle pipelines to end-to-end neural models that learn to 'read' directly from pixels."

### 1.4 Our Dataset: PDMX-Synth

For this study, we use PDMX-Synth, a dataset of 216,000 synthetically rendered sheet music images paired with ABC notation transcriptions. [...]

| Property | Value |
|----------|-------|
| Total samples | 216,000 |
| Image type | Synthetically rendered single-staff excerpts |
| Label format | ABC notation (BPE tokenized) |
| Tokenizer | LEGATO BPE (4,096 + special tokens) |
| Train / Val / Test | 171K / 21K / 21K |
| Source | `guangyangmusic/PDMX-Synth` on HuggingFace |

*Table 1: PDMX-Synth summary. Synthetic data isolates architecture performance from
data quality issues -- like testing cars on a clean track before going off-road.*

> [!NOTE]
> **Why synthetic data?** Real annotated sheet music is scarce and expensive to label.
> Synthetic data gives us abundant, perfectly labeled examples. The trade-off: models
> trained on synthetic data may struggle with real-world scans (coffee stains, uneven
> lighting, handwritten annotations). We address this limitation in Section 7.

<!-- INTERACTIVE: Image Annotation Overlay
     Hover over regions of a sheet music image to see the corresponding ABC tokens highlighted.
     Builds intuition for the input-output mapping before any architecture discussion. -->

---

## 2. How OMR Models Work {#section-2}

### 2.1 The Encoder-Decoder Idea

At its core, our OMR system follows a pattern used by machine translation and image captioning: an **encoder** that processes the input, a **decoder** that generates the output, and **cross-attention** that connects them. [...]

> [!TIP]
> **The Analogy**: The encoder is a music student studying a full page of sheet music and
> building a mental map. The decoder is that same student performing the piece -- reading
> one note at a time, glancing back at the page (cross-attention) to see what comes next,
> while remembering everything played so far (self-attention).

> **Figure 6: The Encoder-Decoder Architecture**
>
> *[Three labeled blocks:
> (1) Vision Encoder (blue): Image enters, spatial feature map exits
> (2) Cross-Attention (orange): Arrows from decoder to encoder, with attention heatmap overlay
> (3) Decoder (green): Generates tokens one at a time with feedback loop
> Dotted arrow from output back to decoder input.]*
>
> Caption: "The encoder sees the whole image at once. The decoder generates one symbol at
> a time, using cross-attention to focus on the relevant image region for each prediction."

<!-- INTERACTIVE: Architecture Explorer ("Inside MusicTrOCR")
     Scrollytelling diagram: sticky model visualization on left, prose scrolls on right.
     As reader scrolls, each component highlights and shows tensor shapes flowing through.
     Backbone dropdown swaps the encoder and updates all dimensions.
     Click any block for code snippet + explanation panel. -->

#### Key Terms (Glossary Box)

| Term | Plain English |
|------|---------------|
| **d_model** (512) | The width of the representation -- how many numbers describe each token or image region |
| **Attention heads** (8) | Parallel "perspectives" that let the model attend to different information simultaneously |
| **Autoregressive** | Each prediction depends on all previous ones -- like writing a sentence word by word |
| **Cross-attention** | The decoder creates a "question" from the current token; the encoder provides "answers" from image regions; dot product picks the best match |
| **Causal mask** | Prevents the decoder from peeking at future tokens during training |
| **Teacher forcing** | During training, feed the correct previous token (not the model's own prediction) |
| **Pre-norm** | Apply layer normalization before (not after) each sub-layer -- makes training more stable |
| **SwiGLU** | A gated activation function: `gate(x) * activate(x)`. Empirically better than plain ReLU in transformers |

*Table 2: Terminology reference. Hover or tap any term in the post for this definition.*

### 2.2 Why Not CTC?

If you've worked on speech recognition or text OCR, you might wonder: why not use CTC (Connectionist Temporal Classification)? [...]

> [!TIP]
> **CTC vs. Cross-Attention Analogy**: CTC is like a security camera with a fixed left-to-right
> sweep -- it must decide what each column contains. Cross-attention is like a human who can
> freely look anywhere on the page at any time. For 1D text, the fixed sweep works. For 2D
> music (chords, beams, slurs), you need the freedom to look around.

> **Figure 7: CTC vs. Cross-Attention for OMR**
>
> *[Two panels:
> (Left) CTC: 1D sweep struggles with a chord (three stacked notes). Red flash: "CTC can only
>   output one token per time step -- chords break the monotonic alignment assumption."
> (Right) Cross-attention: Spotlight stays on the same region for all three chord notes.
>   The decoder outputs C4, E4, G4 sequentially while attending to the same image region.]*
>
> Caption: "CTC forces left-to-right alignment. Cross-attention lets the decoder attend
> anywhere, handling chords and 2D structure naturally."

<!-- INTERACTIVE: CTC vs Autoregressive Comparison ("Why Not CTC?")
     Split-screen animation. Left: CTC sweep with failure on chord. Right: cross-attention
     spotlight generating tokens one at a time. Play button runs both simultaneously. -->

### 2.3 Why Not Just Use a Vision-Language Model?

LEGATO achieves remarkable accuracy using Llama-3.2-11B-Vision (836M encoder params, ~80GB VRAM). Our question is different: how far can we get with 20x fewer parameters on hardware people actually own? [...]

| | LEGATO | Our Approach |
|--|--------|-------------|
| Encoder | Llama-3.2-11B-Vision (836M params) | 7 smaller backbones (5-28M params) |
| Decoder | 18 layers, d=1024 | 6 layers, d=512 |
| Total params | ~1.2B | ~42-65M |
| Min. VRAM | ~80 GB (A100, ~$15,000) | ~8-16 GB (RTX 4060, ~$300) |

*Table 3: We don't aim to beat LEGATO. We ask: what's achievable on hardware most people own?*

### 2.4 MusicTrOCR Step by Step

Let's walk through our architecture piece by piece. [...]

> **Figure 8: Full MusicTrOCR Architecture**
>
> *[Detailed diagram showing complete data flow:
> Input (3, 224, W) -> [Swappable Backbone] -> Feature Projection (Linear) ->
> [2D Pos Encoding for CNNs] -> Flatten -> Encoder Features
>                                                    |
> Target Tokens -> Embedding + Pos Encoding -> [Masked Self-Attn] -> [Cross-Attn] -> [SwiGLU FFN]
>                                              (x6 layers)
>                                                    |
>                                              Output Projection -> (B, seq_len, vocab_size)
> Parameter counts annotated next to each block.]*
>
> Caption: "The vision encoder is the only component that changes. Everything else stays identical."

#### How CNNs and Transformers See Differently

This is where the backbone choice matters most. CNNs output a 2D spatial grid (height x width), preserving the layout of the score. Transformers output a 1D sequence of patch embeddings. Both end up as a sequence of vectors, but the "spatial memory" they carry differs. [...]

```python
# How VisionEncoder handles different backbone families:

if len(features.shape) == 4:          # CNN: (B, C, H, W)
    features = features.permute(0, 2, 3, 1)  # -> (B, H, W, C)
    features = self.feature_proj(features)     # -> (B, H, W, d_model)
    features = self.pos_encoding(features)     # Add 2D spatial info
    features = features.view(B, H * W, D)      # Flatten to sequence

elif len(features.shape) == 3:        # Transformer: (B, seq_len, D)
    features = features[:, 1:, :]              # Strip CLS token
    features = self.feature_proj(features)     # -> (B, seq_len, d_model)
```

> [!TIP]
> **Why 2D positional encoding for CNNs?** When we flatten a 7x7 spatial grid into 49 tokens,
> we lose which tokens were above, below, left, or right of each other. 2D positional encoding
> restores this spatial information. Transformers don't need it because their patch embeddings
> already encode position.

#### The Swin-Tiny Padding Bug (A War Story)

One backbone required special treatment -- and taught us a lesson about the gap between theory and practice. [...]

> [!WARNING]
> **The bug we shipped (and fixed):** Swin Transformer uses windowed attention in 7x7 patches,
> with 3 patch-merge stages that halve spatial dimensions. Our initial padding only accounted
> for the patch embedding (`stride = 4 * 7 = 28`), not the merges. After 3 halvings, some
> feature maps ended up with dimensions not divisible by 7, causing a cryptic tensor mismatch
> deep inside HuggingFace's Swin code. The fix: `stride = 4 * 7 * 2^3 = 224`.

#### The Decoder: Predicting One Token at a Time

The decoder is a standard Transformer with one critical addition: cross-attention layers that let it "look at" the encoder's visual features. [...]

```python
# One decoder layer (simplified):

# 1. Self-attention: "What have I predicted so far?"
tgt = tgt + self.self_attn(self.norm1(tgt), mask=causal_mask)

# 2. Cross-attention: "Where should I look in the image?"
tgt = tgt + self.cross_attn(self.norm2(tgt), encoder_features)

# 3. Feed-forward: "Process what I've gathered"
tgt = tgt + self.ffn(self.norm3(tgt))
```

#### Teacher Forcing: Training vs. Inference

During training, we use **teacher forcing**: the decoder receives the *correct* previous token, not its own (possibly wrong) prediction. [...]

```
Training:   Input: [BOS, C4, E4, G4]  ->  Target: [C4, E4, G4, EOS]
             (ground truth fed in)          (model learns to predict next token)

Inference:  Input: [BOS]              ->  Predict: C4
            Input: [BOS, C4]          ->  Predict: E4
            Input: [BOS, C4, E4]      ->  Predict: G4
            Input: [BOS, C4, E4, G4]  ->  Predict: EOS (stop)
```

> [!NOTE]
> **Exposure bias**: During training, the decoder always sees correct history. During inference,
> it sees its own predictions -- which may be wrong. This mismatch means errors can compound.
> It's a known limitation of teacher forcing that we accept for training stability.

<!-- INTERACTIVE: Live Inference Demo ("See It Think")
     Select a sample image. Watch step-by-step: preprocessing, feature extraction (with
     backbone-selectable heatmap), autoregressive token generation with moving cross-attention
     spotlight, then audio playback of the result. All pre-computed, no GPU needed. -->

---

## 3. The Experiment: Swapping Vision Backbones {#section-3}

### 3.1 Experimental Design

The idea is simple: take one complete OMR system, keep everything constant except the vision encoder, and measure what changes. [...]

> **Figure 9: The Controlled Experiment**
>
> *[7 parallel pipelines, each with a different colored backbone feeding into the same
> gray decoder. Bracket around shared components: "Controlled variables."
> Bracket around backbones: "Independent variable."]*
>
> Caption: "Any difference in accuracy or efficiency is attributable directly to the backbone."

| Component | Configuration | Why This Value |
|-----------|--------------|----------------|
| Decoder | 6-layer Transformer, d=512, 8 heads, SwiGLU | Matches mid-range decoder; keeps focus on encoder |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01) | AdamW = Adam + decoupled weight decay; lr=3e-4 is standard for transformers |
| LR Schedule | Linear warmup (3%) + decay | Warmup prevents destructive early updates from random init |
| Batch size | 32 effective (8 x 4 accumulation) | 32 is standard; accumulation fits 8 in VRAM at a time |
| Mixed precision | FP16 via AMP | Uses 16-bit for speed, 32-bit for numerically sensitive ops. Free speedup. |
| Epochs | 10 | Matches LEGATO for fair comparison |
| Hardware | 32 GB VRAM GPUs (SLURM cluster) | |

*Table 4: Controlled settings. The "Why" column teaches readers the reasoning behind each choice.*

### 3.2 The Seven Backbones

We selected seven encoders spanning three architectural families. Each tests a specific hypothesis about what matters for music recognition. [...]

> **Figure 10: Backbone Family Taxonomy**
>
> *[2x2 grid: CNN vs Transformer on one axis, Standard vs Efficient on the other.
> Each backbone placed in its cell. Lines connect related architectures
> (ViT -> DeiT as "same arch, better pretraining").]*

#### The CNN Family

**ResNet-50** (25M encoder params): The Honda Civic of vision models -- reliable, well-understood, decades of engineering wisdom. Tests the baseline hypothesis: do standard CNNs with local receptive fields work for music's dense symbols? [...]

**EfficientNet-B0** (5M encoder params): Designed to maximize accuracy per FLOP through compound scaling. At only 5M params, tests whether a tiny encoder can suffice. [...]

**ConvNeXt-Tiny** (28M encoder params): A pure CNN that borrows design choices from Transformers (larger kernels, LayerNorm, GELU). Tests whether modern CNN tricks close the gap with ViTs. [...]

#### The Transformer Family

**ViT-Small** (22M encoder params): Splits the image into 16x16 patches and applies global self-attention. Tests whether global context (seeing the whole score at once) helps OMR. [...]

**DeiT-Small** (22M encoder params): Architecturally identical to ViT but pretrained with knowledge distillation. Tests whether better pretraining transfers to better fine-tuning on a different domain. [...]

**Swin-Tiny** (28M encoder params): Uses windowed attention (7x7 local windows) with a hierarchical feature pyramid. Tests whether local-then-global attention can match full global attention at lower cost. [...]

#### The Hybrid

**MobileViT-Small** (6M encoder params): Combines depthwise separable convolutions with Transformer blocks. Designed for mobile devices. Tests whether "best of both worlds" actually delivers -- or inherits the worst of both. [...]

| Backbone | Family | Encoder Params | Total Params | Key Feature | Hypothesis |
|----------|--------|---------------|-------------|-------------|------------|
| ResNet-50 | CNN | 25M | 61M | Residual connections | Local bias helps dense symbols |
| EfficientNet-B0 | CNN | 5M | 42M | Compound scaling, depthwise convs | 5M params is enough |
| ConvNeXt-Tiny | CNN | 28M | 65M | Modern CNN + Transformer tricks | Modern CNNs match ViTs |
| ViT-Small | Transformer | 22M | 59M | Global self-attention | Global context helps OMR |
| DeiT-Small | Transformer | 22M | 59M | ViT + distillation pretraining | Better pretraining transfers |
| Swin-Tiny | Transformer | 28M | ~65M | Windowed + hierarchical attention | Local-then-global works |
| MobileViT-Small | Hybrid | 6M | 42M | Depthwise convs + Transformer | Mobile-efficient = GPU-efficient? |

*Table 5: The seven backbones. Each tests a different hypothesis about what architectural
properties matter for reading sheet music.*

> **Figure 11: How Each Family Processes Music**
>
> *[Three panels showing the same sheet music processed by:
> (1) CNN: Sliding kernels with growing receptive fields. "local -> local -> local"
> (2) Transformer: Image split into patches, every patch attends to every other. "global from layer 1"
> (3) Hybrid: Local convolutions then Transformer blocks. "local -> global"
> Highlight a long slur spanning many patches -- show which architecture "sees" it.]*

### 3.3 The GPU Efficiency Paradox (Foreshadowing)

EfficientNet-B0 and MobileViT-Small have far fewer parameters and FLOPs than ResNet-50, yet they train *slower* on our GPUs. [...]

> [!TIP]
> **Why "efficient" is slow on GPUs**: GPUs thrive on large, dense matrix multiplications.
> Depthwise separable convolutions split one big convolution into many tiny, independent ones.
> Each tiny operation underutilizes the GPU's parallel cores while still paying the full memory
> access cost. Analogy: imagine 1000 workers (GPU cores) and a task split into 4 chunks
> (depthwise convs). 996 workers sit idle. A "wasteful" architecture that creates 1000 chunks
> runs faster because every worker stays busy.

<!-- INTERACTIVE: Backbone Comparison Card Sorter
     Before seeing results, reader drag-and-drops backbones to rank by predicted performance.
     Then reveal actual ranking. "Prediction before observation" improves learning. -->

---

## 4. Results and Analysis {#section-4}

### 4.1 Training Dynamics

Before looking at final numbers, let's examine *how* each backbone learns -- the shape of the loss curve tells us as much as the final value. [...]

> [!TIP]
> **How to read a loss curve**: If training loss drops but validation loss rises = overfitting
> (memorizing training data). If both plateau high = model lacks capacity or wrong learning rate.
> If curves are noisy = batch size may be too small.

<!-- INTERACTIVE: Training Dynamics Visualizer ("The Race")
     Interactive line chart: toggle backbones on/off, zoom into epochs, hover for exact values.
     "Play" button animates curves drawing from epoch 0 to 10.
     Group by family (CNN/Transformer/Hybrid) or show all.
     Sortable convergence table below. -->

> **Figure 12: Training and Validation Loss Curves**
>
> *[WandB-exported plot: train loss (left) and val loss (right) over 10 epochs, all 7 backbones.
> Annotations: fastest convergence, overfitting signs, ranking crossover points.]*
>
> Caption: "[TO BE COMPLETED with actual observations after training finishes.]"

### 4.2 Main Results

| Backbone | Val Loss | SER (%) | Throughput (it/s) | Peak VRAM (GB) | Params | Time/Epoch |
|----------|----------|---------|-------------------|----------------|--------|------------|
| ResNet-50 | | | 2.43 | | 61M | ~2h 30m |
| EfficientNet-B0 | | | 1.89 | | 42M | ~3h 30m |
| ConvNeXt-Tiny | | | 2.26 | | 65M | ~2h 35m |
| ViT-Small | | | 2.91 | | 59M | ~2h 15m |
| DeiT-Small | | | 2.39 | | 59M | ~2h 15m |
| Swin-Tiny | | | TBD | | ~65M | TBD |
| MobileViT-Small | | | 1.47 | | 42M | ~3h 50m |

*Table 6: Full results. [TO BE COMPLETED.] The "Throughput" and "Time/Epoch" columns already
reveal the GPU efficiency paradox: the smallest models are the slowest.*

> [!NOTE]
> **How to read SER (Symbol Error Rate)**: SER = edit distance between predicted and
> ground truth token sequences, divided by ground truth length. SER of 10% means 1 in 10
> symbols is wrong (inserted, deleted, or substituted). For practical use, aim for <5%.

<!-- INTERACTIVE: Backbone Comparison Dashboard ("The Backbone Scorecard")
     Mode A: Five sliders (accuracy, speed, VRAM, param efficiency, inference speed) weight
     priorities. Backbones reorder in real time. Presets: "16GB GPU", "Max accuracy", "Mobile deploy".
     Mode B: Scatter plot with selectable X/Y axes. Hover for details. Pareto frontier highlighted.
     Mode C: Head-to-head radar chart for any two backbones. -->

### 4.3 Key Findings

#### Finding 1: Which Backbone Family Wins?

[TO BE COMPLETED after training. Opening sentence about the best family and *why*.] [...]

> **Figure 13: Accuracy by Architecture Family**
>
> *[Grouped bar chart: x = backbone (grouped by family), y = SER (%).
> Dashed lines at family averages.]*
>
> Caption: "[TO BE COMPLETED.]"

**Takeaway**: If you only have time to try one backbone, use [TBD] because [TBD].

#### Finding 2: Parameters vs. Accuracy

The relationship between size and quality is not "bigger = better." [...]

> **Figure 14: Parameters vs. SER Scatter Plot**
>
> *[Scatter: x = total params (M), y = SER (%). Each point labeled. Trend line. Annotate outliers.]*
>
> Caption: "[TO BE COMPLETED.]"

**Takeaway**: [TBD] achieves [X]% SER with only [Y]M params -- [Z]% of the accuracy of the largest model at [W]% of the size.

#### Finding 3: The GPU Efficiency Paradox

The most counterintuitive finding: models *designed* for efficiency are the *slowest* to train. [...]

| Backbone | Encoder Params | Throughput (it/s) | Relative Speed |
|----------|---------------|-------------------|----------------|
| ViT-Small | 22M | 2.91 | Fastest |
| ResNet-50 | 25M | 2.43 | |
| DeiT-Small | 22M | 2.39 | |
| ConvNeXt-Tiny | 28M | 2.26 | |
| EfficientNet-B0 | 5M | 1.89 | |
| MobileViT-Small | 6M | 1.47 | Slowest |

*Table 7: The efficiency paradox. Models with fewer params (EfficientNet 5M, MobileViT 6M)
are 40-50% slower than models with 4x more params (ViT 22M). Depthwise separable convolutions
are memory-bound on GPUs, negating their FLOP advantage.*

**Takeaway**: "Efficient" architectures are efficient on *mobile CPUs*, not GPUs. If training on a GPU, prefer dense operations (ViT, ResNet, ConvNeXt). If deploying on mobile, the calculus flips.

#### Finding 4: Swin-Tiny's Quirks

Swin required special handling and crashed twice before we fixed the padding bug (Section 2.4). Its performance story: [...]

### 4.4 Prediction Examples

Raw error rates only tell part of the story. Let's look at what the model actually produces. [...]

> **Figure 15: Successes and Failures**
>
> *[Grid of 4-6 examples showing:
> (1) Input image
> (2) Ground truth (rendered notation or ABC text)
> (3) Model prediction (errors highlighted in red)
> Include: simple passage (success), dense chord (errors), rare symbols, long sequence.]*
>
> Caption: "Models handle simple melodies well but struggle with dense polyphony,
> rare symbols, and very long sequences."

<!-- INTERACTIVE: Attention Heatmap Viewer ("Where Does It Look?")
     Click any predicted token to see where cross-attention focused in the image.
     Backbone dropdown, layer slider (1-6), head tabs (1-8).
     Early layers show broad attention; later layers show sharp focus. -->

<!-- INTERACTIVE: Audio Comparison Player
     Two-column player: "Ground Truth" vs "Model Prediction" for the same snippet.
     Hear the errors -- a wrong note or missing rest is immediately audible. -->

---

## 5. Making It Practical: Efficiency Techniques {#section-5}

### 5.1 The Best Backbone Might Not Fit on Your GPU

The results tell us *which backbone learns best*, but not whether you can *run* it. [...]

> **Figure 16: The VRAM Wall**
>
> *[Bar chart: peak VRAM per backbone during training. Horizontal lines at 16/24/48 GB.]*
>
> Caption: "Peak training VRAM determines which backbones you can use.
> The techniques below help you push past these limits."

**The efficiency toolkit, in order of effort:**

1. **Mixed Precision (AMP)** -- already enabled in all our experiments (free speedup, ~30% VRAM reduction)
2. **LoRA** -- freeze backbone, train tiny adapters (Section 5.2)
3. **QLoRA** -- quantize frozen backbone to 4-bit + LoRA (Section 5.3)
4. **Knowledge Distillation** -- train a small model to mimic a large one (Section 5.4)
5. **Post-training Quantization** -- compress trained model for inference (Section 5.5)

### 5.2 LoRA: Fine-Tuning with 97% Fewer Parameters

What if you could freeze all 25M parameters of your encoder and train only a tiny set of adapter matrices? [...]

> [!TIP]
> **The Analogy**: LoRA is like putting a thin, adjustable lens in front of a camera.
> The camera (backbone) stays untouched -- the lens (adapter) subtly redirects what it sees.
> Original weight matrix: 512x512 = 262,144 params. LoRA rank 8: (512x8)+(8x512) = 8,192 params. A 32x reduction.

> **Figure 17: LoRA Explained Visually**
>
> *[Large weight matrix W (frozen, gray) with LoRA decomposition: W + BA.
> B is tall-thin, A is short-wide. Rank controls the "thin" dimension.
> Concrete numbers for rank 4/8/16.]*
>
> Caption: "LoRA trains two small matrices per layer instead of updating the full weight."

| Configuration | Val Loss | SER (%) | Trainable Params | Peak VRAM | vs. Full Fine-Tune |
|---------------|----------|---------|-----------------|-----------|-------------------|
| Full fine-tune | | | [all] | | baseline |
| Frozen (no adapters) | | | [decoder only] | | |
| LoRA rank 4 | | | | | |
| LoRA rank 8 | | | | | |
| LoRA rank 16 | | | | | |

*Table 8: LoRA rank ablation on [TBD backbone]. [TO BE COMPLETED.]*

<!-- INTERACTIVE: LoRA Rank Explorer ("How Small Can You Go?")
     Slider: LoRA rank 0 (full fine-tune) to 64.
     Updates in real time: trainable params bar, VRAM savings, accuracy curve.
     Animated matrix factorization: W + BA matrices grow/shrink with rank. -->

### 5.3 QLoRA: 4-Bit Backbone + LoRA

QLoRA goes further: compress the frozen backbone from 16-bit to 4-bit, *then* add LoRA adapters. [...]

> [!TIP]
> **What is 4-bit quantization?** Each parameter normally uses 16 or 32 bits. 4-bit
> quantization packs each into just 4 bits -- an 8x memory reduction. The backbone
> is frozen anyway (we're not training it), so minor precision loss is acceptable.

| Config | Val Loss | SER (%) | Peak VRAM | VRAM vs. Full |
|--------|----------|---------|-----------|---------------|
| Full fine-tune (FP16) | | | | baseline |
| LoRA rank 8 (FP16) | | | | |
| QLoRA rank 8 (4-bit) | | | | |

*Table 9: QLoRA vs LoRA vs full fine-tuning. [TO BE COMPLETED.]*

### 5.4 Knowledge Distillation: Teaching a Small Model

Instead of making a large model *cheaper to run*, distillation trains a genuinely *small* model to mimic the large one. [...]

> [!TIP]
> **The Analogy**: A master musician teaches a student not just "play this note" (hard labels)
> but "notice how C4 and C#4 sound similar, and how the rhythm connects to the previous phrase"
> (soft labels). The master's probability distribution contains richer information than a
> sheet of correct answers.

> **Figure 18: Distillation Setup**
>
> *[Teacher (large, frozen) and Student (small, trainable) process the same image.
> Loss = alpha * CE(student, ground_truth) + (1-alpha) * KL(student || teacher)
> Teacher's output shown as smooth curve; ground truth as spike.]*
>
> Caption: "The student learns from both the ground truth and the teacher's 'dark knowledge'
> about which symbols look similar."

| Config | Val Loss | SER (%) | Params | Inference Speed |
|--------|----------|---------|--------|----------------|
| Teacher (best backbone) | | | | |
| Student (from scratch) | | | | |
| Student (distilled) | | | | |

*Table 10: Distillation results. [TO BE COMPLETED.] Does a distilled small model approach a large model's accuracy?*

### 5.5 Post-Training Quantization

The simplest technique: convert a trained model's weights from FP32 to INT8 *after* training. No retraining needed. [...]

> [!TIP]
> **The Analogy**: Painting a sunset with 16 million colors (32-bit) vs 256 colors (8-bit).
> For most of the sky, 256 is indistinguishable. But in subtle gradients, you may see banding.

| Config | SER (%) | Model Size (MB) | Inference Speed | SER Change |
|--------|---------|-----------------|----------------|------------|
| FP32 | | | | baseline |
| FP16 | | | | |
| INT8 (dynamic) | | | | |

*Table 11: Post-training quantization impact. [TO BE COMPLETED.]*

### 5.6 Decision Tree: What Should You Do?

> **Figure 19: Efficiency Decision Tree**
>
> *[Flowchart:
> "What's your goal?"
>   -> "Best accuracy" -> Full fine-tune, best backbone
>   -> "Limited training VRAM" ->
>       "<16GB" -> QLoRA on smallest backbone
>       "16-24GB" -> LoRA rank 8 on mid-size backbone
>       "24-48GB" -> Full fine-tune or LoRA on best backbone
>   -> "Fast inference / mobile" -> Small backbone + distillation + INT8]*

---

## 6. The VRAM Budget Guide {#section-6}

### 6.1 Where Does the Memory Go?

The model's parameter count is a surprisingly poor predictor of actual VRAM usage. [...]

> **Figure 20: VRAM Breakdown**
>
> *[Stacked bar chart for 3 representative backbones:
> Model params | Activations | Gradients | Optimizer states (2x for AdamW)
> Show how activations often dominate.]*
>
> Caption: "Parameters are often the smallest VRAM component. Activations and
> optimizer states dominate. AMP halves parameter storage; LoRA eliminates
> backbone gradients and optimizer states entirely."

> [!TIP]
> **VRAM Analogy**: Think of GPU memory as a kitchen counter. Model params = appliances
> (always there). Activations = ingredients spread out while cooking. Gradients = dirty
> dishes accumulating. Optimizer states = recipe books open beside you. AMP = smaller plates.
> LoRA = freezing the appliances (no dirty dishes from them).

<!-- INTERACTIVE: VRAM Budget Calculator ("Will It Fit?")
     Three sliders: GPU memory (8-80GB), batch size (1-64), image height (128-320px).
     Stacked bar per backbone showing VRAM breakdown.
     Red line = GPU limit. Green = fits, red = doesn't.
     Toggles: AMP, LoRA, QLoRA.
     Recommendation cards: "Best that fits", "Most efficient", "With LoRA". -->

### 6.2 The 16 GB Tier (RTX 4060 / 4070)

The most common consumer GPU. [...]

| Setting | Recommendation |
|---------|---------------|
| Backbone | [TBD] |
| Training mode | [TBD: LoRA / QLoRA] |
| Batch size | [TBD] |
| Expected SER | [TBD] |

*Table 12: 16 GB configuration. [TO BE COMPLETED.]*

### 6.3 The 24 GB Tier (RTX 3090 / 4090)

Significantly more room -- enough for most backbones with full fine-tuning. [...]

| Setting | Recommendation |
|---------|---------------|
| Backbone | [TBD] |
| Training mode | [TBD] |
| Batch size | [TBD] |
| Expected SER | [TBD] |

*Table 13: 24 GB configuration. [TO BE COMPLETED.]*

### 6.4 The 48 GB Tier (A40 / A6000)

Every backbone fits. The question shifts from "what can I run?" to "what gives diminishing returns?" [...]

| Setting | Recommendation |
|---------|---------------|
| Backbone | [TBD] |
| Training mode | [TBD] |
| Batch size | [TBD] |
| Expected SER | [TBD] |

*Table 14: 48 GB configuration. [TO BE COMPLETED.]*

### 6.5 The Pareto Frontier

> **Figure 21: Accuracy vs. VRAM Pareto Frontier**
>
> *[Scatter: x = peak VRAM (GB), y = SER (%, inverted). Each point = one (backbone, technique) config.
> Color by technique: blue=full, orange=LoRA, red=QLoRA, green=distilled.
> Pareto frontier drawn through non-dominated points.]*
>
> Caption: "[TO BE COMPLETED.]"

> **Figure 22: Accuracy vs. Throughput Pareto Frontier**
>
> *[Same format, x = throughput instead of VRAM.]*

### 6.6 On-Device: The Mobile OMR Use Case

If your goal is scanning sheet music with a phone, the calculus flips. [...]

> [!NOTE]
> **Mobile deployment is a different optimization.** On mobile, you care about inference
> latency, model size, and battery -- not training VRAM. MobileViT, the slowest to *train*
> on a GPU, may be the fastest to *run* on a phone's neural accelerator, because depthwise
> convolutions are exactly what mobile NPUs are optimized for.

---

## 7. Conclusion and What's Next {#section-7}

### 7.1 What We Learned

This project asked: if you swap the eyes of a music-reading AI, how much does it matter? [...]

**Key takeaways** (numbered for reference):

1. [TBD: Best backbone and why]
2. [TBD: Parameter count vs accuracy relationship]
3. "Efficient" architectures are efficient on mobile CPUs, not GPUs
4. [TBD: LoRA recovers X% of accuracy at Y% of VRAM]
5. [TBD: Sweet spot recommendation]

> **Figure 23: Summary Infographic**
>
> *[Clean infographic with the 5 key takeaways, each with an icon and one-sentence summary.]*

### 7.2 Limitations

| Limitation | Impact | Future Work |
|-----------|--------|-------------|
| Synthetic data only | May not generalize to real scans | Evaluate on MUSCIMA++, real-world scans |
| Single dataset | Results may not transfer | Test on other notation styles |
| Monophonic focus | Polyphonic scores are harder | Extend to multi-voice OMR |
| Fixed decoder | Different decoder could change rankings | Ablate decoder size |
| Single run per backbone | Small differences may not be significant | Multiple seeds + confidence intervals |
| No beam search | Greedy decoding leaves accuracy on the table | Implement beam search |

*Table 15: Honest limitations. Being explicit about what we did NOT test matters.*

### 7.3 Using Our Model on Your Sheet Music

The entire post is about training, but your real question is: can I scan my sheet music? [...]

```bash
# Quick-start: inference on your own image
git clone https://github.com/[USERNAME]/music_recognition.git
cd music_recognition && pip install -r requirements.txt

python -m src.inference --image your_score.png --checkpoint networks/checkpoints/deit_small/best.pt
# Output: ABC notation -> render to PDF or play as MIDI
```

### 7.4 Reproducing Our Results

| Resource | Link |
|----------|------|
| Code | [GitHub repo TBD] |
| WandB dashboard | [TBD] |
| Model checkpoints | [TBD] |
| Dataset | `guangyangmusic/PDMX-Synth` |
| Tokenizer | `guangyangmusic/legato` |
| Configs | `configs/backbones/*.yaml` |

*Table 16: Everything needed to reproduce or extend this work.*

<!-- INTERACTIVE: OMR Configurator ("Build Your Own")
     Capstone interactive: select backbone, decoder size, efficiency technique, target hardware.
     See estimated params, VRAM, accuracy, training time.
     "Generate Config" button downloads a ready-to-use YAML file. -->

---

## Appendix A: Metric Definitions

| Metric | Definition | Good Value |
|--------|-----------|------------|
| Val Loss | Cross-entropy per token (lower = better) | < 0.3 |
| SER | Edit distance / ground truth length | < 5% |
| CER | Character-level edit distance / length | < 3% |
| Sequence Accuracy | % of sequences predicted perfectly | > 50% |
| Throughput | Training iterations per second | Higher = better |
| Peak VRAM | Max GPU memory during training | Depends on hardware |

*Table 17: All metrics used in this post.*

## Appendix B: Full Training Configuration

```yaml
# Annotated reference config (configs/backbones/deit_small.yaml)
model:
  type: "MusicTrOCR"
  params:
    vision_model_name: "facebook/deit-small-patch16-224"  # HuggingFace model ID
    d_model: 512          # Representation width (all layers)
    n_heads: 8            # Parallel attention perspectives
    n_decoder_layers: 6   # Depth of the decoder
    d_ff: 2048            # Feed-forward hidden dimension (4x d_model)
    max_seq_len: 2048     # Maximum output sequence length
    dropout: 0.1          # Regularization

data:
  dataset_id: "guangyangmusic/PDMX-Synth"
  tokenizer_id: "guangyangmusic/legato"
  img_height: 224         # Fixed height, width scaled proportionally

training:
  batch_size: 8                       # Per-GPU batch size
  gradient_accumulation_steps: 4      # Effective batch = 8 * 4 = 32
  epochs: 10
  optimizer:
    type: "AdamW"                     # Adam + decoupled weight decay
    learning_rate: 0.0003             # 3e-4, standard for transformers
    weight_decay: 0.01                # L2 regularization
    betas: [0.9, 0.99]               # Momentum / adaptive LR coefficients
  scheduler:
    type: "LinearWarmup"
    warmup_ratio: 0.03                # 3% of steps at increasing LR

hardware:
  device: "auto"
  mixed_precision: true               # FP16 via AMP
```
