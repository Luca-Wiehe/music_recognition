"""
Music-TrOCR: Vision Encoder + Autoregressive Text Decoder for Optical Music Recognition

This module implements a transformer-based architecture for converting sheet music images
into sequences of music notation tokens. The architecture follows the TrOCR paradigm with:
- Vision Encoder: Pre-trained image backbone for visual feature extraction
- Text Decoder: Autoregressive transformer decoder for token-by-token prediction
- Cross-attention: Dynamic region focusing during sequence generation

Architecture Requirements:
1. Region focusing via cross-attention mechanisms
2. Token-by-token autoregressive prediction
3. d1B parameters total
4. Pre-trained HuggingFace backbone integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from transformers import AutoModel, AutoConfig


class PositionalEncoding2D(nn.Module):
    """
    2D Positional encoding for spatial image features.
    Adds learned position embeddings to preserve spatial relationships.
    """

    def __init__(self, d_model: int, max_height: int = 128, max_width: int = 512):
        super().__init__()
        self.d_model = d_model

        # Create learnable position embeddings for height and width
        self.height_embed = nn.Parameter(torch.randn(max_height, d_model // 2))
        self.width_embed = nn.Parameter(torch.randn(max_width, d_model // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (batch_size, height, width, d_model)
        Returns:
            Position-encoded features of same shape
        """
        batch_size, height, width, d_model = x.shape

        # Get position embeddings for current spatial dimensions
        h_embed = self.height_embed[:height].unsqueeze(1).expand(-1, width, -1)  # (H, W, d_model//2)
        w_embed = self.width_embed[:width].unsqueeze(0).expand(height, -1, -1)   # (H, W, d_model//2)

        # Concatenate height and width embeddings
        pos_embed = torch.cat([h_embed, w_embed], dim=-1)  # (H, W, d_model)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, H, W, d_model)

        return x + pos_embed


class VisionEncoder(nn.Module):
    """
    Vision encoder using pre-trained backbone.
    Extracts spatial features from sheet music images.

    Handles two output families:
    - CNN-style (B, C, H, W): ResNet, EfficientNet, ConvNeXt, MobileViT
    - Sequence-style (B, seq_len, D): ViT, DeiT, Swin
    """

    def __init__(self,
                 model_name: str = "microsoft/convnext-tiny-224",
                 d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.model_name = model_name.lower()

        # Load pre-trained vision model
        self.backbone = AutoModel.from_pretrained(model_name)

        # Detect output dimension and backbone family from config
        config = AutoConfig.from_pretrained(model_name)
        backbone_dim = self._detect_backbone_dim(config)

        # Detect backbone family for input handling
        model_type = getattr(config, 'model_type', '').lower()
        self._is_vit = model_type in ('vit', 'deit')
        self._is_swin = model_type == 'swin'
        # Swin needs input padded so that spatial dims stay >= window_size
        # at every stage.  After patch-embed (÷patch_size) there are
        # (num_stages-1) patch-merge layers, each halving spatial dims.
        # Total downsampling = patch_size * 2^(num_stages-1).
        # Every intermediate resolution must be divisible by window_size,
        # so the input must be divisible by total_downsample * window_size.
        if self._is_swin:
            patch_size = getattr(config, 'patch_size', 4)
            window_size = getattr(config, 'window_size', 7)
            num_stages = len(getattr(config, 'depths', [2, 2, 6, 2]))
            self._swin_stride = patch_size * (2 ** (num_stages - 1)) * window_size

        print(f"Loaded pretrained {model_name} with output dim {backbone_dim}")

        # Project backbone features to model dimension
        self.feature_proj = nn.Linear(backbone_dim, d_model)

        # 2D positional encoding (only used for CNN-style spatial outputs)
        self.pos_encoding = PositionalEncoding2D(d_model)

    @staticmethod
    def _detect_backbone_dim(config) -> int:
        """Detect the output feature dimension from a HuggingFace model config."""
        # MobileViT: actual output comes from neck, not hidden_sizes
        if hasattr(config, 'neck_hidden_sizes'):
            return config.neck_hidden_sizes[-1]
        # EfficientNet: uses hidden_dim
        if hasattr(config, 'hidden_dim'):
            return config.hidden_dim
        # ResNet, ConvNeXt: uses hidden_sizes (list per stage)
        if hasattr(config, 'hidden_sizes'):
            return config.hidden_sizes[-1]
        # ViT, DeiT, Swin: uses hidden_size (single int)
        if hasattr(config, 'hidden_size'):
            return config.hidden_size
        return 768  # fallback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (batch_size, channels, height, width)
        Returns:
            Encoded features (batch_size, seq_len, d_model)
        """
        batch_size = x.shape[0]

        # Convert grayscale to RGB for pretrained models
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Swin: pad H and W to multiples of window stride (patch_size * window_size)
        if self._is_swin:
            _, _, H, W = x.shape
            stride = self._swin_stride
            pad_h = (stride - H % stride) % stride
            pad_w = (stride - W % stride) % stride
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right and bottom

        # ViT/DeiT: pass interpolate_pos_encoding to handle variable-size input
        if self._is_vit:
            outputs = self.backbone(x, interpolate_pos_encoding=True)
        else:
            outputs = self.backbone(x)

        # Extract features (different models have different output structures)
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        else:
            features = outputs[0] if isinstance(outputs, tuple) else outputs

        if len(features.shape) == 4:
            # CNN-style spatial output: (B, C, H, W)
            # ResNet, EfficientNet, ConvNeXt, MobileViT
            if features.shape[1] > features.shape[-1]:
                features = features.permute(0, 2, 3, 1)  # -> (B, H, W, C)
            # features is (B, H, W, C)
            features = self.feature_proj(features)      # (B, H, W, d_model)
            features = self.pos_encoding(features)      # (B, H, W, d_model)
            B, H, W, D = features.shape
            features = features.view(B, H * W, D)       # (B, H*W, d_model)

        elif len(features.shape) == 3:
            # Sequence-style output: (B, seq_len, D)
            # ViT, DeiT, Swin
            # ViT/DeiT include a CLS token at position 0 — strip it
            # (Swin does not have CLS, but stripping position 0 loses one 7x7 patch)
            # Use a heuristic: if seq_len is not a perfect square, strip first token
            seq_len = features.shape[1]
            sqrt = int(seq_len ** 0.5)
            if sqrt * sqrt != seq_len:
                features = features[:, 1:, :]  # strip CLS token

            features = self.feature_proj(features)      # (B, seq_len, d_model)

        return features


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: gate_proj(x) * silu(up_proj(x)), then down_proj."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerDecoderLayer(nn.Module):
    """
    Pre-norm transformer decoder layer with self-attention, cross-attention, and SwiGLU FFN.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Self-attention (causal/masked)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Cross-attention to encoder features
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # SwiGLU feed-forward network
        self.ffn = SwiGLU(d_model, d_ff, dropout)

        # Pre-norm: LayerNorm applied before each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence (B, tgt_len, d_model)
            memory: Encoder memory/features (B, src_len, d_model)
            tgt_mask: Causal mask for target sequence
            tgt_key_padding_mask: Padding mask for target
            memory_key_padding_mask: Padding mask for encoder features
        """
        # Pre-norm self-attention with residual
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2,
                                attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        # Pre-norm cross-attention with residual
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.cross_attn(tgt2, memory, memory,
                                  key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        # Pre-norm SwiGLU FFN with residual
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.ffn(tgt2)

        return tgt


class MusicTransformerDecoder(nn.Module):
    """
    Autoregressive transformer decoder for music token prediction.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings for sequence
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm (required for pre-norm architecture)
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with appropriate scaling"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask to prevent attention to future tokens"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: Target token indices (B, tgt_len)
            memory: Encoder features (B, src_len, d_model)
            tgt_key_padding_mask: Padding mask for target sequence (B, tgt_len)
        Returns:
            Logits over vocabulary (B, tgt_len, vocab_size)
        """
        batch_size, tgt_len = tgt.shape

        # Token embeddings
        tgt_emb = self.token_embedding(tgt)  # (B, tgt_len, d_model)

        # Add positional embeddings
        pos_emb = self.pos_embedding[:tgt_len].unsqueeze(0).expand(batch_size, -1, -1)
        tgt_emb = tgt_emb + pos_emb

        # Generate causal mask
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(tgt.device)

        # Pass through decoder layers
        output = tgt_emb
        for layer in self.layers:
            output = layer(output, memory,
                          tgt_mask=tgt_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask)

        # Final norm (pre-norm requires this before output projection)
        output = self.final_norm(output)

        # Project to vocabulary
        logits = self.output_proj(output)  # (B, tgt_len, vocab_size)

        return logits


class MusicTrOCR(nn.Module):
    """
    Main Music-TrOCR model combining vision encoder and autoregressive text decoder.

    This model follows the TrOCR architecture for optical music recognition:
    1. Vision encoder extracts spatial features from sheet music images
    2. Transformer decoder generates music tokens autoregressively with cross-attention

    Uses vocabulary directly without token shifting:
    - Special token IDs (PAD, BOS, EOS) are passed from the vocabulary
    - No artificial vocabulary offset - works directly with dataset token IDs
    """

    def __init__(self,
                 vocab_size: int,
                 pad_token_id: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 vision_model_name: str = "microsoft/convnext-tiny-224",
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()

        # Use vocabulary token IDs directly (no shifts!)
        self.PAD_TOKEN_ID = pad_token_id
        self.START_TOKEN_ID = bos_token_id
        self.END_TOKEN_ID = eos_token_id
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            d_model=d_model
        )

        # Transformer decoder
        self.decoder = MusicTransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        print(f"MusicTrOCR initialized with {self.count_parameters():,} parameters")

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature representations"""
        return self.vision_encoder(images)

    def decode_model_tokens_to_dataset(self, model_tokens: torch.Tensor) -> torch.Tensor:
        """
        No conversion needed - model uses vocabulary directly.

        Args:
            model_tokens: Tensor with vocabulary indices

        Returns:
            Same tensor (no conversion needed)
        """
        return model_tokens

    def prepare_targets(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare target sequences for training with teacher forcing.
        Handles sequences that already have BOS/EOS tokens.

        Args:
            targets: Token indices from dataset (B, seq_len)
                    - Already includes BOS at start and EOS at end
                    - PAD_TOKEN_ID: padding

        Returns:
            decoder_input: Input to decoder - targets without last token
            decoder_target: Target for loss computation - targets without first token
        """
        batch_size, seq_len = targets.shape
        device = targets.device

        # targets already have [BOS, token1, token2, ..., tokenN, EOS]
        # decoder_input: [BOS, token1, token2, ..., tokenN] (all but last)
        # decoder_target: [token1, token2, ..., tokenN, EOS] (all but first)

        decoder_input = targets[:, :-1]  # Remove last token (EOS for input)
        decoder_target = targets[:, 1:]  # Remove first token (BOS for target)

        return decoder_input, decoder_target

    def forward(self,
                images: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (teacher forcing) or inference.

        Args:
            images: Input images (B, C, H, W)
            targets: Target sequences for training (B, seq_len), None for inference

        Returns:
            Dictionary with logits and optionally loss
        """
        # Encode images
        memory = self.encode_image(images)  # (B, src_len, d_model)

        if targets is not None:
            # Training mode with teacher forcing
            decoder_input, decoder_target = self.prepare_targets(targets)

            # Create padding mask for decoder input
            tgt_key_padding_mask = (decoder_input == self.PAD_TOKEN_ID)

            # Forward through decoder
            logits = self.decoder(decoder_input, memory, tgt_key_padding_mask)

            return {
                'logits': logits,
                'decoder_input': decoder_input,
                'decoder_target': decoder_target,
                'tgt_key_padding_mask': tgt_key_padding_mask
            }
        else:
            # Inference mode - implement later in generate() method
            raise NotImplementedError("Use generate() method for inference")

    def generate(self,
                 images: torch.Tensor,
                 max_length: int = 512,
                 temperature: float = 1.0,
                 do_sample: bool = False) -> torch.Tensor:
        """
        Autoregressive generation for inference.

        Args:
            images: Input images (B, C, H, W)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated sequences (B, generated_len)
        """
        self.eval()
        batch_size = images.shape[0]
        device = images.device

        # Encode images once
        memory = self.encode_image(images)  # (B, src_len, d_model)

        # Initialize with start tokens
        sequences = torch.full((batch_size, 1), self.START_TOKEN_ID, device=device)

        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Create padding mask (no padding in current sequences since we're generating)
            tgt_key_padding_mask = torch.zeros(batch_size, sequences.shape[1],
                                             device=device, dtype=torch.bool)

            # Forward pass
            logits = self.decoder(sequences, memory, tgt_key_padding_mask)

            # Get logits for the last position
            next_token_logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Sample or select greedily
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)

            # Append to sequences
            sequences = torch.cat([sequences, next_tokens], dim=1)

            # Check if all sequences have generated END token
            if (next_tokens == self.END_TOKEN_ID).all():
                break

        return sequences

    def training_step(self, batch, device, config=None, epoch=None, batch_idx=None,
                      scaler=None, accumulation_steps=1):
        """
        Forward + backward for one micro-batch (supports AMP and gradient accumulation).

        The caller (train_epoch) is responsible for optimizer.zero_grad / step.

        Args:
            batch: Tuple of (images, targets) from dataloader
            device: Device to run computation on
            config: Training configuration dict
            epoch: Current epoch (for debug output)
            batch_idx: Current batch index (for debug output)
            scaler: Optional GradScaler for mixed precision
            accumulation_steps: Number of micro-batches per optimizer step (for loss scaling)

        Returns:
            Loss value (unscaled, for logging)
        """
        self.train()

        images, targets = batch
        images, targets = images.to(device), targets.to(device)

        use_amp = scaler is not None

        # Forward pass (with optional autocast)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = self.forward(images, targets)

            logits = outputs['logits']
            decoder_target = outputs['decoder_target']

            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = decoder_target.reshape(-1)

            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.PAD_TOKEN_ID)
            scaled_loss = loss / accumulation_steps

        # Debug output if enabled
        if (config and epoch is not None and batch_idx is not None and
            config.get('logging', {}).get('verbose', False)):
            from src.utils.debug_utils import print_debug_info
            print_debug_info(logits, decoder_target, loss, self, epoch, batch_idx)

        # Backward pass (accumulates gradients)
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss

    def validation_step(self, batch, device, config=None, epoch=None, batch_idx=None,
                        use_amp=False):
        """
        Perform a validation step (supports AMP).

        Args:
            batch: Tuple of (images, targets) from dataloader
            device: Device to run computation on
            config: Training configuration dict
            epoch: Current epoch (for debug output)
            batch_idx: Current batch index (for debug output)
            use_amp: Whether to use automatic mixed precision

        Returns:
            Loss tensor
        """
        self.eval()

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            outputs = self.forward(images, targets)

            logits = outputs['logits']
            decoder_target = outputs['decoder_target']

            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = decoder_target.reshape(-1)

            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.PAD_TOKEN_ID)

            if (config and epoch is not None and batch_idx is not None and
                config.get('logging', {}).get('verbose', False)):
                from src.utils.debug_utils import print_debug_info
                print_debug_info(logits, decoder_target, loss, self, epoch, batch_idx, phase="VALIDATION")

        return loss

    def set_optimizer(self, hparams):
        """Set optimizer according to hyperparameters"""
        optim_hparams = hparams["optimizer"]

        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=optim_hparams["learning_rate"],
            weight_decay=optim_hparams["weight_decay"]
        )
