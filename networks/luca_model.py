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
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/convnext-tiny-224",
                 d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # Load pre-trained vision model
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get the output dimension of the backbone
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, 'hidden_size'):
            backbone_dim = config.hidden_size
        elif hasattr(config, 'hidden_sizes'):
            backbone_dim = config.hidden_sizes[-1]  # Use last layer
        else:
            backbone_dim = 768  # Default fallback
                
        print(f"Loaded pretrained {model_name} with output dim {backbone_dim}")
            
        # Project backbone features to model dimension
        self.feature_proj = nn.Linear(backbone_dim, d_model)
        
        # 2D positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (batch_size, channels, height, width)
        Returns:
            Encoded features (batch_size, seq_len, d_model) where seq_len = H*W after pooling
        """
        batch_size = x.shape[0]
        
        # Convert grayscale to RGB for pretrained models
        if x.shape[1] == 1:  
            x = x.repeat(1, 3, 1, 1)
            
        outputs = self.backbone(x)
        
        # Extract features (different models have different output structures)
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        else:
            # Fallback to first output
            features = outputs[0] if isinstance(outputs, tuple) else outputs
            
        # Handle different feature map shapes
        if len(features.shape) == 4:  # (B, H, W, C) or (B, C, H, W)
            # For ConvNeXt and similar models, output is typically (B, C, H, W)
            # Check if channels dimension is likely to be the large one
            if features.shape[1] > features.shape[-1]:  # Likely (B, C, H, W)
                features = features.permute(0, 2, 3, 1)  # -> (B, H, W, C)
            # features is now (B, H, W, C)
        elif len(features.shape) == 3:  # Already flattened (B, seq_len, C)
            # Reshape to spatial format - need to infer H, W
            seq_len, hidden_dim = features.shape[1], features.shape[2]
            # Assume square-ish feature map
            feat_size = int(seq_len ** 0.5)
            features = features.view(batch_size, feat_size, feat_size, hidden_dim)
            
        # Project to model dimension
        features = self.feature_proj(features)  # (B, H, W, d_model)
        
        # Add 2D positional encoding
        features = self.pos_encoding(features)  # (B, H, W, d_model)
        
        # Flatten spatial dimensions for transformer
        batch_size, height, width, d_model = features.shape
        features = features.view(batch_size, height * width, d_model)  # (B, H*W, d_model)
        
        return features


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with self-attention and cross-attention.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Self-attention (causal/masked)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention to encoder features
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
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
        # Self-attention with residual connection
        tgt2, _ = self.self_attn(tgt, tgt, tgt, 
                                attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        
        # Cross-attention with residual connection  
        tgt2, cross_attn_weights = self.cross_attn(tgt, memory, memory,
                                                  key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        
        # Feed-forward with residual connection
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))
        
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
        
        # Project to vocabulary
        logits = self.output_proj(output)  # (B, tgt_len, vocab_size)
        
        return logits


class MusicTrOCR(nn.Module):
    """
    Main Music-TrOCR model combining vision encoder and autoregressive text decoder.
    
    This model follows the TrOCR architecture for optical music recognition:
    1. Vision encoder extracts spatial features from sheet music images
    2. Transformer decoder generates music tokens autoregressively with cross-attention
    
    Uses BeKern vocabulary directly without token shifting:
    - Special token IDs (PAD, BOS, EOS) are passed from the BeKern vocabulary
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
        
        # Use BeKern vocabulary token IDs directly (no shifts!)
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
        No conversion needed - model uses BeKern vocabulary directly.
        
        Args:
            model_tokens: Tensor with BeKern vocabulary indices
            
        Returns:
            Same tensor (no conversion needed)
        """
        return model_tokens
        
    def prepare_targets(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare target sequences for training with teacher forcing.
        Handles sequences that already have BOS/EOS tokens (SMT-style).
        
        Args:
            targets: BeKern token indices from dataset (B, seq_len)
                    - Already includes BOS at start and EOS at end (SMT format)
                    - PAD_TOKEN_ID: padding 
                    
        Returns:
            decoder_input: Input to decoder - targets without last token
            decoder_target: Target for loss computation - targets without first token
        """
        batch_size, seq_len = targets.shape
        device = targets.device
        
        # SMT-style: targets already have [BOS, token1, token2, ..., tokenN, EOS]
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
        
    def training_step(self, batch, device, optimizer, config=None, epoch=None, batch_idx=None):
        """
        Perform a single training step.
        
        Args:
            batch: Tuple of (images, targets) from dataloader
            device: Device to run computation on
            optimizer: Optimizer instance
            config: Training configuration dict
            epoch: Current epoch (for debug output)
            batch_idx: Current batch index (for debug output)
            
        Returns:
            Loss tensor
        """
        self.train()
        
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.forward(images, targets)
        
        # Compute cross-entropy loss
        logits = outputs['logits']  # (B, seq_len, vocab_size)
        decoder_target = outputs['decoder_target']  # (B, seq_len)
        
        # Reshape for loss computation (ensure contiguity for sliced tensors)
        logits_flat = logits.contiguous().view(-1, self.vocab_size)  # (B*seq_len, vocab_size)
        targets_flat = decoder_target.contiguous().view(-1)  # (B*seq_len,)
        
        # Compute loss (ignore padding tokens)
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.PAD_TOKEN_ID)
        
        # Debug output if enabled
        if (config and epoch is not None and batch_idx is not None and 
            config.get('logging', {}).get('verbose', False)):
            from utils.debug_utils import print_debug_info
            print_debug_info(logits, decoder_target, loss, self, epoch, batch_idx)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config and config.get('training', {}).get('grad_clip_norm'):
            torch.nn.utils.clip_grad_norm_(self.parameters(), config['training']['grad_clip_norm'])
        
        # Optimizer step
        optimizer.step()
        
        return loss
        
    def validation_step(self, batch, device, config=None, epoch=None, batch_idx=None):
        """
        Perform a validation step.
        
        Args:
            batch: Tuple of (images, targets) from dataloader
            device: Device to run computation on
            config: Training configuration dict
            epoch: Current epoch (for debug output)
            batch_idx: Current batch index (for debug output)
            
        Returns:
            Loss tensor
        """
        self.eval()
        
        with torch.no_grad():
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = self.forward(images, targets)
            
            # Compute loss
            logits = outputs['logits']
            decoder_target = outputs['decoder_target']
            
            logits_flat = logits.contiguous().view(-1, self.vocab_size)
            targets_flat = decoder_target.contiguous().view(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.PAD_TOKEN_ID)
            
            # Debug output if enabled
            if (config and epoch is not None and batch_idx is not None and 
                config.get('logging', {}).get('verbose', False)):
                from utils.debug_utils import print_debug_info
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