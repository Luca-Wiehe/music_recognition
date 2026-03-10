"""Transformer decoder for the OMR CRNN ablation study."""

import math
import torch
import torch.nn as nn
from networks.decoders import BaseDecoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding module."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor of shape (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class TransformerDecoder(BaseDecoder):
    """Transformer-based decoder using bidirectional self-attention."""

    def __init__(
        self,
        input_size=2048,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_size, d_model)

        # Sinusoidal positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000, dropout=dropout)

        # Transformer encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @property
    def output_size(self) -> int:
        """Returns the output feature size."""
        return self.d_model

    def forward(self, x):
        """
        Process encoder features through Transformer.

        Args:
            x: Tensor of shape (batch, seq_len, 2048)

        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        # Project input to model dimension
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        return x
