"""GRU decoder for the OMR CRNN ablation study."""

import torch.nn as nn
from networks.decoders import BaseDecoder


class GRUDecoder(BaseDecoder):
    """Bidirectional GRU decoder with two stacked layers."""

    def __init__(self, input_size=2048, hidden_size=256, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size

        self.gru1 = nn.GRU(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.gru2 = nn.GRU(
            hidden_size * 2,
            hidden_size,
            bidirectional=True,
            batch_first=True
        )

    @property
    def output_size(self) -> int:
        """Returns the output feature size."""
        return self.hidden_size * 2

    def forward(self, x):
        """
        Process encoder features through GRU layers.

        Args:
            x: Tensor of shape (batch, seq_len, 2048)

        Returns:
            Tensor of shape (batch, seq_len, 512)
        """
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x
