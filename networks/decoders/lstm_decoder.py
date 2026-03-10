"""LSTM decoder for the OMR CRNN ablation study."""

import torch.nn as nn
from networks.decoders import BaseDecoder


class LSTMDecoder(BaseDecoder):
    """Bidirectional LSTM decoder with two stacked layers."""

    def __init__(self, input_size=2048, hidden_size=256, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
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
        Process encoder features through LSTM layers.

        Args:
            x: Tensor of shape (batch, seq_len, 2048)

        Returns:
            Tensor of shape (batch, seq_len, 512)
        """
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x
