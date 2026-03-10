"""
Decoder ablation modules for the OMR CRNN model.

Provides a shared CNN encoder, base decoder interface, and an AblationModel
that combines any encoder+decoder pair for CTC-based training.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SharedEncoder(nn.Module):
    """CNN encoder identical to the conv_block in MonophonicModel."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

    def forward(self, x):
        # Output shape: [batch_size, 256, H', W']
        x = self.conv_block(x)
        # Reshape to [batch_size, width, 256*height] for decoder
        x = x.view(x.shape[0], x.shape[3], x.shape[1] * x.shape[2])
        return x


class BaseDecoder(ABC, nn.Module):
    """Abstract base class for all decoder variants."""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Size of the decoder output features."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process encoder features.

        Args:
            x: Tensor of shape (batch, seq_len, 2048)

        Returns:
            Tensor of shape (batch, seq_len, output_size)
        """
        ...


class AblationModel(nn.Module):
    """Model combining a shared encoder with an interchangeable decoder."""

    def __init__(self, hparams, output_size, decoder: BaseDecoder,
                 encoder=None, in_channels=1):
        super().__init__()
        self.hparams_dict = hparams
        self.encoder = encoder or SharedEncoder(in_channels=in_channels)
        self.decoder = decoder

        self.output_block = nn.Sequential(
            nn.Linear(decoder.output_size, output_size + 1),
            nn.LogSoftmax(dim=-1)
        )

        self.set_optimizer()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_block(x)
        return x

    def training_step(self, batch, loss_func, device):
        self.train()
        self.optimizer.zero_grad()

        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        preds = self.forward(inputs)
        preds = preds.permute(1, 0, 2)

        input_lengths = torch.full((preds.shape[1],), preds.shape[0], dtype=torch.int32, device=device)
        target_lengths = _calculate_target_lengths(targets)

        loss = loss_func(preds, targets, input_lengths, target_lengths)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        if loss is None or torch.isnan(loss):
            print(f"Loss: {loss.item()}\ntargets: {targets}\npreds: {torch.argmax(preds, dim=-1).squeeze(0)}")

        return loss

    def validation_step(self, batch, loss_func, device):
        self.eval()

        with torch.no_grad():
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            preds = self.forward(inputs)
            preds = preds.permute(1, 0, 2)

            input_lengths = torch.full((preds.shape[1],), preds.shape[0], dtype=torch.int32, device=device)
            target_lengths = _calculate_target_lengths(targets)

            loss = loss_func(preds, targets, input_lengths, target_lengths)

            if loss is None or torch.isnan(loss):
                print(f"targets.shape: {targets.shape}, preds.shape: {preds.shape}")
                print(f"Loss: {loss.item()}\ntargets: {targets}\npreds: {torch.argmax(preds, dim=-1).squeeze(0)}")

        return loss

    def set_optimizer(self):
        optim_hparams = self.hparams_dict["optimizer"]
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=optim_hparams["learning_rate"],
            weight_decay=optim_hparams["weight_decay"]
        )


def _calculate_target_lengths(targets):
    return torch.sum(targets != 0, dim=1)


# Decoder imports (populated after decoder files are created)
from networks.decoders.lstm_decoder import LSTMDecoder
from networks.decoders.rnn_decoder import RNNDecoder
from networks.decoders.gru_decoder import GRUDecoder
from networks.decoders.transformer_decoder import TransformerDecoder

DECODERS = {
    "lstm": LSTMDecoder,
    "rnn": RNNDecoder,
    "gru": GRUDecoder,
    "transformer": TransformerDecoder,
}

# Decoder configs tuned for ~6.3M decoder parameters each.
COMPARABLE_CONFIGS = {
    "lstm": dict(input_size=2048, hidden_size=256),
    "rnn": dict(input_size=2048, hidden_size=664),
    "gru": dict(input_size=2048, hidden_size=312),
    "transformer": dict(input_size=2048, d_model=512, nhead=8,
                        num_layers=3, dim_feedforward=704, dropout=0.1),
}

# ~12.6M decoder parameters (2x baseline).
COMPARABLE_CONFIGS_2X = {
    "lstm": dict(input_size=2048, hidden_size=422),
    "rnn": dict(input_size=2048, hidden_size=1024),
    "gru": dict(input_size=2048, hidden_size=512),
    "transformer": dict(input_size=2048, d_model=640, nhead=8,
                        num_layers=4, dim_feedforward=900, dropout=0.1),
}

# ~25.2M decoder parameters (4x baseline).
COMPARABLE_CONFIGS_4X = {
    "lstm": dict(input_size=2048, hidden_size=666),
    "rnn": dict(input_size=2048, hidden_size=1536),
    "gru": dict(input_size=2048, hidden_size=798),
    "transformer": dict(input_size=2048, d_model=768, nhead=8,
                        num_layers=6, dim_feedforward=1012, dropout=0.1),
}

SCALE_CONFIGS = {
    "1x": COMPARABLE_CONFIGS,
    "2x": COMPARABLE_CONFIGS_2X,
    "4x": COMPARABLE_CONFIGS_4X,
}


def create_ablation_model(hparams, output_size, decoder_type="lstm",
                          comparable_params=True, scale="1x", in_channels=1,
                          **decoder_kwargs):
    """
    Factory function to create an AblationModel with the specified decoder.

    Args:
        hparams: Hyperparameters dict (must contain 'optimizer' key).
        output_size: Number of output classes (vocabulary size).
        decoder_type: One of 'lstm', 'rnn', 'gru', 'transformer'.
        comparable_params: If True, use config from SCALE_CONFIGS so all
            decoders have comparable parameter counts.
        scale: Parameter scale - '1x' (~6.3M), '2x' (~12.6M), '4x' (~25.2M).
        in_channels: Number of input image channels (1=grayscale, 3=RGB).
        **decoder_kwargs: Additional keyword arguments passed to the decoder.

    Returns:
        AblationModel instance.
    """
    if decoder_type not in DECODERS:
        raise ValueError(f"Unknown decoder type '{decoder_type}'. Choose from {list(DECODERS.keys())}")
    if scale not in SCALE_CONFIGS:
        raise ValueError(f"Unknown scale '{scale}'. Choose from {list(SCALE_CONFIGS.keys())}")
    if comparable_params:
        kwargs = {**SCALE_CONFIGS[scale][decoder_type], **decoder_kwargs}
    else:
        kwargs = decoder_kwargs
    decoder = DECODERS[decoder_type](**kwargs)
    return AblationModel(hparams, output_size, decoder, in_channels=in_channels)
