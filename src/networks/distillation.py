"""
Knowledge distillation for MusicTrOCR.

Distills a large teacher model (e.g. DeiT-Small backbone, 59M params) into a
smaller student (e.g. MobileViT-Small, 42M params) by transferring:

1. **Encoder feature representations** — the visual features most relevant for
   music-note recognition.  An MSE loss aligns student encoder outputs to the
   teacher's, after adaptive pooling to handle different spatial resolutions.

2. **Output logit distributions** — temperature-scaled KL divergence transfers
   inter-symbol relationships (e.g. quarter-note C4 vs D4) from teacher to
   student (Hinton et al., 2015).

Total loss:
    L = alpha * CE(student, ground_truth)
      + beta  * MSE(student_features, teacher_features)
      + gamma * T^2 * KL(softmax(teacher/T) || softmax(student/T))

Usage:
    teacher = MusicTrOCR(...)          # load from checkpoint, frozen
    student = MusicTrOCR(...)          # smaller backbone, trainable
    model   = DistillationWrapper(teacher, student, ...)
    # model exposes training_step / validation_step compatible with train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DistillationWrapper(nn.Module):
    """
    Wraps a frozen teacher and a trainable student MusicTrOCR model.

    Exposes ``training_step`` and ``validation_step`` with the same signature
    used by ``src.train.train_epoch`` / ``validate_epoch``, so the existing
    training loop works without modification.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.2,
        temperature: float = 4.0,
    ):
        """
        Args:
            teacher:     Pre-trained MusicTrOCR (will be frozen).
            student:     Smaller MusicTrOCR to train.
            alpha:       Weight for hard-label cross-entropy loss.
            beta:        Weight for encoder feature distillation (MSE).
            gamma:       Weight for logit distillation (KL divergence).
            temperature: Softmax temperature for logit KD.
        """
        super().__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

        # Freeze teacher entirely
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Proxy attributes so the training loop can access vocab info
        self.PAD_TOKEN_ID = student.PAD_TOKEN_ID
        self.START_TOKEN_ID = student.START_TOKEN_ID
        self.END_TOKEN_ID = student.END_TOKEN_ID
        self.vocab_size = student.vocab_size
        self.d_model = student.d_model
        self.max_seq_len = student.max_seq_len

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Return only student parameters (for the optimizer)."""
        return self.student.parameters()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.student.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        """Student follows mode; teacher is always eval."""
        super().train(mode)
        self.student.train(mode)
        self.teacher.eval()
        return self

    # ------------------------------------------------------------------
    # Checkpoint helpers — save / load only the student
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        return self.student.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.student.load_state_dict(state_dict, *args, **kwargs)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_distillation_loss(
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE between student and teacher encoder feature sequences.

        Both tensors have shape ``(B, seq_len, d_model)`` but ``seq_len`` may
        differ because different backbones produce different spatial resolutions.
        We adaptively pool both to the shorter length before computing MSE.
        """
        sf = student_features  # (B, S_s, D)
        tf = teacher_features  # (B, S_t, D)

        target_len = min(sf.shape[1], tf.shape[1])

        # Adaptive average-pool along the sequence dimension
        if sf.shape[1] != target_len:
            sf = F.adaptive_avg_pool1d(
                sf.transpose(1, 2), target_len
            ).transpose(1, 2)
        if tf.shape[1] != target_len:
            tf = F.adaptive_avg_pool1d(
                tf.transpose(1, 2), target_len
            ).transpose(1, 2)

        # Normalise each feature vector so MSE is scale-invariant
        sf = F.layer_norm(sf, [sf.shape[-1]])
        tf = F.layer_norm(tf, [tf.shape[-1]])

        return F.mse_loss(sf, tf)

    def _logit_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        KL divergence between temperature-scaled teacher and student logits.

        Args:
            student_logits: (B, T, V)
            teacher_logits: (B, T, V)
            padding_mask:   (B, T) — True where padded (to be ignored).
        """
        T = self.temperature

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1).detach()

        # Per-position KL, summed over vocabulary
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)  # (B, T)

        # Mask out padding positions
        if padding_mask is not None:
            valid = ~padding_mask
            kl = (kl * valid.float()).sum() / valid.sum().clamp(min=1)
        else:
            kl = kl.mean()

        # Scale by T^2 (Hinton et al.)
        return kl * (T * T)

    def _compute_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined distillation loss.

        Returns dict with 'loss' (total) and individual components for logging.
        """
        s_logits = student_outputs["logits"]
        t_logits = teacher_outputs["logits"]
        decoder_target = student_outputs["decoder_target"]
        padding_mask = student_outputs["tgt_key_padding_mask"]

        # 1. Hard-label cross-entropy
        ce_loss = F.cross_entropy(
            s_logits.reshape(-1, self.vocab_size),
            decoder_target.reshape(-1),
            ignore_index=self.PAD_TOKEN_ID,
        )

        # 2. Encoder feature distillation
        feature_loss = self._feature_distillation_loss(
            student_outputs["encoder_features"],
            teacher_outputs["encoder_features"],
        )

        # 3. Logit KD
        logit_kd_loss = self._logit_distillation_loss(s_logits, t_logits, padding_mask)

        total = self.alpha * ce_loss + self.beta * feature_loss + self.gamma * logit_kd_loss

        return {
            "loss": total,
            "ce_loss": ce_loss.detach(),
            "feature_loss": feature_loss.detach(),
            "logit_kd_loss": logit_kd_loss.detach(),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run both teacher and student, return combined outputs."""
        with torch.no_grad():
            teacher_out = self.teacher(images, targets)

        student_out = self.student(images, targets)
        return student_out, teacher_out

    # ------------------------------------------------------------------
    # Training / validation steps (same interface as MusicTrOCR)
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch,
        device,
        config=None,
        epoch=None,
        batch_idx=None,
        scaler=None,
        accumulation_steps=1,
    ):
        self.train()

        images, targets = batch
        images, targets = images.to(device), targets.to(device)

        use_amp = scaler is not None

        with torch.amp.autocast("cuda", enabled=use_amp):
            # Teacher forward (no grad, always fp32-safe via autocast)
            with torch.no_grad():
                teacher_out = self.teacher(images, targets)

            # Student forward
            student_out = self.student(images, targets)

            losses = self._compute_loss(student_out, teacher_out)
            scaled_loss = losses["loss"] / accumulation_steps

        # Backward (accumulates gradients)
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return losses["loss"]

    def validation_step(
        self,
        batch,
        device,
        config=None,
        epoch=None,
        batch_idx=None,
        use_amp=False,
    ):
        self.eval()

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            teacher_out = self.teacher(images, targets)
            student_out = self.student(images, targets)

            losses = self._compute_loss(student_out, teacher_out)

        return losses["loss"]

    # ------------------------------------------------------------------
    # Generation (delegates to student)
    # ------------------------------------------------------------------

    def generate(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.student.generate(images, **kwargs)
