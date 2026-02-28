"""
Unified dataset for optical music recognition.
Loads image-ABC pairs from HuggingFace (e.g. PDMX-Synth) and
tokenises labels with a BPE tokenizer (e.g. LEGATO).
"""

import logging
import re
from typing import List, Optional, Tuple

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


# ====================================================================== #
# ABC filtering
# ====================================================================== #

def filter_abc_pitch_only(text: str) -> str:
    """Strip an ABC transcription down to notes, durations, rests, key
    signatures, accidentals, voice markers, barlines, and chord brackets.

    Removes: metadata headers (X:, T:, C:, L:, M:, Q:, I:, %%score),
    lyrics (w: lines), ties (-), slurs (()), linebreak markers ($),
    and inline comments (%...).
    """
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        # Keep key-signature lines verbatim
        if line.startswith("K:"):
            kept.append(line)
            continue
        # Keep voice markers (strip clef / transposition info)
        if line.startswith("V:"):
            kept.append(line.split()[0])
            continue
        # Drop other header / metadata / lyric lines
        if re.match(r"^[A-Za-z]:", line) or line.startswith("%%") or line.startswith("w:"):
            continue
        # Music lines — strip non-pitch/rhythm elements
        filt = re.sub(r"[()]", "", line)       # slurs
        filt = re.sub(r"-", "", filt)           # ties
        filt = re.sub(r"\$", "", filt)          # linebreak markers
        filt = re.sub(r"%.*", "", filt)         # inline comments
        filt = re.sub(r" +", " ", filt).rstrip()
        if filt.strip():
            kept.append(filt)
    return "\n".join(kept)


class UnifiedDataset(data.Dataset):
    """
    Wraps a HuggingFace Dataset of music-score images and ABC transcriptions.
    Labels are encoded with a BPE tokenizer.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        img_height: int = 128,
        max_seq_len: int = None,
        transform: Optional[callable] = None,
        strip_non_pitch: bool = False,
    ):
        """
        Args:
            hf_dataset: A HuggingFace Dataset with 'image' and 'transcription' columns.
            tokenizer: A HuggingFace tokenizer (BPE) for encoding ABC notation.
            img_height: Target image height (width scaled proportionally).
            max_seq_len: Truncate label sequences to this length (None = no truncation).
            transform: Optional additional image transforms.
            strip_non_pitch: If True, filter transcriptions to keep only notes,
                durations, rests, key signatures, accidentals, and structural tokens.
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.img_height = img_height
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.strip_non_pitch = strip_non_pitch

        # Vocabulary mappings (compatible with create_model / train_stage)
        self.vocabulary_to_index = tokenizer.get_vocab()
        self.index_to_vocabulary = {v: k for k, v in self.vocabulary_to_index.items()}

        # Special token IDs
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        logger.info(
            f"Dataset: {len(self.dataset)} samples, "
            f"vocab size: {len(self.vocabulary_to_index)}, "
            f"pad={self.pad_token_id}, bos={self.bos_token_id}, eos={self.eos_token_id}"
        )

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[index]
        image = self._process_image(sample["image"])
        labels = self._tokenize(sample["transcription"])
        return image, labels

    # ------------------------------------------------------------------ #
    # Image processing
    # ------------------------------------------------------------------ #

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """Resize to fixed height (keep aspect ratio), normalise to [-1, 1]."""
        image = image.convert("RGB")
        w, h = image.size
        new_w = max(1, int((w / h) * self.img_height))

        image = transforms.Resize((self.img_height, new_w))(image)
        image = transforms.ToTensor()(image)  # [3, H, W], values in [0, 1]
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)

        if self.transform:
            image = self.transform(image)
        return image

    # ------------------------------------------------------------------ #
    # Label tokenisation
    # ------------------------------------------------------------------ #

    def _tokenize(self, text: str) -> torch.Tensor:
        """Encode ABC-notation text with the BPE tokenizer, truncating if needed."""
        if self.strip_non_pitch:
            text = filter_abc_pitch_only(text)
        token_ids = self.tokenizer.encode(text)
        if self.max_seq_len is not None and len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len - 1] + [self.eos_token_id]
        return torch.tensor(token_ids, dtype=torch.long)


# ====================================================================== #
# Collate
# ====================================================================== #

def create_collate_fn(pad_token_id: int = 0):
    """Return a collate function that pads images and labels to uniform size."""

    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels = zip(*batch)

        max_w = max(img.shape[2] for img in images)
        max_h = max(img.shape[1] for img in images)
        max_label_len = max(len(lbl) for lbl in labels)

        padded_images = []
        for img in images:
            pl = (max_w - img.shape[2]) // 2
            pr = max_w - img.shape[2] - pl
            pt = (max_h - img.shape[1]) // 2
            pb = max_h - img.shape[1] - pt
            padded_images.append(
                torch.nn.functional.pad(img, (pl, pr, pt, pb), "constant", 0)
            )

        padded_labels = []
        for lbl in labels:
            pad_len = max_label_len - len(lbl)
            padded_labels.append(
                torch.cat([lbl, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            )

        return torch.stack(padded_images), torch.stack(padded_labels)

    return collate_fn


# ====================================================================== #
# Convenience loader
# ====================================================================== #

def load_pdmx_synth(
    dataset_id: str,
    tokenizer_id: str,
    img_height: int = 128,
    max_seq_len: int = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    transform: Optional[callable] = None,
    strip_non_pitch: bool = False,
) -> Tuple["UnifiedDataset", "UnifiedDataset", "UnifiedDataset"]:
    """
    Load a HuggingFace image-ABC dataset and split into train / val / test.

    Returns three UnifiedDataset instances.
    """
    from datasets import load_dataset as hf_load_dataset
    from transformers import AutoTokenizer

    logger.info(f"Loading dataset '{dataset_id}' from HuggingFace ...")
    full_dataset = hf_load_dataset(dataset_id, split="train")
    logger.info(f"Loaded {len(full_dataset)} samples")

    logger.info(f"Loading tokenizer '{tokenizer_id}' ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Split: train | val + test
    holdout = val_fraction + test_fraction
    split1 = full_dataset.train_test_split(test_size=holdout, seed=seed)
    train_hf = split1["train"]

    # Split holdout into val | test
    relative_test = test_fraction / holdout
    split2 = split1["test"].train_test_split(test_size=relative_test, seed=seed)
    val_hf = split2["train"]
    test_hf = split2["test"]

    logger.info(f"Splits -- train: {len(train_hf)}, val: {len(val_hf)}, test: {len(test_hf)}")

    train_ds = UnifiedDataset(train_hf, tokenizer, img_height=img_height, max_seq_len=max_seq_len, transform=transform, strip_non_pitch=strip_non_pitch)
    val_ds = UnifiedDataset(val_hf, tokenizer, img_height=img_height, max_seq_len=max_seq_len, transform=transform, strip_non_pitch=strip_non_pitch)
    test_ds = UnifiedDataset(test_hf, tokenizer, img_height=img_height, max_seq_len=max_seq_len, transform=transform, strip_non_pitch=strip_non_pitch)

    return train_ds, val_ds, test_ds
