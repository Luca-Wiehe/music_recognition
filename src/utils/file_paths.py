"""Centralized path registry for the music recognition project."""

from pathlib import Path

# Project root (two levels up from this file: src/utils/file_paths.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# External data root on shared storage
DATA_ROOT = Path("/storage/user/wiel/data")

# Dataset directories
DATASETS_DIR = DATA_ROOT / "datasets"
PDMX_SYNTH_DIR = DATA_ROOT / "pdmx-synth"

# Model checkpoints
CHECKPOINTS_DIR = DATA_ROOT / "checkpoints"

# Vocabulary
VOCAB_DIR = DATA_ROOT / "vocab"

# Config files (inside repo)
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Training output logs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
