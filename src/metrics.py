"""
Evaluation metrics for optical music recognition.

Provides Symbol Error Rate (SER), Character Error Rate (CER), and
sequence-level accuracy based on Levenshtein edit distance.

Definitions (consistent with blog/draft.md Appendix A):
  SER  = edit_distance(pred_tokens, gt_tokens) / len(gt_tokens)
  CER  = edit_distance(pred_chars,  gt_chars)  / len(gt_chars)
  SeqAcc = fraction of samples where pred == gt exactly

All public functions operate on **token-ID sequences** (lists or 1-D
tensors) with special tokens (BOS / EOS / PAD) already stripped.  The
helper ``strip_special_tokens`` is provided for that purpose.
"""

from __future__ import annotations

from typing import List, Sequence

import torch


# ------------------------------------------------------------------
# Levenshtein edit distance
# ------------------------------------------------------------------

def edit_distance(seq_a: Sequence, seq_b: Sequence) -> int:
    """
    Levenshtein edit distance between two sequences.

    Works on any sequences whose elements support equality comparison
    (token IDs, characters, strings, …).
    """
    la, lb = len(seq_a), len(seq_b)
    # Optimise for common trivial cases
    if la == 0:
        return lb
    if lb == 0:
        return la

    # Single-row DP (O(min(la, lb)) space)
    if la < lb:
        seq_a, seq_b = seq_b, seq_a
        la, lb = lb, la

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev = curr

    return prev[lb]


# ------------------------------------------------------------------
# Sequence cleaning
# ------------------------------------------------------------------

def strip_special_tokens(
    token_ids: torch.Tensor | list[int],
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    """
    Remove PAD, BOS, and EOS tokens from a 1-D sequence of token IDs.

    Truncates at the first EOS (if present) and then strips any remaining
    PAD / BOS tokens.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    # Truncate at first EOS
    try:
        eos_pos = token_ids.index(eos_id)
        token_ids = token_ids[:eos_pos]
    except ValueError:
        pass

    # Remove BOS and PAD
    return [t for t in token_ids if t not in (pad_id, bos_id)]


# ------------------------------------------------------------------
# Batch-level metric computation
# ------------------------------------------------------------------

def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    index_to_vocab: dict[int, str] | None = None,
) -> dict[str, float]:
    """
    Compute SER, CER, and sequence accuracy for a batch.

    Args:
        predictions: (B, pred_len) token IDs from ``model.generate()``.
        targets:     (B, tgt_len) ground-truth token IDs (may contain
                     BOS, EOS, PAD).
        pad_id / bos_id / eos_id: special token IDs.
        index_to_vocab: Optional mapping ``int → str`` for CER
                        computation.  When *None*, CER is skipped.

    Returns:
        Dictionary with keys:
          ``ser``          – mean Symbol Error Rate (0–1, lower is better)
          ``cer``          – mean Character Error Rate (0–1), or -1 if no vocab
          ``sequence_acc`` – fraction of exact matches (0–1, higher is better)
          ``num_samples``  – batch size used
    """
    batch_size = predictions.shape[0]

    total_ser = 0.0
    total_cer = 0.0
    exact_matches = 0
    cer_available = index_to_vocab is not None

    for i in range(batch_size):
        pred_tokens = strip_special_tokens(predictions[i], pad_id, bos_id, eos_id)
        gt_tokens = strip_special_tokens(targets[i], pad_id, bos_id, eos_id)

        # --- SER (token-level edit distance) ---
        if len(gt_tokens) > 0:
            total_ser += edit_distance(pred_tokens, gt_tokens) / len(gt_tokens)
        else:
            # Empty ground truth: any non-empty prediction counts as 100% error
            total_ser += 0.0 if len(pred_tokens) == 0 else 1.0

        # --- Sequence accuracy ---
        if pred_tokens == gt_tokens:
            exact_matches += 1

        # --- CER (character-level edit distance) ---
        if cer_available:
            pred_str = _tokens_to_str(pred_tokens, index_to_vocab)
            gt_str = _tokens_to_str(gt_tokens, index_to_vocab)
            if len(gt_str) > 0:
                total_cer += edit_distance(list(pred_str), list(gt_str)) / len(gt_str)
            else:
                total_cer += 0.0 if len(pred_str) == 0 else 1.0

    return {
        "ser": total_ser / batch_size,
        "cer": total_cer / batch_size if cer_available else -1.0,
        "sequence_acc": exact_matches / batch_size,
        "num_samples": batch_size,
    }


def _tokens_to_str(token_ids: list[int], index_to_vocab: dict[int, str]) -> str:
    """Join token strings (space-separated) for character-level comparison."""
    return " ".join(index_to_vocab.get(t, f"<{t}>") for t in token_ids)


# ------------------------------------------------------------------
# Aggregate helpers
# ------------------------------------------------------------------

def aggregate_metrics(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    """
    Weighted-average a list of per-batch metric dicts (from
    ``compute_metrics``).
    """
    total_samples = sum(m["num_samples"] for m in metric_dicts)
    if total_samples == 0:
        return {"ser": 0.0, "cer": 0.0, "sequence_acc": 0.0, "num_samples": 0}

    agg = {"ser": 0.0, "cer": 0.0, "sequence_acc": 0.0}
    for m in metric_dicts:
        w = m["num_samples"] / total_samples
        agg["ser"] += m["ser"] * w
        if m["cer"] >= 0:
            agg["cer"] += m["cer"] * w
        agg["sequence_acc"] += m["sequence_acc"] * w

    agg["num_samples"] = total_samples
    return agg
