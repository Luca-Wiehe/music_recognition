#!/usr/bin/env python3
"""
Generate VRAM profiling visualisations from JSON results.

Reads one or more profiling JSON files (produced by vram_profiler.py) and
generates:
  1. Stacked bar chart  — VRAM breakdown per backbone
  2. Pareto frontier     — accuracy vs peak VRAM
  3. Pareto frontier     — accuracy vs training throughput
  4. Hardware tier table  — recommendations for 16 / 24 / 48 GB budgets

Usage:
    python -m src.profiling.plot_results --results-dir profiling_results
    python -m src.profiling.plot_results --results-dir profiling_results --output-dir figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> list[dict]:
    """Load all JSON profiling results from a directory."""
    results = []
    for p in sorted(Path(results_dir).glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    if not results:
        raise FileNotFoundError(f"No JSON files found in {results_dir}")
    return results


# ---------------------------------------------------------------------------
# 1.  VRAM breakdown stacked bar chart
# ---------------------------------------------------------------------------

def plot_vram_breakdown(results: list[dict], output_dir: Path):
    """Stacked bar chart: parameters / gradients / optimizer / activations."""
    names = [_short_name(r) for r in results]
    params = [r["vram"]["parameters_mb"] for r in results]
    grads = [r["vram"]["gradients_mb"] for r in results]
    optim = [r["vram"]["optimizer_states_mb"] for r in results]
    acts = [r["vram"]["activations_estimated_mb"] for r in results]

    x = np.arange(len(names))
    width = 0.55

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))

    b1 = ax.bar(x, params, width, label="Parameters", color="#4C72B0")
    b2 = ax.bar(x, grads, width, bottom=params, label="Gradients", color="#55A868")
    bot2 = [p + g for p, g in zip(params, grads)]
    b3 = ax.bar(x, optim, width, bottom=bot2, label="Optimizer states", color="#C44E52")
    bot3 = [b + o for b, o in zip(bot2, optim)]
    b4 = ax.bar(x, acts, width, bottom=bot3, label="Activations (est.)", color="#8172B2")

    # Reference lines for GPU tiers
    for tier, color, ls in [(16_384, "#888", "--"), (24_576, "#aaa", ":"), (32_768, "#ccc", ":")]:
        ax.axhline(tier, color=color, linestyle=ls, linewidth=0.8)
        ax.text(len(names) - 0.4, tier + 200, f"{tier // 1024} GB", fontsize=8, color=color)

    ax.set_ylabel("VRAM (MB)")
    ax.set_title("Training VRAM Breakdown by Backbone")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "vram_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  Saved vram_breakdown.png")


# ---------------------------------------------------------------------------
# 2.  Pareto frontier: accuracy vs VRAM
# ---------------------------------------------------------------------------

def plot_pareto_vram(results: list[dict], output_dir: Path):
    """Scatter plot with Pareto frontier: val loss vs peak VRAM."""
    results_with_loss = [r for r in results if "best_val_loss" in r]
    if not results_with_loss:
        print("  Skipping pareto_vram.png (no val loss data)")
        return

    names = [_short_name(r) for r in results_with_loss]
    vram = [r["vram"]["peak_training_mb"] / 1024 for r in results_with_loss]
    loss = [r["best_val_loss"] for r in results_with_loss]
    colors = [_technique_color(r) for r in results_with_loss]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (v, l, n, c) in enumerate(zip(vram, loss, names, colors)):
        ax.scatter(v, l, c=c, s=80, zorder=3, edgecolors="black", linewidths=0.5)
        ax.annotate(n, (v, l), textcoords="offset points", xytext=(6, 6), fontsize=8)

    # Draw Pareto frontier
    _draw_pareto_frontier(ax, vram, loss)

    # GPU tier bands
    for tier, label in [(16, "16 GB"), (24, "24 GB"), (32, "32 GB"), (48, "48 GB")]:
        ax.axvline(tier, color="#ddd", linestyle="--", linewidth=0.8)
        ax.text(tier + 0.2, ax.get_ylim()[1] * 0.97, label, fontsize=8, color="#999")

    ax.set_xlabel("Peak Training VRAM (GB)")
    ax.set_ylabel("Best Validation Loss (lower = better)")
    ax.set_title("Accuracy vs VRAM — Pareto Frontier")

    _add_technique_legend(ax, results_with_loss)
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_vram.png", dpi=150)
    plt.close(fig)
    print(f"  Saved pareto_vram.png")


# ---------------------------------------------------------------------------
# 3.  Pareto frontier: accuracy vs throughput
# ---------------------------------------------------------------------------

def plot_pareto_throughput(results: list[dict], output_dir: Path):
    """Scatter plot with Pareto frontier: val loss vs throughput."""
    results_with_loss = [r for r in results if "best_val_loss" in r]
    if not results_with_loss:
        print("  Skipping pareto_throughput.png (no val loss data)")
        return

    names = [_short_name(r) for r in results_with_loss]
    throughput = [r["throughput"]["train_it_per_sec"] for r in results_with_loss]
    loss = [r["best_val_loss"] for r in results_with_loss]
    colors = [_technique_color(r) for r in results_with_loss]

    fig, ax = plt.subplots(figsize=(8, 5))
    for v, l, n, c in zip(throughput, loss, names, colors):
        ax.scatter(v, l, c=c, s=80, zorder=3, edgecolors="black", linewidths=0.5)
        ax.annotate(n, (v, l), textcoords="offset points", xytext=(6, 6), fontsize=8)

    # Pareto frontier (here "better" = higher throughput AND lower loss)
    _draw_pareto_frontier(ax, throughput, loss, higher_x_better=True)

    ax.set_xlabel("Training Throughput (it/s)")
    ax.set_ylabel("Best Validation Loss (lower = better)")
    ax.set_title("Accuracy vs Throughput — Pareto Frontier")

    _add_technique_legend(ax, results_with_loss)
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_throughput.png", dpi=150)
    plt.close(fig)
    print(f"  Saved pareto_throughput.png")


# ---------------------------------------------------------------------------
# 4.  Hardware tier recommendations
# ---------------------------------------------------------------------------

def print_hardware_recommendations(results: list[dict]):
    """Print recommendations for 16 / 24 / 48 GB GPU budgets."""
    results_with_loss = sorted(
        [r for r in results if "best_val_loss" in r],
        key=lambda r: r["best_val_loss"],
    )

    tiers = [
        ("16 GB (RTX 4060 / 4070)", 16_384),
        ("24 GB (RTX 3090 / 4090)", 24_576),
        ("48 GB (A40 / A6000)",     49_152),
    ]

    print("\n" + "=" * 70)
    print("  Hardware Tier Recommendations")
    print("=" * 70)

    for tier_name, budget_mb in tiers:
        fits = [r for r in results_with_loss if r["vram"]["peak_training_mb"] <= budget_mb]
        print(f"\n  {tier_name}:")
        if not fits:
            print("    No profiled configuration fits this budget.")
            print("    Consider: LoRA, QLoRA, or reducing batch size / image height.")
            continue
        best = fits[0]  # already sorted by loss
        print(f"    Recommended: {_short_name(best)}")
        print(f"    Peak VRAM:   {best['vram']['peak_training_mb']:.0f} MB "
              f"({best['vram']['peak_training_mb'] / budget_mb * 100:.0f}% of budget)")
        print(f"    Val loss:    {best['best_val_loss']:.4f}")
        print(f"    Throughput:  {best['throughput']['train_it_per_sec']:.2f} it/s")
        print(f"    Params:      {best['model']['trainable_params']:,}")

    # Also print configs that don't have val loss
    no_loss = [r for r in results if "best_val_loss" not in r]
    if no_loss:
        print(f"\n  Note: {len(no_loss)} config(s) have no val loss data yet "
              "(no checkpoint found). Re-run after training completes.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_name(result: dict) -> str:
    """Human-readable short name from a profiling result."""
    backbone = result["backbone"].split("/")[-1]
    model_type = result["model_type"]
    if model_type == "Distillation":
        return f"distill→{backbone}"
    return backbone


TECHNIQUE_COLORS = {
    "MusicTrOCR": "#4C72B0",
    "Distillation": "#55A868",
    "LoRA": "#DD8452",
    "QLoRA": "#C44E52",
}


def _technique_color(result: dict) -> str:
    return TECHNIQUE_COLORS.get(result["model_type"], "#8172B2")


def _add_technique_legend(ax, results: list[dict]):
    """Add a color legend for model types present in the results."""
    types_seen = sorted(set(r["model_type"] for r in results))
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=TECHNIQUE_COLORS.get(t, "#8172B2"),
                    markersize=8, markeredgecolor="black", markeredgewidth=0.5)
        for t in types_seen
    ]
    ax.legend(handles, types_seen, loc="upper right", fontsize=9)


def _draw_pareto_frontier(ax, xs, ys, higher_x_better=False):
    """
    Draw the Pareto frontier line.

    Non-dominated points: lower y is better, lower x is better (unless
    higher_x_better is True, e.g. throughput).
    """
    points = sorted(zip(xs, ys), key=lambda p: p[0], reverse=higher_x_better)

    frontier_x, frontier_y = [], []
    best_y = float("inf")
    for x, y in points:
        if y <= best_y:
            frontier_x.append(x)
            frontier_y.append(y)
            best_y = y

    if len(frontier_x) >= 2:
        ax.plot(frontier_x, frontier_y, "k--", linewidth=1.0, alpha=0.5,
                label="Pareto frontier")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate VRAM profiling plots from JSON results"
    )
    parser.add_argument("--results-dir", type=str, default="profiling_results",
                        help="Directory containing profiling JSON files")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory to save generated plots")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(results)} profiling result(s)")

    plot_vram_breakdown(results, output_dir)
    plot_pareto_vram(results, output_dir)
    plot_pareto_throughput(results, output_dir)
    print_hardware_recommendations(results)


if __name__ == "__main__":
    main()
