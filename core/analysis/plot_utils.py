"""
plot_utils.py
=============
Shared plotting utilities for all fine-grained analysis scripts.

Provides:
- A consistent, publication-ready matplotlib style (dark background, custom palette).
- Helper functions to save figures with tight layout and proper DPI.
- A set of reusable plot builders (grouped bar, heatmap, scatter).

Usage:
    from core.analysis.plot_utils import apply_style, save_fig, PALETTE
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from core.utility.config_loader import cfg

# ---------------------------------------------------------------------------
# Design System
# ---------------------------------------------------------------------------

# Publication-ready color palette (colorblind-friendly)
PALETTE = {
    "msp":        "#4E79A7",   # Blue
    "maxlogit":   "#F28E2B",   # Orange
    "maxentropy": "#59A14F",   # Green
    "rba":        "#E15759",   # Red
    "boundary":   "#B07AA1",   # Purple
    "flat":       "#76B7B2",   # Teal
    "accent":     "#FF6B6B",
    "bg":         "#1A1A2E",   # Deep navy background
    "surface":    "#16213E",
    "text":       "#E0E0E0",
    "grid":       "#2D2D4E",
}

# Consistent method label mapping
METHOD_LABELS: Dict[str, str] = {
    "msp":        "MSP",
    "maxlogit":   "MaxLogit",
    "maxentropy": "MaxEntropy",
    "rba":        "RBA",
}

# Default figure size
DEFAULT_FIGSIZE = tuple(cfg.analysis.plotting.figsize)
DEFAULT_DPI     = cfg.analysis.plotting.dpi


# ---------------------------------------------------------------------------
# Style Application
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """
    Apply a consistent dark, publication-ready style to all subsequent plots.
    Should be called once at the top of every analysis script.
    """
    plt.rcParams.update({
        # Font
        "font.family":          "DejaVu Sans",
        "font.size":            cfg.analysis.plotting.font_size,
        "axes.titlesize":       cfg.analysis.plotting.font_size + 2,
        "axes.labelsize":       cfg.analysis.plotting.font_size,
        "xtick.labelsize":      cfg.analysis.plotting.font_size - 2,
        "ytick.labelsize":      cfg.analysis.plotting.font_size - 2,
        "legend.fontsize":      cfg.analysis.plotting.font_size - 2,
        "figure.titlesize":     cfg.analysis.plotting.font_size + 4,

        # Colors
        "figure.facecolor":     PALETTE["bg"],
        "axes.facecolor":       PALETTE["surface"],
        "text.color":           PALETTE["text"],
        "axes.labelcolor":      PALETTE["text"],
        "xtick.color":          PALETTE["text"],
        "ytick.color":          PALETTE["text"],
        "axes.edgecolor":       PALETTE["grid"],
        "grid.color":           PALETTE["grid"],
        "legend.facecolor":     PALETTE["surface"],
        "legend.edgecolor":     PALETTE["grid"],

        # Lines / Grid
        "axes.grid":            True,
        "grid.linestyle":       "--",
        "grid.alpha":           0.4,
        "lines.linewidth":      2.0,

        # Layout
        "figure.dpi":           DEFAULT_DPI,
        "savefig.dpi":          DEFAULT_DPI,
        "savefig.facecolor":    PALETTE["bg"],
        "savefig.bbox":         "tight",
    })


# ---------------------------------------------------------------------------
# Save Helper
# ---------------------------------------------------------------------------

def save_fig(
    fig: plt.Figure,
    out_dir: str,
    filename: str,
    formats: Tuple[str, ...] = ("png", "pdf"),
) -> List[str]:
    """
    Save a figure to disk in one or more formats.

    Args:
        fig:      The matplotlib Figure object.
        out_dir:  Output directory (created if missing).
        filename: Base filename without extension.
        formats:  Tuple of file formats to save (default: png + pdf).

    Returns:
        List of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for fmt in formats:
        path = os.path.join(out_dir, f"{filename}.{fmt}")
        fig.savefig(path, format=fmt)
        saved.append(path)
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Grouped Bar Chart
# ---------------------------------------------------------------------------

def plot_grouped_bar(
    data: Dict[str, Dict[str, float]],
    metric_name: str,
    title: str,
    out_dir: str,
    filename: str,
    higher_is_better: bool = True,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> List[str]:
    """
    Plot a grouped bar chart comparing multiple methods across multiple categories.

    Args:
        data:             {category_label: {method_key: value}}
        metric_name:      Name of the metric (for axis labels).
        title:            Plot title.
        out_dir:          Output directory.
        filename:         Base filename for saved figure.
        higher_is_better: If True, annotate best bar with a star.
        ylabel:           Y-axis label override; defaults to metric_name.
        figsize:          Figure size tuple.

    Returns:
        Paths of saved files.
    """
    apply_style()

    categories = list(data.keys())
    methods     = list(next(iter(data.values())).keys())
    n_cats      = len(categories)
    n_methods   = len(methods)

    x      = np.arange(n_cats)
    width  = 0.75 / n_methods   # total bar group width = 0.75

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        values  = [data[cat].get(method, 0.0) for cat in categories]
        offsets = x - (n_methods - 1) * width / 2 + i * width
        color   = PALETTE.get(method, f"C{i}")
        bars    = ax.bar(
            offsets, values, width,
            label=METHOD_LABELS.get(method, method.upper()),
            color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8, color=PALETTE["text"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel or metric_name)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.tight_layout()
    return save_fig(fig, out_dir, filename)


# ---------------------------------------------------------------------------
# Metric Heatmap
# ---------------------------------------------------------------------------

def plot_metric_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_dir: str,
    filename: str,
    fmt: str = ".2f",
    cmap: str = "YlOrRd",
    figsize: Tuple[int, int] = (10, 5),
) -> List[str]:
    """
    Plot a labelled heatmap (e.g., metric per method × depth band).

    Args:
        matrix:      2-D numpy array [n_rows, n_cols].
        row_labels:  Labels for rows (methods).
        col_labels:  Labels for columns (bands / categories).
        title:       Plot title.
        out_dir:     Output directory.
        filename:    Base filename.
        fmt:         Cell annotation format string.
        cmap:        Matplotlib colormap name.
        figsize:     Figure size.

    Returns:
        Paths of saved files.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=True, fmt=fmt, cmap=cmap,
        xticklabels=col_labels, yticklabels=row_labels,
        ax=ax, linewidths=0.5, linecolor=PALETTE["grid"],
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    fig.tight_layout()
    return save_fig(fig, out_dir, filename)


# ---------------------------------------------------------------------------
# AuPRC vs FPS Scatter
# ---------------------------------------------------------------------------

def plot_auprc_vs_fps(
    results: List[Dict],
    out_dir: str,
    filename: str = "auprc_vs_fps",
    figsize: Tuple[int, int] = (9, 6),
) -> List[str]:
    """
    Scatter plot of AuPRC vs inference speed (FPS) at different resolutions.

    Args:
        results:  List of dicts with keys: 'resolution', 'auprc', 'fps'.
        out_dir:  Output directory.
        filename: Base filename.

    Returns:
        Paths of saved files.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    for r in results:
        ax.scatter(r["fps"], r["auprc"], s=120, zorder=5, color=PALETTE["accent"])
        ax.annotate(
            f"{r['resolution']}px",
            (r["fps"], r["auprc"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=9, color=PALETTE["text"],
        )

    ax.set_xlabel("Inference Speed (FPS)")
    ax.set_ylabel("AuPRC (%)")
    ax.set_title("AuPRC vs Inference Speed — Resolution Trade-off")
    fig.tight_layout()
    return save_fig(fig, out_dir, filename)


# ---------------------------------------------------------------------------
# Attention Overlay
# ---------------------------------------------------------------------------

def plot_attention_overlay(
    image_rgb: np.ndarray,
    attn_map: np.ndarray,
    gt_mask: np.ndarray,
    anomaly_centroid: Tuple[int, int],
    out_dir: str,
    filename: str = "attention_overlay",
    figsize: Tuple[int, int] = (18, 5),
) -> List[str]:
    """
    Side-by-side: original image | GT anomaly mask | attention heatmap overlay.

    Args:
        image_rgb:        HxWx3 uint8 RGB image.
        attn_map:         HxW float attention map (normalised to [0,1]).
        gt_mask:          HxW binary mask (1=anomaly).
        anomaly_centroid: (row, col) pixel of the anomaly centre.
        out_dir:          Output directory.
        filename:         Base filename.

    Returns:
        Paths of saved files.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1 — Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input Image")
    cy, cx = anomaly_centroid
    axes[0].scatter([cx], [cy], s=60, c="r", marker="x", linewidths=1.5)

    # Panel 2 — GT anomaly mask
    axes[1].imshow(image_rgb, alpha=0.4)
    axes[1].imshow(gt_mask, cmap="Reds", alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title("GT Anomaly Mask")

    # Panel 3 — Attention heatmap overlay
    axes[2].imshow(image_rgb, alpha=0.5)
    im = axes[2].imshow(attn_map, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title("Self-Attention (last ViT block)")
    axes[2].scatter([cx], [cy], s=60, c="cyan", marker="x", linewidths=1.5)
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.axis("off")

    fig.suptitle("Anomaly Token Attention Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, out_dir, filename)
