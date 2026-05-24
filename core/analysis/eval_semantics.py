"""
eval_semantics.py  —  TASK 1: Spatial & Semantic Boundary Analysis
==================================================================

Analyses WHERE the anomaly detection model fails spatially by splitting pixels
into two complementary partitions and measuring OOD metrics independently:

  1. SEMANTIC BOUNDARY ANALYSIS
     - Extracts 'boundary' pixels (near class-change transitions in the GT mask)
       using morphological operations (dilation - erosion = boundary ring).
     - Extracts 'flat' pixels (everything else, i.e., homogeneous interiors).
     - Computes AuPRC and FPR@TPR95 separately for each region.
     - Validates whether MaxLogit / MaxEntropy reduce FP rate near boundaries
       compared to MSP.

  2. DEPTH / DISTANCE BAND ANALYSIS
     - Divides each image into N horizontal bands (top = far, bottom = near).
     - Computes AuPRC and FPR@TPR95 per band for every scoring method.
     - Reveals if the model struggles more with distant, small anomalies vs
       close, large ones.

INPUT
-----
Pre-saved logits are expected at the paths defined in config.yml:
  cfg.paths.logits.erfnet   (for ERFNet)
  cfg.paths.logits.eomt     (for EoMT)

The script matches logit files to ground-truth masks from any dataset directory
by following the same path convention used in evalAnomaly.py.

USAGE
-----
# Run on a specific dataset + model (all scoring methods evaluated):
python core/analysis/eval_semantics.py \
    --logits_dir   core/evaluation/eval/saved_logits/erfnet/<dataset_name> \
    --images_glob  "Datasets/Fishyscapes/fs_static/images/*.jpg" \
    --dataset_type fs_static \
    --model_name   ERFNet \
    --n_bands      5 \
    --boundary_kernel_size 15

OUTPUT
------
  results/analysis/semantics/
      boundary_bar_<dataset>.png/.pdf
      depth_heatmap_auprc_<dataset>.png/.pdf
      depth_heatmap_fpr95_<dataset>.png/.pdf
      summary_<dataset>.txt
"""

import os
import sys
import glob
import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.utility.logger import logger
from core.utility.config_loader import cfg
from core.utility.eval_common import gt_path_from_image, load_logits, load_ood_mask
from core.analysis.plot_utils import (
    apply_style, save_fig, plot_grouped_bar, plot_metric_heatmap, PALETTE,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IGNORE_INDEX = cfg.eval.ignore_index

def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute AuPRC and FPR@TPR95 for a 1-D score/label pair.

    Args:
        scores: 1-D float array of anomaly scores.
        labels: 1-D int array (1 = anomaly, 0 = in-distribution).

    Returns:
        (auprc_pct, fpr95_pct) — both as percentages [0, 100].

    Notes:
        - Pixels with labels == IGNORE_INDEX (255) are excluded.
        - Returns (NaN, NaN) if either class is absent after filtering.
    """
    valid = labels != IGNORE_INDEX
    scores = scores[valid]
    labels = labels[valid]

    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    auprc = average_precision_score(labels, scores) * 100.0
    fpr95 = fpr_at_95_tpr(scores, labels) * 100.0
    return auprc, fpr95


def compute_all_scores(logits: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Compute MSP, MaxLogit, and MaxEntropy anomaly scores from a logit tensor.

    Args:
        logits: Float tensor of shape (1, C, H, W).

    Returns:
        Dict mapping method name to 2-D numpy anomaly score map (H, W).
        Higher score = higher probability of being OOD.
    """
    probs = F.softmax(logits, dim=1)
    msp = 1.0 - probs.max(dim=1).values
    maxlogit = -logits.max(dim=1).values
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)

    return {
        "msp":        msp.squeeze(0).numpy(),
        "maxlogit":   maxlogit.squeeze(0).numpy(),
        "maxentropy": entropy.squeeze(0).numpy(),
    }


def extract_boundary_mask(
    gt_mask: np.ndarray,
    kernel_size: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract binary 'boundary' and 'flat' pixel masks from a GT anomaly mask.

    Strategy:
        boundary = dilated_gt XOR eroded_gt
        (the ring of pixels around semantic transitions)

    A pixel in the **boundary** mask is semantically ambiguous — it lies near
    a class edge. This is where MSP (based on max-softmax) is most likely to
    produce false positives because the softmax peak is naturally suppressed
    at class borders.

    Pixels outside the boundary (in the **flat** mask) are deep inside a
    homogeneous region (all-OOD or all-in-dist).

    Args:
        gt_mask:      H×W uint8 with values 0, 1, 255.
        kernel_size:  Size of the square structuring element (pixels).
                      Larger = wider boundary ring.

    Returns:
        boundary_mask: H×W bool, True where pixel is near a semantic edge.
        flat_mask:     H×W bool, True where pixel is deep inside a region.

    Notes:
        - Ignore pixels (255) are excluded from both partitions.
        - We dilate/erode a binary version where (0 or 1) = valid.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )

    valid = (gt_mask != IGNORE_INDEX).astype(np.uint8)

    ood_binary = ((gt_mask == 1) & (valid == 1)).astype(np.uint8)

    if ood_binary.max() == 0:
        flat_mask     = valid.astype(bool)
        boundary_mask = np.zeros_like(flat_mask)
        return boundary_mask, flat_mask

    dilated = cv2.dilate(ood_binary, kernel, iterations=1)
    eroded  = cv2.erode(ood_binary, kernel, iterations=1)

    raw_boundary = ((dilated - eroded) > 0)
    boundary_mask = raw_boundary & (valid.astype(bool))
    flat_mask     = (~boundary_mask) & (valid.astype(bool))

    return boundary_mask, flat_mask


def make_depth_bands(height: int, n_bands: int) -> List[Tuple[int, int]]:
    """
    Divide the image height into `n_bands` equal horizontal strips.

    Band 0 = top of image (distant, near vanishing point).
    Band n-1 = bottom of image (close to camera, large objects).

    Args:
        height:  Image height in pixels.
        n_bands: Number of bands.

    Returns:
        List of (row_start, row_end) tuples (exclusive end, like slice).
    """
    edges = np.linspace(0, height, n_bands + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_bands)]


def depth_band_label(band_idx: int, n_bands: int) -> str:
    """
    Human-readable label for a depth band index.

    Args:
        band_idx: Zero-based index (0 = top/far).
        n_bands:  Total number of bands.

    Returns:
        Label string, e.g. 'Far (B0)' or 'Near (B4)'.
    """
    if band_idx == 0:
        tag = "Far"
    elif band_idx == n_bands - 1:
        tag = "Near"
    elif band_idx <= n_bands // 2:
        tag = "Mid-far"
    else:
        tag = "Mid-near"
    return f"{tag} (B{band_idx})"


class SemanticAnalyser:
    """
    Orchestrates semantic boundary and depth-band metric computation.

    Uses pre-saved .pt logit files (no GPU required) and GT masks from disk.

    Attributes:
        logits_dir:       Directory containing per-image .pt logit files.
        images_glob:      Glob pattern for input images (to find GT paths).
        dataset_type:     Dataset identifier for GT remapping.
        model_name:       Name of the model (for plot titles / filenames).
        n_bands:          Number of horizontal depth bands.
        boundary_kernel:  Morphological kernel size for boundary extraction.
        out_dir:          Directory where results / plots are saved.
        methods:          List of scoring method keys to evaluate.
    """

    METHODS = ["msp", "maxlogit", "maxentropy"]

    def __init__(
        self,
        logits_dir:      str,
        images_glob:     str,
        dataset_type:    str,
        model_name:      str  = "Model",
        n_bands:         int  = 5,
        boundary_kernel: int  = 15,
        out_dir:         str  = None,
    ) -> None:
        if not os.path.isabs(logits_dir):
            logits_dir = os.path.normpath(os.path.join(cfg.paths.root, logits_dir))
        self.logits_dir      = logits_dir
        self.images_glob     = images_glob
        self.dataset_type    = dataset_type
        self.model_name      = model_name
        self.n_bands         = n_bands
        self.boundary_kernel = boundary_kernel
        self.out_dir         = out_dir or os.path.join(
            cfg.paths.root, cfg.analysis.reports_dir, "semantics"
        )

        self._boundary_acc: Dict[str, Dict[str, List]] = {}
        self._flat_acc:     Dict[str, Dict[str, List]] = {}

        self._band_acc: Dict[int, Dict[str, Dict[str, List]]] = {}

    def run(self) -> None:
        """
        Main entry point: load all logit/GT pairs, accumulate statistics,
        compute metrics, plot results, and save a text summary.
        """
        if not os.path.isdir(self.logits_dir):
            logger.error(
                f"Logits directory not found: {self.logits_dir}\n"
                f"  Tip: check that logits were saved by running evalAnomaly.py "
                f"with --save_logits, then pass the correct sub-folder name."
            )
            parent = os.path.dirname(self.logits_dir)
            if os.path.isdir(parent):
                siblings = sorted(os.listdir(parent))
                logger.error(
                    f"  Available sub-folders inside '{parent}':\n"
                    + "\n".join(f"    - {s}" for s in siblings)
                )
            return

        image_paths = sorted(glob.glob(os.path.expanduser(self.images_glob)))
        if not image_paths:
            logger.error(f"No images found matching: {self.images_glob}")
            return

        logger.info(
            f"Semantic analysis | model={self.model_name} "
            f"| dataset={self.dataset_type} | n_images={len(image_paths)} "
            f"| n_bands={self.n_bands} | boundary_kernel={self.boundary_kernel}px"
        )
        logger.info(f"Logits dir : {self.logits_dir}")

        self._init_accumulators()
        n_processed = 0

        for img_path in image_paths:
            basename = Path(img_path).stem
            logit_path = os.path.join(self.logits_dir, f"{basename}.pt")

            if not os.path.exists(logit_path):
                logger.warning(f"Logit not found, skipping: {logit_path}")
                continue

            gt_path = gt_path_from_image(img_path, self.dataset_type)
            if not os.path.exists(gt_path):
                logger.warning(f"GT mask not found, skipping: {gt_path}")
                continue

            try:
                self._process_image(logit_path, gt_path)
                n_processed += 1
            except Exception as exc:
                logger.error(f"Error processing {basename}: {exc}")
                continue

        if n_processed == 0:
            logger.error("No images were successfully processed. Aborting.")
            return

        logger.info(f"Processed {n_processed}/{len(image_paths)} images.")

        boundary_results, flat_results = self._compute_boundary_metrics()
        band_results = self._compute_depth_metrics()

        self._print_summary(boundary_results, flat_results, band_results)
        self._save_plots(boundary_results, flat_results, band_results)

    def _init_accumulators(self) -> None:
        """Initialise empty accumulator dicts."""
        for method in self.METHODS:
            self._boundary_acc[method] = {"scores": [], "labels": []}
            self._flat_acc[method]     = {"scores": [], "labels": []}

        for b in range(self.n_bands):
            self._band_acc[b] = {
                m: {"scores": [], "labels": []} for m in self.METHODS
            }

    def _process_image(self, logit_path: str, gt_path: str) -> None:
        """
        Load one logit file and its GT mask, extract boundary/flat/band
        partitions, and accumulate scores + labels.
        """
        logits = load_logits(logit_path)
        H, W   = logits.shape[2], logits.shape[3]

        gt_mask = load_ood_mask(gt_path, (H, W), self.dataset_type)

        if 1 not in np.unique(gt_mask):
            return

        scores_dict = compute_all_scores(logits)

        boundary_mask, flat_mask = extract_boundary_mask(
            gt_mask, self.boundary_kernel
        )
        gt_flat = gt_mask.flatten()

        for method, scores in scores_dict.items():
            s_flat = scores.flatten()

            self._boundary_acc[method]["scores"].append(s_flat[boundary_mask.flatten()])
            self._boundary_acc[method]["labels"].append(gt_flat[boundary_mask.flatten()])

            self._flat_acc[method]["scores"].append(s_flat[flat_mask.flatten()])
            self._flat_acc[method]["labels"].append(gt_flat[flat_mask.flatten()])

        bands = make_depth_bands(H, self.n_bands)
        for band_idx, (r_start, r_end) in enumerate(bands):
            band_row_mask = np.zeros(H, dtype=bool)
            band_row_mask[r_start:r_end] = True
            band_pixel_mask = np.tile(band_row_mask[:, np.newaxis], (1, W))

            valid_in_band = band_pixel_mask.flatten() & (gt_flat != IGNORE_INDEX)

            for method, scores in scores_dict.items():
                s_flat = scores.flatten()
                self._band_acc[band_idx][method]["scores"].append(s_flat[valid_in_band])
                self._band_acc[band_idx][method]["labels"].append(gt_flat[valid_in_band])

    def _compute_boundary_metrics(
        self,
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        """
        Aggregate boundary and flat accumulators and compute metrics.

        Returns:
            boundary_results: {method: (auprc, fpr95)}
            flat_results:     {method: (auprc, fpr95)}
        """
        boundary_results = {}
        flat_results     = {}

        for method in self.METHODS:
            b_scores = np.concatenate(self._boundary_acc[method]["scores"])
            b_labels = np.concatenate(self._boundary_acc[method]["labels"])
            boundary_results[method] = compute_metrics(b_scores, b_labels)

            f_scores = np.concatenate(self._flat_acc[method]["scores"])
            f_labels = np.concatenate(self._flat_acc[method]["labels"])
            flat_results[method] = compute_metrics(f_scores, f_labels)

        return boundary_results, flat_results

    def _compute_depth_metrics(
        self,
    ) -> Dict[int, Dict[str, Tuple[float, float]]]:
        """
        Aggregate per-band accumulators and compute metrics.

        Returns:
            band_results: {band_idx: {method: (auprc, fpr95)}}
        """
        band_results = {}
        for band_idx in range(self.n_bands):
            band_results[band_idx] = {}
            for method in self.METHODS:
                scores = np.concatenate(self._band_acc[band_idx][method]["scores"])
                labels = np.concatenate(self._band_acc[band_idx][method]["labels"])
                band_results[band_idx][method] = compute_metrics(scores, labels)
        return band_results

    def _print_summary(
        self,
        boundary_results: Dict,
        flat_results: Dict,
        band_results: Dict,
    ) -> None:
        """Print a formatted summary table to the logger and save to txt."""
        lines = []
        sep   = "=" * 70

        lines.append(sep)
        lines.append(
            f"  Semantic Analysis Summary  |  "
            f"Model: {self.model_name}  |  Dataset: {self.dataset_type}"
        )
        lines.append(sep)

        lines.append("")
        lines.append("  REGION ANALYSIS")
        lines.append(f"  {'Method':<15} {'Region':<12} {'AuPRC %':>10} {'FPR95 %':>10}")
        lines.append("  " + "-" * 50)

        for method in self.METHODS:
            b_auprc, b_fpr95 = boundary_results[method]
            f_auprc, f_fpr95 = flat_results[method]
            m_label = method.upper()
            lines.append(f"  {m_label:<15} {'boundary':<12} {b_auprc:>10.2f} {b_fpr95:>10.2f}")
            lines.append(f"  {'':<15} {'flat':<12} {f_auprc:>10.2f} {f_fpr95:>10.2f}")

        lines.append("")
        lines.append("  DEPTH BAND ANALYSIS (AuPRC %)")
        band_labels = [depth_band_label(b, self.n_bands) for b in range(self.n_bands)]
        header = f"  {'Method':<15}" + "".join(f" {lbl:>14}" for lbl in band_labels)
        lines.append(header)
        lines.append("  " + "-" * (15 + 15 * self.n_bands))

        for method in self.METHODS:
            row = f"  {method.upper():<15}"
            for b in range(self.n_bands):
                auprc, _ = band_results[b][method]
                row += f" {auprc:>14.2f}"
            lines.append(row)

        lines.append(sep)

        for line in lines:
            logger.info(line)

        os.makedirs(self.out_dir, exist_ok=True)
        summary_path = os.path.join(
            self.out_dir, f"summary_semantics_{self.dataset_type}_{self.model_name}.txt"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Summary saved to: {summary_path}")

    def _save_plots(
        self,
        boundary_results: Dict,
        flat_results: Dict,
        band_results: Dict,
    ) -> None:
        """Generate and save all visualisation plots for this analysis run."""
        apply_style()
        dataset_tag = f"{self.dataset_type}_{self.model_name}"

        auprc_data: Dict[str, Dict[str, float]] = {}
        for method in self.METHODS:
            auprc_data.setdefault("Boundary", {})[method] = boundary_results[method][0]
            auprc_data.setdefault("Flat",     {})[method] = flat_results[method][0]

        paths = plot_grouped_bar(
            data=auprc_data,
            metric_name="AuPRC (%)",
            title=f"AuPRC — Boundary vs Flat Region\n{self.model_name} | {self.dataset_type}",
            out_dir=self.out_dir,
            filename=f"boundary_auprc_{dataset_tag}",
        )
        logger.info(f"Saved boundary AuPRC plot: {paths}")

        fpr95_data: Dict[str, Dict[str, float]] = {}
        for method in self.METHODS:
            fpr95_data.setdefault("Boundary", {})[method] = boundary_results[method][1]
            fpr95_data.setdefault("Flat",     {})[method] = flat_results[method][1]

        paths = plot_grouped_bar(
            data=fpr95_data,
            metric_name="FPR@95 (%)",
            title=f"FPR@95 — Boundary vs Flat Region\n{self.model_name} | {self.dataset_type}",
            out_dir=self.out_dir,
            filename=f"boundary_fpr95_{dataset_tag}",
            higher_is_better=False,
        )
        logger.info(f"Saved boundary FPR95 plot: {paths}")

        band_labels = [depth_band_label(b, self.n_bands) for b in range(self.n_bands)]
        method_labels_list = [m.upper() for m in self.METHODS]

        auprc_matrix = np.array([
            [band_results[b][m][0] for b in range(self.n_bands)]
            for m in self.METHODS
        ])

        paths = plot_metric_heatmap(
            matrix=auprc_matrix,
            row_labels=method_labels_list,
            col_labels=band_labels,
            title=f"AuPRC (%) by Depth Band\n{self.model_name} | {self.dataset_type}",
            out_dir=self.out_dir,
            filename=f"depth_heatmap_auprc_{dataset_tag}",
            cmap="Blues",
        )
        logger.info(f"Saved depth AuPRC heatmap: {paths}")

        fpr95_matrix = np.array([
            [band_results[b][m][1] for b in range(self.n_bands)]
            for m in self.METHODS
        ])

        paths = plot_metric_heatmap(
            matrix=fpr95_matrix,
            row_labels=method_labels_list,
            col_labels=band_labels,
            title=f"FPR@95 (%) by Depth Band\n{self.model_name} | {self.dataset_type}",
            out_dir=self.out_dir,
            filename=f"depth_heatmap_fpr95_{dataset_tag}",
            cmap="Blues",
        )
        logger.info(f"Saved depth FPR95 heatmap: {paths}")

        self._plot_boundary_delta(boundary_results, flat_results, dataset_tag)

    def _plot_boundary_delta(
        self,
        boundary_results: Dict,
        flat_results: Dict,
        dataset_tag: str,
    ) -> None:
        """
        Plot the delta AuPRC(Flat) - AuPRC(Boundary) for each method.
        A large positive delta means the method degrades a lot near boundaries.
        """
        apply_style()
        fig, ax = plt.subplots(figsize=(8, 5))

        methods      = self.METHODS
        deltas_auprc = [
            flat_results[m][0] - boundary_results[m][0] for m in methods
        ]
        deltas_fpr95 = [
            boundary_results[m][1] - flat_results[m][1] for m in methods
        ]

        x = np.arange(len(methods))
        w = 0.35
        colors_auprc = [PALETTE.get(m, f"C{i}") for i, m in enumerate(methods)]
        colors_fpr95 = [PALETTE["boundary"]] * len(methods)

        bars1 = ax.bar(x - w / 2, deltas_auprc, w, label="ΔAuPRC (Flat−Boundary)",
                       color=colors_auprc, alpha=0.85, edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + w / 2, deltas_fpr95, w, label="ΔFPR95 (Boundary−Flat)",
                       color=colors_fpr95, alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.axhline(0, color=PALETTE["text"], linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.set_ylabel("Delta (%)")
        ax.set_title(
            f"Boundary Degradation Effect\n{self.model_name} | {self.dataset_type}"
        )
        ax.legend()

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.4 if h >= 0 else -1.2),
                f"{h:+.1f}",
                ha="center", va="bottom", fontsize=8, color=PALETTE["text"],
            )

        fig.tight_layout()
        saved = save_fig(fig, self.out_dir, f"boundary_delta_{dataset_tag}")
        logger.info(f"Saved boundary delta plot: {saved}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for semantic boundary/depth analysis."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            eval_semantics.py — Task 1: Semantic Boundary & Depth Analysis

            Computes AuPRC and FPR@TPR95 for:
              (a) boundary pixels (near semantic edges) vs flat pixels
              (b) horizontal depth bands (top=far, bottom=near)

            Pre-saved logit .pt files must exist in --logits_dir.
        """),
    )
    parser.add_argument(
        "--logits_dir",
        required=True,
        help="Directory with per-image .pt logit files (e.g. saved_logits/erfnet/fs_static).",
    )
    parser.add_argument(
        "--images_glob",
        required=True,
        help='Glob pattern matching input images (e.g. "Datasets/Fishyscapes/fs_static/images/*.jpg").',
    )
    parser.add_argument(
        "--dataset_type",
        required=True,
        choices=["fs_static", "road_anomaly", "road_anomaly21", "road_obstacle21", "lost_found", "streethazard"],
        help="Dataset type — determines GT label remapping.",
    )
    parser.add_argument(
        "--model_name",
        default="Model",
        help="Name of the model (used for plot titles and filenames).",
    )
    parser.add_argument(
        "--n_bands",
        type=int,
        default=cfg.analysis.n_bands,
        help=f"Number of horizontal depth bands. Default: {cfg.analysis.n_bands}.",
    )
    parser.add_argument(
        "--boundary_kernel_size",
        type=int,
        default=cfg.analysis.boundary_kernel_size,
        help=f"Morphological kernel size for boundary extraction (pixels). Default: {cfg.analysis.boundary_kernel_size}.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for plots and summary. Default: results/analysis/semantics/.",
    )
    return parser.parse_args()


def main() -> None:
    """Run semantic boundary/depth analysis from CLI arguments."""
    args    = parse_args()
    analyser = SemanticAnalyser(
        logits_dir      = args.logits_dir,
        images_glob     = args.images_glob,
        dataset_type    = args.dataset_type,
        model_name      = args.model_name,
        n_bands         = args.n_bands,
        boundary_kernel = args.boundary_kernel_size,
        out_dir         = args.out_dir,
    )
    analyser.run()


if __name__ == "__main__":
    main()
