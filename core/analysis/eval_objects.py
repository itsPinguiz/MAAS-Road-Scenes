"""
eval_objects.py — TASK 2: Analisi per Tipologia di Oggetto
==========================================================

Analyses WHERE the anomaly detection model fails by grouping errors based on:
  1. Connected Component Bucketing: Finding connected components in GT masks, separating
     small, medium, and large anomalies, and checking AuPRC for each size bracket.
  2. In-Distribution Confusion: Over false negative pixels, what class does the model
     erroneously predict?
  3. Things vs Stuff: Grouping those predicted classes into macro categories.

INPUT
-----
Pre-saved logits and the corresponding images/labels.

USAGE
-----
python core/analysis/eval_objects.py \
    --logits_dir   core/evaluation/eval/saved_logits/erfnet/Fishyscapes_Static \
    --images_glob  "Datasets/Fishyscapes/fs_static/images/*.jpg" \
    --dataset_type fs_static \
    --model_name   ERFNet 
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.utility.logger import logger
from core.utility.config_loader import cfg
from core.analysis.plot_utils import (
    apply_style, save_fig, plot_grouped_bar, PALETTE,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IGNORE_INDEX = cfg.eval.ignore_index

# Cityscapes Classes (19 eval classes)
CS_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
]

CS_THINGS = {
    "pole", "traffic light", "traffic sign", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
}

CS_STUFF = {
    "road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky"
}

# Ensure coverage
assert len(CS_THINGS) + len(CS_STUFF) == len(CS_CLASSES)

def compute_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    valid = labels != IGNORE_INDEX
    scores = scores[valid]
    labels = labels[valid]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return average_precision_score(labels, scores) * 100.0

def load_logits(logit_path: str) -> torch.Tensor:
    t = torch.load(logit_path, map_location="cpu")
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.float()

def load_gt_mask(gt_path: str, target_size: Tuple[int, int], dataset_type: str) -> np.ndarray:
    mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"GT mask not found: {gt_path}")
    mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    if dataset_type == "road_anomaly":
        mask = np.where(mask == 2, 1, mask).astype(np.uint8)
    elif dataset_type == "streethazard":
        out = np.full_like(mask, IGNORE_INDEX)
        out[mask < 20] = 0
        out[mask >= 20] = 1
        out[mask == 14] = IGNORE_INDEX
        mask = out
    # Final safety remap: some datasets use 255 for ignore, but we want to stick to config's ignore_index
    if IGNORE_INDEX != 255:
        mask[mask == 255] = IGNORE_INDEX

    return mask.astype(np.uint8)

def gt_path_from_image(image_path: str, dataset_type: str) -> str:
    gt = image_path.replace("images", "labels_masks")
    if dataset_type == "road_obstacle21":
        gt = gt.replace(".webp", ".png")
    elif dataset_type == "fs_static":
        gt = gt.replace(".jpg", ".png")
    elif dataset_type in ("road_anomaly", "road_anomaly21"):
        gt = gt.replace(".jpg", ".png")
    return gt

def compute_all_scores(logits: torch.Tensor) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    probs = F.softmax(logits, dim=1)
    msp = 1.0 - probs.max(dim=1).values
    maxlogit = -logits.max(dim=1).values
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
    
    # Argmax for class prediction
    preds = logits.argmax(dim=1).squeeze(0).numpy()

    scores = {
        "msp": msp.squeeze(0).numpy(),
        "maxlogit": maxlogit.squeeze(0).numpy(),
        "maxentropy": entropy.squeeze(0).numpy(),
    }
    return scores, preds

class ObjectAnalyser:
    METHODS = ["msp", "maxlogit", "maxentropy"]

    def __init__(
        self,
        logits_dir: str,
        images_glob: str,
        dataset_type: str,
        model_name: str = "Model",
        thresh_small: int = None,
        thresh_medium: int = None,
        out_dir: str = None,
    ):
        if not os.path.isabs(logits_dir):
            logits_dir = os.path.normpath(os.path.join(cfg.paths.root, logits_dir))
        self.logits_dir = logits_dir
        self.images_glob = images_glob
        self.dataset_type = dataset_type
        self.model_name = model_name
        self.thresh_small = thresh_small or cfg.analysis.thresh_small
        self.thresh_medium = thresh_medium or cfg.analysis.thresh_medium
        self.out_dir = out_dir or os.path.join(cfg.paths.root, cfg.analysis.reports_dir, "objects")
        
        # Sizing accumulators
        self._size_acc: Dict[str, Dict[str, Dict[str, List]]] = {
            "Small": {m: {"scores": [], "labels": []} for m in self.METHODS},
            "Medium": {m: {"scores": [], "labels": []} for m in self.METHODS},
            "Large": {m: {"scores": [], "labels": []} for m in self.METHODS},
        }

        # Baseline In-distribution vs All OOD (for comparison)
        self._baseline_acc: Dict[str, Dict[str, List]] = {m: {"scores": [], "labels": []} for m in self.METHODS}

        # Confusion accumulators over all OOD pixels (not just FN logically, but FN implies they have an argmax)
        # Actually, let's collect the class predictions for all OOD pixels. 
        # But specifically, we are interested in "false negatives" where anomaly score < threshold.
        # Without a fixed threshold, we can just look at *all* OOD pixels and see what the model classifies them as.
        self._ood_class_counts = np.zeros(len(CS_CLASSES), dtype=np.int64)

    def run(self):
        if not os.path.isdir(self.logits_dir):
            logger.error(f"Logits directory not found: {self.logits_dir}")
            return

        image_paths = sorted(glob.glob(os.path.expanduser(self.images_glob)))
        if not image_paths:
            logger.error(f"No images found: {self.images_glob}")
            return

        logger.info(f"Object analysis | model={self.model_name} | dataset={self.dataset_type}")

        for img_path in image_paths:
            basename = Path(img_path).stem
            logit_path = os.path.join(self.logits_dir, f"{basename}.pt")
            if not os.path.exists(logit_path):
                continue
            gt_path = gt_path_from_image(img_path, self.dataset_type)
            if not os.path.exists(gt_path):
                continue
            self._process_image(logit_path, gt_path)

        size_metrics = self._compute_metrics()
        self._save_summary(size_metrics)
        self._save_plots(size_metrics)

    def _process_image(self, logit_path: str, gt_path: str):
        logits = load_logits(logit_path)
        H, W = logits.shape[2], logits.shape[3]
        gt_mask = load_gt_mask(gt_path, (H, W), self.dataset_type)

        if 1 not in np.unique(gt_mask):
            return

        scores_dict, preds = compute_all_scores(logits)

        # Baseline
        for method in self.METHODS:
            self._baseline_acc[method]["scores"].append(scores_dict[method].flatten())
            self._baseline_acc[method]["labels"].append(gt_mask.flatten())

        # Connected components on OOD (gt == 1)
        # Binary mask for components
        bin_ood = (gt_mask == 1).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_ood, connectivity=8)

        # Build masks for sizes
        small_mask = np.zeros_like(gt_mask, dtype=bool)
        med_mask = np.zeros_like(gt_mask, dtype=bool)
        large_mask = np.zeros_like(gt_mask, dtype=bool)

        for l in range(1, num_labels):
            area = stats[l, cv2.CC_STAT_AREA]
            comp_mask = (labels == l)
            if area < self.thresh_small:
                small_mask |= comp_mask
            elif area < self.thresh_medium:
                med_mask |= comp_mask
            else:
                large_mask |= comp_mask

        # Accumulate component metrics
        # For evaluation, we evaluate the component positive pixels mixed with ALL negative pixels.
        # This properly evaluates metrics "detecting small objects vs detecting large ones".
        # To do this, we need the in-distribution pixel indices.
        ind_mask = (gt_mask == 0)
        
        def push_acc(bucket_name, bucket_mask):
            mask = bucket_mask | ind_mask
            filtered_gt = gt_mask[mask]
            for method in self.METHODS:
                filtered_scores = scores_dict[method][mask]
                self._size_acc[bucket_name][method]["scores"].append(filtered_scores)
                self._size_acc[bucket_name][method]["labels"].append(filtered_gt)

        push_acc("Small", small_mask)
        push_acc("Medium", med_mask)
        push_acc("Large", large_mask)

        # Confusion: aggregate predictions for OOD pixels
        ood_preds = preds[bin_ood == 1]
        for p in ood_preds:
            if 0 <= p < len(CS_CLASSES):
                self._ood_class_counts[p] += 1

    def _compute_metrics(self):
        res = {"Small": {}, "Medium": {}, "Large": {}}
        for bucket in res.keys():
            for m in self.METHODS:
                sc = np.concatenate(self._size_acc[bucket][m]["scores"]) if self._size_acc[bucket][m]["scores"] else np.empty(0)
                la = np.concatenate(self._size_acc[bucket][m]["labels"]) if self._size_acc[bucket][m]["labels"] else np.empty(0)
                if len(sc) > 0 and len(np.unique(la)) > 1:
                    res[bucket][m] = compute_auprc(sc, la)
                else:
                    res[bucket][m] = float('nan')
        return res

    def _save_summary(self, size_metrics):
        os.makedirs(self.out_dir, exist_ok=True)
        lines = []
        lines.append(f"Model: {self.model_name} | Dataset: {self.dataset_type}")
        lines.append("-" * 30)
        for bucket in ["Small", "Medium", "Large"]:
            lines.append(f"Size: {bucket}")
            for m in self.METHODS:
                lines.append(f"  {m:10}: {size_metrics[bucket][m]:.2f}%")

        # Confusion
        lines.append("\nClass Prediction on OOD Pixels:")
        total_ood = self._ood_class_counts.sum()
        if total_ood > 0:
            for i, c in enumerate(CS_CLASSES):
                cnt = self._ood_class_counts[i]
                if cnt > 0:
                    lines.append(f"  {c:15}: {cnt/total_ood*100:.1f}%")

        with open(os.path.join(self.out_dir, f"summary_objects_{self.dataset_type}_{self.model_name}.txt"), "w") as f:
            f.write("\n".join(lines))

    def _save_plots(self, size_metrics):
        apply_style()
        dataset_tag = f"{self.dataset_type}_{self.model_name}"

        # 1. Grouped Bar for sizes
        auprc_data = {
            "Small": size_metrics["Small"],
            "Medium":  size_metrics["Medium"],
            "Large": size_metrics["Large"],
        }
        plot_grouped_bar(
            data=auprc_data,
            metric_name="AuPRC (%)",
            title=f"AuPRC by Anomaly Size Bucket\n{self.model_name} | {self.dataset_type}",
            out_dir=self.out_dir,
            filename=f"size_auprc_{dataset_tag}"
        )

        # 2. Confusion Bar Chart
        if self._ood_class_counts.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            pcts = self._ood_class_counts / self._ood_class_counts.sum() * 100
            
            # Sort by pct
            idx = np.argsort(pcts)
            
            classes = [CS_CLASSES[i] for i in idx if pcts[i] > 0.5]
            vals = [pcts[i] for i in idx if pcts[i] > 0.5]

            bars = ax.barh(classes, vals, color=PALETTE["accent"], alpha=0.8)
            ax.set_xlabel("% of OOD Pixels Classified As")
            ax.set_title(f"In-Distribution 'Confusion' for Anomalies\n{self.model_name} | {self.dataset_type}")

            for bar in bars:
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        f"{bar.get_width():.1f}%", va='center', color=PALETTE["text"], fontsize=9)
            
            fig.tight_layout()
            save_fig(fig, self.out_dir, f"confusion_bar_{dataset_tag}")

        # 3. Things vs Stuff Pie or Bar
        things_cnt = sum(self._ood_class_counts[CS_CLASSES.index(c)] for c in CS_THINGS)
        stuff_cnt = sum(self._ood_class_counts[CS_CLASSES.index(c)] for c in CS_STUFF)
        tot = things_cnt + stuff_cnt
        if tot > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie([things_cnt, stuff_cnt], labels=["Things", "Stuff"],
                   colors=[PALETTE["msp"], PALETTE["maxlogit"]], textprops={'color': "white"},
                   autopct='%1.1f%%', startangle=90)
            ax.set_title("OOD pixels classified as Things vs Stuff")
            fig.tight_layout()
            save_fig(fig, self.out_dir, f"things_vs_stuff_{dataset_tag}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits_dir", required=True)
    parser.add_argument("--images_glob", required=True)
    parser.add_argument("--dataset_type", required=True)
    parser.add_argument("--model_name", default="Model")
    parser.add_argument("--out_dir", default=None, help="Output directory for plots and summary.")
    return parser.parse_args()

def main():
    args = parse_args()
    analyser = ObjectAnalyser(
        logits_dir=args.logits_dir,
        images_glob=args.images_glob,
        dataset_type=args.dataset_type,
        model_name=args.model_name,
        out_dir=args.out_dir,
    )
    analyser.run()

if __name__ == "__main__":
    main()
