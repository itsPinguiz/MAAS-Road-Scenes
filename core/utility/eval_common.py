"""Shared evaluation helpers for anomaly and Cityscapes scripts."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from core.utility.config_loader import cfg

IGNORE_INDEX = cfg.eval.ignore_index

CITYSCAPES_ID_TO_TRAIN_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}

OOD_DATASETS = {
    "Fishyscapes Static": cfg.paths.datasets.fishyscapes_static,
    "Fishyscapes Lost & Found": cfg.paths.datasets.fishyscapes_lost_found,
    "RoadAnomaly": cfg.paths.datasets.road_anomaly,
    "RoadAnomaly21": cfg.paths.datasets.road_anomaly21,
    "RoadObsticle21": cfg.paths.datasets.road_obstacle21,
}


def cityscapes_label_mapping(ignore_index: int = 255) -> np.ndarray:
    """Build a 256-entry Cityscapes labelId-to-trainId lookup table."""
    mapping = np.ones(256, dtype=np.uint8) * ignore_index
    for label_id, train_id in CITYSCAPES_ID_TO_TRAIN_ID.items():
        mapping[label_id] = train_id
    return mapping


def anomaly_scores(logits: torch.Tensor, include_rba: bool = False) -> Dict[str, torch.Tensor]:
    """Compute common dense anomaly score maps from logits."""
    probs = F.softmax(logits, dim=1)
    scores = {
        "msp": 1.0 - probs.max(dim=1).values,
        "maxlogit": -logits.max(dim=1).values,
        "maxentropy": -(probs * torch.log(probs + 1e-12)).sum(dim=1),
    }
    if include_rba:
        scores["rba"] = -torch.sum(torch.tanh(logits), dim=1)
    return scores


def anomaly_scores_numpy(logits: torch.Tensor, include_rba: bool = False) -> Dict[str, np.ndarray]:
    """Compute common anomaly scores as squeezed CPU numpy arrays."""
    return {
        method: score.squeeze(0).detach().cpu().numpy()
        for method, score in anomaly_scores(logits, include_rba=include_rba).items()
    }


def compute_auprc(scores: np.ndarray, labels: np.ndarray, ignore_index: int = IGNORE_INDEX) -> float:
    """Compute AuPRC over valid pixels, returning NaN for one-class inputs."""
    valid = labels != ignore_index
    scores = scores[valid]
    labels = labels[valid]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return average_precision_score(labels, scores) * 100.0


def load_logits(logit_path: str) -> torch.Tensor:
    """Load saved logits and normalize shape to [B, C, H, W]."""
    logits = torch.load(logit_path, map_location="cpu")
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    return logits.float()


def gt_path_from_image(image_path: str, dataset_type: str) -> str:
    """Derive the ground-truth mask path from an input image path."""
    gt_path = image_path.replace("images", "labels_masks")
    if dataset_type == "road_obstacle21":
        gt_path = gt_path.replace(".webp", ".png")
    elif dataset_type == "fs_static":
        gt_path = gt_path.replace(".jpg", ".png")
    elif dataset_type in ("road_anomaly", "road_anomaly21"):
        gt_path = gt_path.replace(".jpg", ".png")
    return gt_path


def load_ood_mask(gt_path: str, target_size: Tuple[int, int], dataset_type: str) -> np.ndarray:
    """Load and remap a dataset-specific OOD mask to binary/ignore labels."""
    mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"GT mask not found: {gt_path}")

    mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    if dataset_type == "road_anomaly":
        mask = np.where(mask == 2, 1, mask).astype(np.uint8)
    elif dataset_type == "streethazard":
        remapped = np.full_like(mask, IGNORE_INDEX)
        remapped[mask < 20] = 0
        remapped[mask >= 20] = 1
        remapped[mask == 14] = IGNORE_INDEX
        mask = remapped

    if IGNORE_INDEX != 255:
        mask[mask == 255] = IGNORE_INDEX
    return mask.astype(np.uint8)


def load_state_dict_flexible(model: torch.nn.Module, state_dict: dict) -> torch.nn.Module:
    """Load checkpoints that may or may not include a DataParallel prefix."""
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            own_state[name].copy_(param)
        elif name.startswith("module."):
            clean_name = name.split("module.", 1)[1]
            if clean_name in own_state:
                own_state[clean_name].copy_(param)
    return model
