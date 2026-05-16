"""Run ERFNet OOD evaluation on one dataset.

Execution flow:
1. Parse CLI arguments and choose a device.
2. Load ERFNet weights.
3. For each image, load cached logits or run inference.
4. Load the matching OOD mask and accumulate score maps.
5. Report AuPRC and FPR@TPR95 for MSP, MaxLogit, and MaxEntropy.
"""

import os
import sys
import os.path as osp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TP_EVAL = os.path.join(_ROOT, "third_party", "eval")
if _TP_EVAL not in sys.path:
    sys.path.append(_TP_EVAL)

from argparse import ArgumentParser
import glob
import random
import warnings

import numpy as np
import torch
from PIL import Image
from erfnet import ERFNet
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor

from core.utility.config_loader import cfg
from core.utility.logger import logger, console
from core.utility.runtime import get_device, print_device_health
from core.utility.eval_common import anomaly_scores_numpy, load_state_dict_flexible
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

warnings.filterwarnings("ignore", ".*'network' is an instance.*")

DEVICE = get_device()
SEED = 42
NUM_CLASSES = 20
METHODS = ("msp", "maxlogit", "maxentropy")
LOGIT_SIZE = (512, 1024)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([Resize(LOGIT_SIZE, Image.BILINEAR), ToTensor()])
target_transform = Compose([Resize(LOGIT_SIZE, Image.NEAREST)])


def parse_args():
    """Parse command-line arguments for ERFNet anomaly evaluation."""
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", help="Glob pattern for input images")
    parser.add_argument('--loadDir', default=cfg.paths.models.trained_models_dir + os.sep)
    parser.add_argument('--loadWeights', default=cfg.paths.models.erfnet_weights)
    parser.add_argument('--loadModel', default=cfg.paths.models.erfnet_model)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--save_logits', action='store_true', help='Save logits to disk')
    parser.add_argument('--dataset_name', default='default', help='Dataset folder name for saved logits')
    return parser.parse_args()


def resolve_weights_path(load_dir: str, load_weights: str) -> str:
    """Resolve absolute and legacy loadDir-relative checkpoint paths."""
    if os.path.isabs(load_weights) and os.path.exists(load_weights):
        return load_weights
    return os.path.join(load_dir, load_weights)


def load_model(weightspath: str, device: torch.device) -> torch.nn.Module:
    """Build ERFNet, load checkpoint weights, and switch to eval mode."""
    logger.info(f"Loading weights from: [cyan]{weightspath}[/cyan]")

    model = ERFNet(NUM_CLASSES)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)

    model = load_state_dict_flexible(model, torch.load(weightspath, map_location=device))
    model.eval()
    return model


def logits_cache_path(image_path: str, load_weights: str, dataset_name: str) -> str:
    """Return the cache path used for saved ERFNet logits."""
    base_name = osp.splitext(osp.basename(image_path))[0]
    ckpt_name = osp.splitext(osp.basename(load_weights))[0]
    save_dir = osp.join("saved_logits", "erfnet", ckpt_name, dataset_name.replace(" ", "_"))
    return osp.join(save_dir, f"{base_name}.pt")


def get_or_compute_logits(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    save_path: str,
    save_logits: bool,
) -> torch.Tensor:
    """Load cached logits when present, otherwise run ERFNet inference."""
    if osp.exists(save_path):
        return torch.load(save_path, map_location=device)

    img_pil = Image.open(image_path).convert("RGB")
    images = input_transform(img_pil).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(images)

    if save_logits:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        torch.save(logits.cpu(), save_path)
    return logits


def gt_path_from_input(image_path: str) -> str:
    """Derive the OOD ground-truth mask path from the input image path."""
    gt_path = image_path.replace("images", "labels_masks")
    if "RoadObsticle21" in gt_path:
        gt_path = gt_path.replace("webp", "png")
    if "fs_static" in gt_path:
        gt_path = gt_path.replace("jpg", "png")
    if "RoadAnomaly" in gt_path:
        gt_path = gt_path.replace("jpg", "png")
    return gt_path


def load_ood_ground_truth(image_path: str) -> np.ndarray:
    """Load the matching OOD mask and apply dataset-specific remapping."""
    gt_path = gt_path_from_input(image_path)
    mask = Image.open(gt_path)
    ood_gts = np.array(target_transform(mask))

    if "RoadAnomaly" in gt_path:
        ood_gts = np.where(ood_gts == 2, 1, ood_gts)
    elif "LostAndFound" in gt_path:
        ood_gts = np.where(ood_gts == 0, 255, ood_gts)
        ood_gts = np.where(ood_gts == 1, 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    elif "Streethazard" in gt_path:
        ood_gts = np.where(ood_gts == 14, 255, ood_gts)
        ood_gts = np.where(ood_gts < 20, 0, ood_gts)
        ood_gts = np.where(ood_gts == 255, 1, ood_gts)
    return ood_gts


def collect_scores(input_paths, model, device, args):
    """Run the dataset loop and collect score maps plus ground-truth masks."""
    anomaly_score_lists = {method: [] for method in METHODS}
    ood_gts_list = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=args.quiet
    ) as progress:
        task_id = progress.add_task("Evaluating ERFNet", total=len(input_paths))

        for image_path in input_paths:
            save_path = logits_cache_path(image_path, args.loadWeights, args.dataset_name)
            logits = get_or_compute_logits(image_path, model, device, save_path, args.save_logits)
            score_maps = anomaly_scores_numpy(logits)
            ood_gts = load_ood_ground_truth(image_path)

            if 1 in np.unique(ood_gts):
                ood_gts_list.append(ood_gts)
                for method, score in score_maps.items():
                    anomaly_score_lists[method].append(score)

            del logits, score_maps, ood_gts
            if device.type == "cuda":
                torch.cuda.empty_cache()
            progress.advance(task_id)

    return anomaly_score_lists, ood_gts_list


def log_metrics(anomaly_score_lists, ood_gts_list) -> None:
    """Compute and log AuPRC/FPR95 for every anomaly scoring method."""
    if not ood_gts_list:
        logger.warning("No valid OOD pixels found.")
        return

    ood_gts_all = np.concatenate([gt.flatten() for gt in ood_gts_list])
    ood_mask = (ood_gts_all == 1)
    ind_mask = (ood_gts_all == 0)

    for method in METHODS:
        scores_all = np.concatenate([score.flatten() for score in anomaly_score_lists[method]])
        ind_out = scores_all[ind_mask]
        ood_out = scores_all[ood_mask]

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((np.zeros(len(ind_out)), np.ones(len(ood_out))))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        logger.info(
            f"Method: {method.upper()} -> AUPRC score: {prc_auc * 100.0:.2f}, "
            f"FPR@TPR95: {fpr * 100.0:.2f}"
        )


def main():
    """Evaluate ERFNet anomaly metrics for one dataset glob."""
    print_device_health(DEVICE)
    args = parse_args()

    device = torch.device("cpu") if args.cpu else DEVICE
    weightspath = resolve_weights_path(args.loadDir, args.loadWeights)
    model = load_model(weightspath, device)

    input_paths = glob.glob(os.path.expanduser(str(args.input[0])))
    if not input_paths:
        logger.error(f"\n[!] ERROR: No images found: {args.input[0]}")
        return

    anomaly_score_lists, ood_gts_list = collect_scores(input_paths, model, device, args)
    log_metrics(anomaly_score_lists, ood_gts_list)

if __name__ == '__main__':
    main()
