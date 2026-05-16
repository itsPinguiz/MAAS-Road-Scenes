import os
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EVAL_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.utility.config_loader import cfg
import torch
import numpy as np
import glob
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from argparse import ArgumentParser
import warnings

from core.utility.eval_common import OOD_DATASETS, cityscapes_label_mapping
from core.utility.runtime import get_device, print_device_health

DEVICE = get_device()

warnings.filterwarnings("ignore", ".*'network' is an instance.*")

from core.utility.update_table_t import update_table_t_entry
from core.utility.logger import logger, console
from ood_metrics import fpr_at_95_tpr
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from torchmetrics import JaccardIndex

def get_msp_score(logits, T, device):
    """Compute MSP anomaly scores after temperature scaling."""
    scaled_logits = logits / T
    probs = F.softmax(scaled_logits, dim=1)
    msp, _ = torch.max(probs, dim=1)
    return (1.0 - msp).squeeze().data.cpu().numpy()

def main():
    """Evaluate configured temperatures on saved logits and update TABLE_T."""
    print_device_health(DEVICE)

    parser = ArgumentParser()
    parser.add_argument('--model', required=True, choices=['ERFNET', 'EOMT'])
    parser.add_argument('--logits_dir', required=True)
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else DEVICE
    logger.info(f"Using device: [bold cyan]{device}[/bold cyan]", extra={"markup": True})

    TEMPS = cfg.eval.temperatures
    DATASETS = {**OOD_DATASETS, "Cityscapes": cfg.paths.datasets.cityscapes}
    mapping_256 = cityscapes_label_mapping(ignore_index=255)

    for ds_name, ds_pattern in DATASETS.items():
        ds_slug = ds_name.replace(" ", "_")
        ckpt_name = os.path.splitext(os.path.basename(cfg.paths.models.eomt_checkpoint if args.model == 'EOMT' else cfg.paths.models.erfnet_weights))[0]
        current_logits_path = os.path.join(args.logits_dir, ckpt_name, ds_slug)
        logit_files = glob.glob(os.path.join(current_logits_path, "*.pt"))
        
        if not logit_files:
            logger.warning(f"Skipping {ds_name}: No logits found in {current_logits_path}")
            continue

        if ds_name != 'Cityscapes':
            best_metrics = {"auprc": -1, "fpr95": 100, "t": None}
    
            for T in TEMPS:
                all_scores, all_gts = [], []
                
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    task_id = progress.add_task(f"{ds_name} (T={T})", total=len(logit_files))
                    
                    for l_path in logit_files:
                        logits = torch.load(l_path, map_location=device).to(device)
                        score = get_msp_score(logits, T, device)
                        
                        base_name = os.path.basename(l_path).replace(".pt", "")
                        img_ref_list = glob.glob(ds_pattern.replace("*", base_name))
                        if not img_ref_list: 
                            progress.advance(task_id)
                            continue
                        
                        gt_path = img_ref_list[0].replace("images", "labels_masks")
                        if "RoadObsticle21" in gt_path: gt_path = gt_path.replace(".webp", ".png")
                        if "fs_static" in gt_path: gt_path = gt_path.replace(".jpg", ".png")
                        if "RoadAnomaly" in gt_path: gt_path = gt_path.replace(".jpg", ".png")
    
                        gt_img = Image.open(gt_path).resize((score.shape[1], score.shape[0]), Image.NEAREST)
                        gt_np = np.array(gt_img)
    
                        if "RoadAnomaly" in ds_name: 
                            gt_np = np.where(gt_np==2, 1, gt_np)
                        elif "LostAndFound" in ds_name:
                            gt_np = np.where(gt_np==0, 255, gt_np)
                            gt_np = np.where(gt_np==1, 0, gt_np)
                            gt_np = np.where((gt_np>1)&(gt_np<201), 1, gt_np)
    
                        mask = (gt_np != 255)
                        if np.any(mask):
                            all_scores.append(score[mask])
                            all_gts.append(gt_np[mask])
                        del logits
                        
                        progress.advance(task_id)
    
                if not all_gts: continue
                
                y_true = np.concatenate(all_gts)
                y_scores = np.concatenate(all_scores)
                
                auprc = average_precision_score(y_true, y_scores) * 100
                fpr95 = fpr_at_95_tpr(y_scores, y_true) * 100
                
                logger.info(f" {ds_name} | T={T} -> AuPRC: {auprc:.2f}, FPR95: {fpr95:.2f}")
                update_table_t_entry(args.model, f"MSP (t = {T})", ds_name, f"{auprc:.2f}", f"{fpr95:.2f}")
                
                if auprc > best_metrics["auprc"]:
                    best_metrics.update({"auprc": auprc, "fpr95": fpr95, "t": T})
        
        if ds_name == 'Cityscapes' and logit_files:
            T_ref = 1.0
            metric = JaccardIndex(task="multiclass", num_classes=19, ignore_index=255).to(device)
            
            for l_path in logit_files:
                logits = torch.load(l_path, map_location=device).to(device)
                preds = torch.argmax(logits, dim=1)
                
                base_name = os.path.basename(l_path).replace(".pt", "")
                img_ref_list = []
                for root, dirs, files in os.walk(ds_pattern):
                    for file in files:
                        if base_name in file and file.endswith(".png"):
                            img_ref_list.append(os.path.join(root, file))
                            break
                            
                if not img_ref_list: continue
                
                img_path = img_ref_list[0]
                gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest') \
                                  .replace('leftImg8bit', 'gtFine') \
                                  .replace('.png', '_labelIds.png')
                
                if not os.path.exists(gt_path): continue
                
                label_raw_np = np.array(Image.open(gt_path))
                label_mapped_np = mapping_256[label_raw_np]
                label_tensor = torch.from_numpy(label_mapped_np).to(device)
                
                if preds.shape[-2:] != label_tensor.shape[-2:]:
                    preds_reshaped = F.interpolate(preds.unsqueeze(1).float(), 
                                                 size=label_tensor.shape[-2:], 
                                                 mode='nearest').squeeze().long()
                else:
                    preds_reshaped = preds.squeeze()
                    
                metric.update(preds_reshaped, label_tensor)
                del logits, preds
            
            miou = metric.compute().item() * 100
            logger.info(f" [bold green]mIoU (Cityscapes): {miou:.2f}%[/bold green]")
            
            for T in TEMPS:
                update_table_t_entry(args.model, f"MSP (t = {T})", miou=f"{miou:.2f}")
            update_table_t_entry(args.model, "MSP (best t)", miou=f"{miou:.2f}")

        if best_metrics["t"] is not None:
            update_table_t_entry(args.model, "MSP (best t)", ds_name, 
                                 f"{best_metrics['auprc']:.2f}", f"{best_metrics['fpr95']:.2f}")

if __name__ == "__main__":
    main()
