# --- ENVIRONMENT SETUP BLOCK ---
import os
import sys

# 1. Conditional Environment Detection
IS_COLAB = 'google.colab' in sys.modules

# 2. Hybrid Pathing
if IS_COLAB:
    BASE_PATH = '/content/drive/MyDrive/Project'
    # Optional: Automatically install dependencies if on Colab
    import subprocess
    print("Checking requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', os.path.join(BASE_PATH, 'requirements.txt')])
    except Exception as e:
        print(f"Warning: Could not install requirements: {e}")
else:
    BASE_PATH = '.'

def resolve_path(relative_path):
    """ Helper to resolve paths consistently between environments. """
    if IS_COLAB and relative_path.startswith('../'):
        relative_path = relative_path.lstrip('../')
    return os.path.join(BASE_PATH, relative_path)

# 3. Unified Device Logic
import torch
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# 4. GPU Health Check
def print_gpu_health(device):
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        details = f"[bold]Hardware environment:[/bold] {device.type.upper()}\n"
        if device.type == 'cuda':
            details += f"CUDA Device: {torch.cuda.get_device_name(device)}\n"
            vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            details += f"Available VRAM: {vram:.2f} GB"
        elif device.type == 'mps':
            details += "Apple Silicon (MPS) detected."
        else:
            details += "[yellow]Running on CPU. Performance will be limited.[/yellow]"
        console.print(Panel(details, title="[bold blue]GPU Health Check[/bold blue]", border_style="blue", expand=False))
    except ImportError:
        pass

print_gpu_health(DEVICE)
# --- END SETUP BLOCK ---

import os
import torch
import numpy as np
import glob
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
import sys
from argparse import ArgumentParser
import warnings

warnings.filterwarnings("ignore", ".*'network' is an instance.*")

# Import utility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from update_table_t import update_table_t_entry
from ood_metrics import fpr_at_95_tpr
from logger import logger, console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

def get_msp_score(logits, T, device):
    scaled_logits = logits / T
    probs = F.softmax(scaled_logits, dim=1)
    msp, _ = torch.max(probs, dim=1)
    return (1.0 - msp).squeeze().data.cpu().numpy()

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', required=True, choices=['ERFNET', 'EOMT'])
    parser.add_argument('--logits_dir', required=True)
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else DEVICE
    logger.info(f"Using device: [bold cyan]{device}[/bold cyan]", extra={"markup": True})

    TEMPS = [0.5, 0.75, 1.1]
    DATASETS = {
        'RoadAnomaly21': resolve_path('Datasets/SegmentMeIfYouCan/RoadAnomaly21/images/*.png'),
        'RoadObsticle21': resolve_path('Datasets/SegmentMeIfYouCan/RoadObsticle21/images/*.webp'),
        'Fishyscapes Lost & Found': resolve_path('Datasets/Fishyscapes/FS_LostFound_full/images/*.png'),
        'Fishyscapes Static': resolve_path('Datasets/Fishyscapes/fs_static/images/*.jpg'),
        'RoadAnomaly': resolve_path('Datasets/RoadAnomaly/images/*.jpg')
    }

    for ds_name, ds_pattern in DATASETS.items():
        ds_slug = ds_name.replace(" ", "_")
        current_logits_path = os.path.join(args.logits_dir, ds_slug)
        logit_files = glob.glob(os.path.join(current_logits_path, "*.pt"))
        
        if not logit_files:
            logger.warning(f"Skipping {ds_name}: No logits found in {current_logits_path}")
            continue

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
                    
                    # Mapping preciso della GT
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

                    # Mappature OOD
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
                    del logits # Free the large tensor
                    
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
        
        if best_metrics["t"] is not None:
            update_table_t_entry(args.model, "MSP (best t)", ds_name, 
                                 f"{best_metrics['auprc']:.2f}", f"{best_metrics['fpr95']:.2f}")

if __name__ == "__main__":
    main()