# Copyright (c) OpenMMLab. All rights reserved.

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
import os.path as osp
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", ".*'network' is an instance.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger, console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

def compute_msp_anomaly_score(logits):
    probs = F.softmax(logits, dim=1)
    msp, _ = torch.max(probs, dim=1)
    return 1.0 - msp 

def compute_maxlogit_anomaly_score(logits):
    max_logit, _ = torch.max(logits, dim=1)
    return -max_logit

def compute_maxentropy_anomaly_score(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
    return entropy

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 20
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([Resize((512, 1024), Image.BILINEAR), ToTensor()])
target_transform = Compose([Resize((512, 1024), Image.NEAREST)])

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", help="Glob pattern per le immagini")  
    parser.add_argument('--loadDir', default=resolve_path("../trained_models/"))
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    # --- NUOVI ARGOMENTI PER LOGITS ---
    parser.add_argument('--save_logits', action='store_true', help='Salva i logits su disco')
    parser.add_argument('--dataset_name', default='default', help='Nome cartella per i logits')
    
    args = parser.parse_args()

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    model = ERFNet(NUM_CLASSES)
    
    device = torch.device("cpu") if args.cpu else DEVICE
    
    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=device))
    model.eval()
    
    input_paths = glob.glob(os.path.expanduser(str(args.input[0])))
    if not input_paths:
        logger.error(f"\n[!] ERROR: No images found: {args.input[0]}")
        return

    # Liste per calcolo simultaneo
    anomaly_score_lists = {'msp': [], 'maxlogit': [], 'maxentropy': []}
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

        for path in input_paths:
            base_name = osp.splitext(osp.basename(path))[0]
            save_dir = osp.join("saved_logits", "erfnet", args.dataset_name.replace(" ", "_"))
            save_path = osp.join(save_dir, f"{base_name}.pt")
            
            # --- LOGICA IBRIDA: CARICAMENTO O INFERENZA ---
            if osp.exists(save_path):
                # Carica i logit pre-calcolati (velocissimo, non serve GPU)
                result = torch.load(save_path, map_location=device)
            else:
                # Esegue l'inferenza solo se i logit non esistono
                img_pil = Image.open(path).convert('RGB')
                images = input_transform(img_pil).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    result = model(images)
                
                
                # Salva i logit per la prossima volta se richiesto
                if args.save_logits:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(result.cpu(), save_path)
            # ----------------------------------------------

            # Calcolo punteggi per i 3 metodi classici
            msp_res = compute_msp_anomaly_score(result).squeeze(0).data.cpu().numpy()
            maxlogit_res = compute_maxlogit_anomaly_score(result).squeeze(0).data.cpu().numpy()
            maxent_res = compute_maxentropy_anomaly_score(result).squeeze(0).data.cpu().numpy()

            # Gestione Ground Truth
            pathGT = path.replace("images", "labels_masks")                
            if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")                
            if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")  

            mask = Image.open(pathGT)
            mask = target_transform(mask)
            ood_gts = np.array(mask)

            # Mappatura OOD standard
            if "RoadAnomaly" in pathGT:
                ood_gts = np.where((ood_gts==2), 1, ood_gts)
            elif "LostAndFound" in pathGT:
                ood_gts = np.where((ood_gts==0), 255, ood_gts)
                ood_gts = np.where((ood_gts==1), 0, ood_gts)
                ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
            elif "Streethazard" in pathGT:
                ood_gts = np.where((ood_gts==14), 255, ood_gts)
                ood_gts = np.where((ood_gts<20), 0, ood_gts)
                ood_gts = np.where((ood_gts==255), 1, ood_gts)

            if 1 in np.unique(ood_gts):
                ood_gts_list.append(ood_gts)
                anomaly_score_lists['msp'].append(msp_res)
                anomaly_score_lists['maxlogit'].append(maxlogit_res)
                anomaly_score_lists['maxentropy'].append(maxent_res)
                
            del result, msp_res, maxlogit_res, maxent_res, ood_gts, mask
            if device.type == "cuda": torch.cuda.empty_cache()
            progress.advance(task_id)

    if not ood_gts_list:
        logger.warning("No valid OOD pixels found.")
        return

    ood_gts_all = np.concatenate([gt.flatten() for gt in ood_gts_list])
    ood_mask = (ood_gts_all == 1)
    ind_mask = (ood_gts_all == 0)

    # Output metriche per il wrapper
    for method in ['msp', 'maxlogit', 'maxentropy']:
        scores_all = np.concatenate([s.flatten() for s in anomaly_score_lists[method]])
        
        ood_out = scores_all[ood_mask]
        ind_out = scores_all[ind_mask]

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((np.zeros(len(ind_out)), np.ones(len(ood_out))))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)
        
        logger.info(f"Method: {method.upper()} -> AUPRC score: {prc_auc*100.0:.2f}, FPR@TPR95: {fpr*100.0:.2f}")

if __name__ == '__main__':
    main()