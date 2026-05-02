# --- ENVIRONMENT SETUP BLOCK ---
import os
import sys

# Add project root to path so core.* is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Add third_party/eval to path for erfnet and utilities
_TP_EVAL = os.path.join(_ROOT, "third_party", "eval")
if _TP_EVAL not in sys.path:
    sys.path.append(_TP_EVAL)

from core.utility.config_loader import cfg

# Unified Device Logic
import torch
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# GPU Health Check
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
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore", ".*'network' is an instance.*")

from core.utility.logger import logger, console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import numpy as np

from erfnet import ERFNet
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry

NUM_CLASSES = 20
IGNORE_INDEX = 19

input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, IGNORE_INDEX),   # ignore label to 19
])

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                continue
        else:
            own_state[name].copy_(param)
    return model

def main(args):
    modelpath = os.path.join(args.loadDir, args.loadModel)
    # If the weight path from config is already absolute and valid, use it directly.
    # Otherwise, join with loadDir for backward compatibility.
    if os.path.isabs(args.loadWeights) and os.path.exists(args.loadWeights):
        weightspath = args.loadWeights
    else:
        weightspath = os.path.join(args.loadDir, args.loadWeights)
    
    logger.info(f"Loading weights from: [cyan]{weightspath}[/cyan]")
    
    model = ERFNet(NUM_CLASSES)

    device = torch.device('cpu') if args.cpu else DEVICE

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    model = load_my_state_dict(model, torch.load(weightspath, map_location=device))
    model.eval()

    # Ricerca infallibile con os.walk (come fatto in EoMT)
    image_paths = []
    for root, dirs, files in os.walk(args.datadir):
        if 'val' in root.split(os.sep): 
            for file in files:
                if file.endswith("leftImg8bit.png"):
                    image_paths.append(os.path.join(root, file))

    if len(image_paths) == 0:
        logger.error(f"ERRORE CRITICO: Nessuna immagine trovata in {args.datadir}")
        return

    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=IGNORE_INDEX)
    
    # 1. Definiamo la mappatura da labelIds (0-33) a trainIds (0-18)
    mapping_256 = np.ones(256, dtype=np.uint8) * 255
    cityscapes_mapping = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
    }
    for k, v in cityscapes_mapping.items():
        mapping_256[k] = v

    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=IGNORE_INDEX)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=args.quiet
    ) as progress:
        task_id = progress.add_task("Evaluating ERFNet", total=len(image_paths))

        for img_path in image_paths:
            gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest') \
                              .replace('leftImg8bit', 'gtFine') \
                              .replace('.png', '_labelIds.png')
            
            if not os.path.exists(gt_path):
                progress.advance(task_id)
                continue

            # 2. Carichiamo le immagini
            img = Image.open(img_path).convert('RGB')
            
            # 3. Carichiamo la maschera RAW e applichiamo la mappatura prima di trasformarla
            label_raw_np = np.array(Image.open(gt_path))
            label_mapped_np = mapping_256[label_raw_np]
            gt = Image.fromarray(label_mapped_np) # Riconvertiamo in PIL Image per i transform

            img_t = input_transform_cityscapes(img).unsqueeze(0).to(device)
            gt_t = target_transform_cityscapes(gt).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_t)
                
            pred = outputs.max(1)[1].unsqueeze(1).data
            
            # Ora i dati sono corretti alla radice, ma manteniamo la sicurezza
            gt_t[gt_t >= NUM_CLASSES] = IGNORE_INDEX
            pred[pred >= NUM_CLASSES] = IGNORE_INDEX
            
            iouEvalVal.addBatch(pred, gt_t)

            if args.save_logits:
                ckpt_name = os.path.splitext(os.path.basename(args.loadWeights))[0]
                save_dir = os.path.join("saved_logits", "erfnet", ckpt_name, "Cityscapes")
                os.makedirs(save_dir, exist_ok=True)
                # Save as .pt file with the same basename as image
                save_name = os.path.basename(img_path).replace(".png", ".pt")
                torch.save(outputs.cpu(), os.path.join(save_dir, save_name))

            progress.advance(task_id)

    iouVal, iou_classes = iouEvalVal.getIoU()

    miou_val = iouVal.item() * 100
    if args.quiet:
        logger.info(f"[Metrics] Model: ERFNet, Dataset: Cityscapes, Method: mIoU, mIoU: {miou_val:.2f}%")
    else:
        logger.info("\n[bold]Model:[/bold]   ERFNet")
        logger.info("[bold]Dataset:[/bold] Cityscapes")
        logger.info("[bold]Method:[/bold]  N/A (mIoU)")
        logger.info("---------------------------------------")
        logger.info(f"[bold cyan]mIoU:[/bold cyan]    {miou_val:.2f}%")
        logger.info("=======================================\n")

    from core.utility.update_table import update_table_entry
    
    miou_str = f"{iouVal.item() * 100:.2f}"
    # Aggiorniamo la colonna mIoU per tutti i metodi di ERFNet
    for method in ['MSP', 'MaxLogit', 'Max Entropy']:
        update_table_entry(model="ERFNET", method=method, miou=miou_str)
    
    logger.success("Tabella TABLE.md aggiornata con successo con la mIoU di ERFNet!")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default=cfg.paths.models.trained_models_dir + os.sep)
    parser.add_argument('--loadWeights', default=cfg.paths.models.erfnet_weights)
    parser.add_argument('--loadModel', default=cfg.paths.models.erfnet_model)
    parser.add_argument('--datadir', default=cfg.paths.datasets.cityscapes)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--quiet', action='store_true', help='Minimal output for bulk runs')
    parser.add_argument('--save_logits', action='store_true', help='Save logits to disk')

    main(parser.parse_args())