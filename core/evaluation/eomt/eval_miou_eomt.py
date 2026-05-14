# --- ENVIRONMENT SETUP BLOCK ---
import os
import sys

# Add project root to path so core.* is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Add third_party/eomt to path for models and training
_TP_EOMT = os.path.join(_ROOT, "third_party", "eomt")
if _TP_EOMT not in sys.path:
    sys.path.append(_TP_EOMT)

# Add third_party/eval to path for dataset and transform
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

import sys
import os
# Append the eval path to import existing dataloader structures without duplication
# Append the eval path to import existing dataloader structures without duplication
sys.path.append(os.path.join(_ROOT, "core", "evaluation", "eval"))

import warnings
warnings.filterwarnings("ignore", ".*'network' is an instance.*")

from core.utility.logger import logger, console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

import torch
import time
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from torchmetrics import JaccardIndex

from dataset import cityscapes
from transform import Relabel, ToLabel
from evalAnomaly_eomt import get_dense_logits

def load_eomt_model(ckpt_path):
    import torch
    from models.vit import ViT
    from models.eomt import EoMT
    from training.mask_classification_semantic import MaskClassificationSemantic
    
    # Parametri esatti estratti dal file .ckpt
    img_size = (1024, 1024)
    num_classes = 19
    
    encoder = ViT(
        img_size=img_size, 
        patch_size=16, 
        backbone_name="vit_base_patch14_reg4_dinov2"
    )

    network = EoMT(
        num_q=100,
        encoder=encoder,
        num_blocks=3,
        masked_attn_enabled=True,
        num_classes=num_classes,
    )

    model = MaskClassificationSemantic(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        attn_mask_annealing_enabled=True,
    ).eval()

    # Caricamento dei pesi — supporta sia .ckpt (Lightning) che .pth (fine-tuned plain state_dict)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error(f"Impossibile aprire il checkpoint '{ckpt_path}': {e}")
        raise

    # .ckpt Lightning wraps weights under "state_dict" with a "model." prefix.
    # .pth fine-tuned checkpoints saved by train.py are plain state_dicts.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw = ckpt["state_dict"]
        state_dict = {k.replace("model.", "", 1): v for k, v in raw.items()}
        strict = True
    else:
        state_dict = ckpt  # already a plain state_dict
        strict = True

    try:
        model.load_state_dict(state_dict, strict=strict)
        logger.success(f"Pesi caricati con successo (strict={strict}): {ckpt_path}")
    except RuntimeError as e:
        logger.warning(f"strict=True fallito: {e}\nRitento con strict=False...")
        model.load_state_dict(state_dict, strict=False)

    return model


# 19 cityscapes classes + 1 ignored class mapped to 19
NUM_CLASSES = 20
IGNORE_INDEX = 19

def main():
    parser = ArgumentParser()
    parser.add_argument('--datadir', default=cfg.paths.root)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--ckpt_path', default=cfg.paths.models.eomt_checkpoint)
    parser.add_argument('--device', default='cuda:0', help='Device to use for computation')
    parser.add_argument('--quiet', action='store_true', help='Minimal output for bulk runs')
    parser.add_argument('--save_logits', action='store_true', help='Save logits to disk')
    parser.add_argument('--crop-batch-size', type=int, default=2, help='Batch size for crop window inference')
    args = parser.parse_args()

    model = load_eomt_model(args.ckpt_path)
    device = DEVICE
    model = model.to(device)
    model.eval()

    import os
    import numpy as np
    from PIL import Image
    from torchmetrics import JaccardIndex
    from tqdm import tqdm

    # Ricerca infallibile con os.walk
    image_paths = []
    for root, dirs, files in os.walk(args.datadir):
        # Controlliamo che sia dentro una cartella 'val' per non valutare il train set
        if 'val' in root.split(os.sep): 
            for file in files:
                if file.endswith("leftImg8bit.png"):
                    image_paths.append(os.path.join(root, file))

    if len(image_paths) == 0:
        logger.error(f"ERRORE CRITICO: Nessuna immagine trovata in {args.datadir}")
        return

    # Metrica: 19 classi (ignorando l'indice 255)
    metric = JaccardIndex(task="multiclass", num_classes=19, ignore_index=255).to(device)

    valid_images_count = 0

    # Mappatura standard di Cityscapes da LabelIds (0-33) a TrainIds (0-18).
    # Tutto ciò che non è in questo dizionario viene mappato a 255 (Background/Ignored)
    mapping_256 = np.ones(256, dtype=np.uint8) * 255
    cityscapes_mapping = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
    }
    for k, v in cityscapes_mapping.items():
        mapping_256[k] = v

    valid_paths = []
    for img_path in image_paths:
        gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest') \
                          .replace('leftImg8bit', 'gtFine') \
                          .replace('.png', '_labelIds.png')
        if os.path.exists(gt_path):
            valid_paths.append((img_path, gt_path))
        else:
            logger.warning(f"ATTENZIONE: Manca la label per {img_path}\nCercata in: {gt_path}")

    class CityscapesValDataset(Dataset):
        def __init__(self, paths, mapping):
            self.paths = paths
            self.mapping = mapping

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img_path, gt_path = self.paths[idx]
            img_np = np.array(Image.open(img_path).convert('RGB'))
            label_raw_np = np.array(Image.open(gt_path))
            label_mapped_np = self.mapping[label_raw_np]
            
            # Keep inputs in [0, 255]; LightningModule.forward scales by 1/255.
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            label_tensor = torch.from_numpy(label_mapped_np)
            return img_tensor, label_tensor, img_path

    dataset = CityscapesValDataset(valid_paths, mapping_256)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=args.quiet
    ) as progress:
        task_id = progress.add_task("Evaluating images", total=len(dataset))

        for batch_idx, (img_tensor_batch, label_tensor_batch, img_path_batch) in enumerate(dataloader):
            img_path = img_path_batch[0]
            img_tensor = img_tensor_batch.to(device)
            label_tensor = label_tensor_batch.to(device)

            with torch.no_grad():
                dense_logits, _, _ = get_dense_logits(model, img_tensor, crop_batch_size=args.crop_batch_size)
                preds = torch.argmax(dense_logits, dim=1)
                metric.update(preds, label_tensor)
                
                if args.save_logits:
                    ckpt_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
                    save_dir = os.path.join("saved_logits", "eomt", ckpt_name, "Cityscapes")
                    os.makedirs(save_dir, exist_ok=True)
                    # Save as .pt file with the same basename as image
                    save_name = os.path.basename(img_path).replace(".png", ".pt")
                    torch.save(dense_logits.cpu(), os.path.join(save_dir, save_name))
                
            # Alla fine del ciclo for, dopo metric.update()
            del dense_logits, preds, img_tensor, label_tensor
            torch.cuda.empty_cache()
            progress.advance(task_id)

    # Calcola il risultato finale
    mIoU = metric.compute().item()
    miou_val = mIoU * 100
    if args.quiet:
        logger.info(f"[Metrics] Model: EoMT, Dataset: Cityscapes, Method: mIoU, mIoU: {miou_val:.2f}%")
    else:
        logger.info(f"\n[bold]Model:[/bold]   EoMT")
        logger.info(f"[bold]Dataset:[/bold] Cityscapes")
        logger.info(f"[bold]Method:[/bold]  N/A (mIoU)")
        logger.info(f"---------------------------------------")
        logger.info(f"[bold cyan]mIoU:[/bold cyan]    {miou_val:.2f}%")
        logger.info(f"=======================================\n")

    import sys
    from core.utility.update_table import update_table_entry
    
    miou_str = f"{mIoU * 100:.2f}"
    # Aggiorniamo la colonna mIoU per tutti i metodi di EoMT (incluso RbA)
    for method in ['MSP', 'MaxLogit', 'Max Entropy', 'RbA']:
        update_table_entry(model="EoMT", method=method, miou=miou_str)
    
    logger.success("Tabella TABLE.md aggiornata con successo con la mIoU di EoMT!")

if __name__ == '__main__':
    main()
