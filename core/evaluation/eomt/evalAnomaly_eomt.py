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

import os
import glob
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
import torch.nn.functional as F

from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore", ".*'network' is an instance.*")

from core.utility.logger import logger, console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from training.lightning_module import LightningModule
from models.vit import ViT
from models.eomt import EoMT
from training.mask_classification_semantic import MaskClassificationSemantic

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

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
    
def compute_rba_anomaly_score(logits):
    # logits shape: (B, C, H, W)
    return -torch.sum(torch.tanh(logits), dim=1)

import yaml
import importlib
from huggingface_hub import hf_hub_download

def load_eomt_model(ckpt_path):
    
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

    # Caricamento dei pesi
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        
        # Pulizia delle chiavi di PyTorch Lightning se necessario (rimuove "network." se il modello base è già network)
        # Ma nel nostro caso, MaskClassificationSemantic ha un attributo self.network, quindi le chiavi dovrebbero combaciare!
        
        # Mettiamo strict=True. Se fallisce ora, significa che abbiamo ancora un mismatch, ma con questi parametri non dovrebbe!
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        # Fallback senza strict nel caso ci siano chiavi extra non importanti, ma avvisiamo l'utente
        logger.warning("Attempting fallback with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    return model

def get_dense_logits(model, img_tensor, crop_batch_size=2):
    """
    Versione ottimizzata con Crop Batching e Automatic Mixed Precision.
    """
    # 1. Taglia l'immagine grande in crop
    crops, origins = model.window_imgs_semantic(img_tensor)
    
    final_mask_logits_list = []
    final_class_logits_list = []
    
    if not torch.is_tensor(crops):
        crops = torch.stack(crops)
        
    num_crops = crops.shape[0]
    
    # 2. Usa AMP (Mixed Precision) per dimezzare l'uso della VRAM
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        
        # Gestisce i crop in batch
        for i in range(0, num_crops, crop_batch_size):
            crop_batch = crops[i:i+crop_batch_size]
            
            # Forward pass per questo specifico batch di crop
            mask_logits_per_layer, class_logits_per_layer = model(crop_batch)
            
            # Salviamo solo l'ultimo layer e forziamo a float32 per evitare problemi matematici dopo
            final_mask_logits_list.append(mask_logits_per_layer[-1].float())
            final_class_logits_list.append(class_logits_per_layer[-1].float())
            
            # Puliamo la memoria subito dopo il forward pass
            del mask_logits_per_layer, class_logits_per_layer
            torch.cuda.empty_cache()
            
    # Uniamo i risultati calcolati singolarmente
    final_mask_logits = torch.cat(final_mask_logits_list, dim=0)
    final_class_logits = torch.cat(final_class_logits_list, dim=0)
    
    # Interpolazione alla dimensione nativa del crop (1024x1024)
    final_mask_logits = F.interpolate(final_mask_logits, model.img_size, mode="bilinear")
    
    # 3. Ricostruisce i logit densi per pixel (H, W) per ogni crop
    crop_logits = model.to_per_pixel_logits_semantic(
        final_mask_logits, final_class_logits
    )
    
    # 4. Ricuce assieme i crop per formare l'immagine ad alta risoluzione originale
    img_sizes = [img.shape[-2:] for img in img_tensor]
    dense_logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
    
    return dense_logits[0].unsqueeze(0), final_mask_logits, final_class_logits

from torch.utils.data import Dataset, DataLoader

class AnomalyDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img_pil = Image.open(path).convert('RGB')
        w, h = img_pil.size
        img_np_temp = np.array(img_pil)
        img_tensor = torch.from_numpy(img_np_temp).permute(2, 0, 1).float() / 255.0
        return img_tensor, path, w, h

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default=cfg.paths.datasets.road_obstacle21,
        nargs="+",
        help="A list of space separated input images; or a single glob pattern",
    )  
    parser.add_argument('--ckpt_path', default=cfg.paths.models.eomt_checkpoint)
    parser.add_argument('--save_logits', action='store_true', help='Save dense reconstructed logits to disk')
    parser.add_argument('--dataset_name', default='default_dataset', help='Name of the dataset for organizing saved logits folder')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for computation (e.g., "cpu", "cuda:0")')
    parser.add_argument('--quiet', action='store_true', help='Minimal output for bulk runs')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of background workers for DataLoader')
    parser.add_argument('--crop-batch-size', type=int, default=2, help='Batch size for crop window inference')
    args = parser.parse_args()

    # Dictionary to store anomaly scores for each method
    anomaly_scores_all = { 'msp': [], 'maxlogit': [], 'maxentropy': [], 'rba': [] }
    ood_gts_list = []

    model = load_eomt_model(args.ckpt_path)
    device = DEVICE
    model = model.to(device)
    model.eval()
    
    input_paths = glob.glob(os.path.expanduser(str(args.input[0])))
    if not input_paths:
        logger.error(f"No images found for pattern: {args.input[0]}")
        return

    dataset = AnomalyDataset(input_paths)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=args.quiet
    ) as progress:
        task_id = progress.add_task("Evaluating EoMT", total=len(input_paths))

        for batch_idx, (img_tensor_batch, path_batch, w_batch, h_batch) in enumerate(dataloader):
            path = path_batch[0]
            w = w_batch[0].item()
            h = h_batch[0].item()
            img_tensor = img_tensor_batch.to(device)

            base_name = osp.splitext(osp.basename(path))[0]
            ckpt_name = osp.splitext(osp.basename(args.ckpt_path))[0]
            save_dir = osp.join("saved_logits", "eomt", ckpt_name, args.dataset_name.replace(" ", "_"))
            save_path = osp.join(save_dir, f"{base_name}.pt")

            # --- LOGICA IBRIDA: CARICAMENTO O INFERENZA ---
            if osp.exists(save_path):
                # Caricamento istantaneo (funziona su CPU/GPU)
                dense_logits = torch.load(save_path, map_location=device)
            else:
                # Esecuzione inferenza (Richiede GPU)
                with torch.no_grad():
                    dense_logits, _, _ = get_dense_logits(model, img_tensor, crop_batch_size=args.crop_batch_size)
                    
                    if args.save_logits:
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(dense_logits.cpu(), save_path)
    
            # Calcolo simultaneo dei 4 punteggi
            anomaly_result_msp = compute_msp_anomaly_score(dense_logits).squeeze(0).data.cpu().numpy()
            anomaly_result_maxlogit = compute_maxlogit_anomaly_score(dense_logits).squeeze(0).data.cpu().numpy()
            anomaly_result_maxentropy = compute_maxentropy_anomaly_score(dense_logits).squeeze(0).data.cpu().numpy()
            anomaly_result_rba = compute_rba_anomaly_score(dense_logits).squeeze(0).data.cpu().numpy()
    
            # --- GESTIONE GROUND TRUTH ---
            pathGT = path.replace("images", "labels_masks")                
            if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")                
            if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")  
    
            try:
                mask_img = Image.open(pathGT)
                # Usiamo le dimensioni w, h ottenute sopra
                mask_img = mask_img.resize((w, h), Image.NEAREST)
                ood_gts = np.array(mask_img)
            except Exception as e:
                logger.error(f"Could not load mask for {pathGT}: {e}")
                progress.advance(task_id)
                continue
    
            # Mappature classi (RoadAnomaly, LostAndFound, etc.)
            if "RoadAnomaly" in pathGT:
                ood_gts = np.where((ood_gts==2), 1, ood_gts)
            if "LostAndFound" in pathGT:
                ood_gts = np.where((ood_gts==0), 255, ood_gts)
                ood_gts = np.where((ood_gts==1), 0, ood_gts)
                ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
            if "Streethazard" in pathGT:
                ood_gts = np.where((ood_gts==14), 255, ood_gts)
                ood_gts = np.where((ood_gts<20), 0, ood_gts)
                ood_gts = np.where((ood_gts==255), 1, ood_gts)
    
            if 1 in np.unique(ood_gts):
                ood_gts_list.append(ood_gts)
                anomaly_scores_all['msp'].append(anomaly_result_msp)
                anomaly_scores_all['maxlogit'].append(anomaly_result_maxlogit)
                anomaly_scores_all['maxentropy'].append(anomaly_result_maxentropy)
                anomaly_scores_all['rba'].append(anomaly_result_rba)
                
            # Pulizia memoria
            del dense_logits, anomaly_result_msp, anomaly_result_maxlogit, anomaly_result_maxentropy, anomaly_result_rba, ood_gts, mask_img
            if img_tensor is not None: del img_tensor
            torch.cuda.empty_cache()
            progress.advance(task_id)
        
    if len(ood_gts_list) == 0:
        logger.warning("No valid evaluations found.")
        return
    
    # Flatten lists for metrics
    ood_gts = np.concatenate([gt.flatten() for gt in ood_gts_list])
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    if not args.quiet:
        logger.info("\n=======================================")
        logger.info(f"Model:   EoMT")
        logger.info(f"Dataset: {args.dataset_name}")
        logger.info("---------------------------------------")
    
    # Evaluate for each method
    for method in ['msp', 'maxlogit', 'maxentropy', 'rba']:
        anomaly_scores = np.concatenate([score.flatten() for score in anomaly_scores_all[method]])

        # We only care about positive (1) and negative (0) OOD classes. 
        # Ignore index is 255 etc.
        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        method_label = method.upper() if method != 'maxentropy' else 'MAX ENTROPY'
        
        if args.quiet:
            logger.info(f"[Metrics] Model: EoMT, Dataset: {args.dataset_name}, Method: {method_label}, AuPRC: {prc_auc*100.0:.2f}, FPR95: {fpr*100.0:.2f}")
        else:
            logger.info(f"Method:  {method_label}")
            logger.info(f"AuPRC:   {prc_auc*100.0:.2f}")
            logger.info(f"FPR95:   {fpr*100.0:.2f}")
            logger.info("---------------------------------------")
        
        # Update TABLE.md directly
        from core.utility.update_table import update_table_entry
        update_table_entry(model="EoMT", method=method_label, dataset=args.dataset_name, miou='-', auprc=f"{prc_auc*100.0:.2f}", fpr95=f"{fpr*100.0:.2f}")
        
    if not args.quiet:
        logger.info("=======================================\n")
    file.close()

if __name__ == '__main__':
    main()
