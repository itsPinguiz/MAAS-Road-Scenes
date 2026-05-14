import os
import sys
import yaml
import importlib
import gc
from datetime import datetime

# FIX OOM: riduce la frammentazione della memoria CUDA su run lunghe
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Assicuriamoci che python estragga la base root del progetto per i moduli core.*
_SOLUTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SOLUTIONS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Aggiungiamo anche la directory di EoMT per mappare correttamente gli import dinamici
_EOMT_DIR = os.path.join(_ROOT, "third_party", "eomt")
if _EOMT_DIR not in sys.path:
    sys.path.insert(0, _EOMT_DIR)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import utility
from core.utility.config_loader import cfg
from core.utility.logger import logger, console

# Import nuovi moduli
from core.solutions.augmentation import PerspectiveOutlierPasting
from core.solutions.losses import CombinedFineTuningLoss
from core.solutions.dataset import OutlierAugmentedDataset
from third_party.eval.dataset import cityscapes

def build_model_eomt(checkpoint_path, device, num_classes=19, img_size=(1024, 1024)):
    """
    Ripristina la struttura dell'architettura EoMT e la ripopola con i pesi addestrati.
    """
    logger.info("Caricamento architettura EoMT statica...")
    from models.vit import ViT
    from models.eomt import EoMT
    from training.mask_classification_semantic import MaskClassificationSemantic

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
    ).to(device)

    # Load Checkpoint State Dict
    logger.info(f"Caricamento Pesi Pre-Addestrati da: {checkpoint_path}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        
        # FIX: Pulizia dei prefissi di PyTorch Lightning per matchare la struttura pura
        clean_state_dict = {}
        for k, v in state_dict.items():
            # Rimuoviamo i prefissi generati dai wrapper Lightning, mantendo la dicitura pura PyTorch ('network')
            new_k = k.replace("model.", "")
            clean_state_dict[new_k] = v
            
        # FIX: Manteniamo strict=True. Se fallisce ora, vogliamo che il programma si fermi!
        model.load_state_dict(clean_state_dict, strict=True)
        logger.success("Pesi originali EoMT caricati con successo (Strict=True)!")
    except Exception as e:
        logger.error(f"Errore critico nel caricamento dei pesi: {e}")
        raise e # Blocca l'esecuzione se i pesi non caricano
        
    return model

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Esegue una singola epoca di fine-tuning OOD sfruttando
    Bfloat16 per annullare l'Out-Of-Memory ed evitare overflow dei ViT.
    """
    model.train()
    
    total_ce_loss = 0.0
    total_ood_loss = 0.0
    
    w_ce = cfg.solutions.training.ce_loss_weight
    w_ood = cfg.solutions.training.ood_loss_weight

    for batch_idx, data in enumerate(dataloader):
        
        aug_images, aug_masks, ood_masks = data
        
        aug_images = aug_images.to(device)
        aug_masks = aug_masks.to(device)
        ood_masks = ood_masks.to(device)
        
        optimizer.zero_grad()
        
        # ========================================================
        # 1. FORWARD PASS in BFloat16 (Il FIX per i Vision Transformer)
        # ========================================================
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            import torch.nn.functional as F
            # LightningModule.forward divides by 255.0; keep model inputs in [0, 255]
            aug_images_for_model = aug_images * 255.0
            mask_logits_list, class_logits_list = model(aug_images_for_model)
            
            final_mask_logits = mask_logits_list[-1].float()
            final_class_logits = class_logits_list[-1].float()
            
            final_mask_logits = F.interpolate(
                final_mask_logits, 
                size=(aug_images.shape[2], aug_images.shape[3]), 
                mode="bilinear", 
                align_corners=False
            )

            dense_logits = model.to_per_pixel_logits_semantic(
                final_mask_logits, final_class_logits
            )
            
            # ========================================================
            # 2. LOSS CALCULATION IN FP32
            # ========================================================
            losses = loss_fn(dense_logits, aug_masks, ood_masks)
            
            loss_ce = losses["loss_ce"]
            loss_ood = losses["loss_ood"]
            
            loss = (w_ce * loss_ce) + (w_ood * loss_ood)
        
        # ========================================================
        # 3. BACKWARD NATIVO (Senza GradScaler)
        # ========================================================
        loss.backward()
        
        # FIX: Clipping dei gradienti per stabilizzare l'attention del ViT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_ce_loss += loss_ce.item()
        total_ood_loss += loss_ood.item()

        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)} | CE: {loss_ce.item():.4f} | OOD: {loss_ood.item():.4f}")
            
        del mask_logits_list, class_logits_list, final_mask_logits, final_class_logits, dense_logits, losses, loss, loss_ce, loss_ood
        torch.cuda.empty_cache()
        
        # FIX OOM: garbage collection periodico per evitare frammentazione su batch lunghi
        if batch_idx % 200 == 0:
            gc.collect()

    avg_ce = total_ce_loss / len(dataloader)
    avg_ood = total_ood_loss / len(dataloader)
    return avg_ce, avg_ood

def run_finetuning():
    logger.info("[bold magenta]Starting OOD Fine-Tuning Pipeline[/bold magenta]")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device attivo: [bold cyan]{device}[/bold cyan]", extra={"markup": True})
    
    # --- Checkpoint directory (save & resume) — from config ---
    ckpt_base_dir = os.path.join(cfg.paths.root, cfg.solutions.training.checkpoint_dir)

    # resume_from: path to a specific run folder (relative to project root or absolute).
    # When set, training resumes from the latest checkpoint inside that folder and
    # continues saving new epochs there. When null/empty, a new timestamped folder is created.
    resume_from = getattr(cfg.solutions.training, 'resume_from', None) or ''
    if resume_from:
        # Support both absolute paths and paths relative to project root
        if not os.path.isabs(resume_from):
            resume_from = os.path.join(cfg.paths.root, resume_from)
        ckpt_dir = resume_from
        logger.info(f"[resume_from] Ripresa da run esistente: {ckpt_dir}")
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(ckpt_base_dir, run_timestamp)
        logger.info(f"Nuova run — checkpoint dir: {ckpt_dir}")

    os.makedirs(ckpt_dir, exist_ok=True)
    
    # --- Base pretrained weights (from config) ---
    base_ckpt_path = os.path.join(cfg.paths.root, cfg.paths.models.eomt_checkpoint)
    
    model = build_model_eomt(base_ckpt_path, device, num_classes=19, img_size=(1024, 1024))
    
    # Congela l'Encoder DINOv2 per proteggere le feature stradali da catastrophic forgetting
    for param in model.network.encoder.parameters():
        param.requires_grad = False
    
    loss_fn = CombinedFineTuningLoss(
        ood_loss_type=cfg.solutions.training.ood_loss_type, 
        ignore_index=19
    ).to(device)
    
    # Optimizer compatibile con parametri frizzati (lr da config)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.solutions.training.lr,
        weight_decay=1e-4
    )
    
    from torchvision.transforms import Compose, ToTensor
    import glob
    from PIL import Image
    import numpy as np

    class PILToLongTensorWrap:
        def __call__(self, pic):
            return torch.from_numpy(np.array(pic)).long()

    class LocalCityscapesDataset:
        def __init__(self, root_dir, subset='train', transform=None, target_transform=None):
            search_pattern = os.path.join(root_dir, "**", subset, "**", "*leftImg8bit.png")
            self.images = sorted(glob.glob(search_pattern, recursive=True))
            
            self.labels = []
            for img_path in self.images:
                gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest') \
                                  .replace('leftImg8bit', 'gtFine') \
                                  .replace('.png', '_labelIds.png')
                self.labels.append(gt_path)
                
            self.transform = transform
            self.target_transform = target_transform
            
            self.mapping_256 = np.ones(256, dtype=np.uint8) * 19
            mapping = {
                7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
            }
            for k, v in mapping.items():
                self.mapping_256[k] = v
            
            if len(self.images) == 0:
                logger.error(f"Errore fatale: Nessuna immagine trovata con pattern:\n{search_pattern}")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            gt_path = self.labels[idx]
            
            try:
                img = Image.open(img_path).convert('RGB')
                label_raw_np = np.array(Image.open(gt_path))
            except OSError as e:
                logger.warning(f"File corrotto o non leggibile trovato a '{img_path}' o '{gt_path}'. Salto all'indice successivo. Dettaglio: {e}")
                return self.__getitem__((idx + 1) % len(self.images))
            
            label_mapped_np = self.mapping_256[label_raw_np]
            label = Image.fromarray(label_mapped_np)
            
            w, h = img.size
            if w > 1024 or h > 1024:
                import random
                th, tw = 1024, 1024
                top = random.randint(0, max(0, h - th))
                left = random.randint(0, max(0, w - tw))
                img = img.crop((left, top, left + tw, top + th))
                label = label.crop((left, top, left + tw, top + th))
            
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                label = self.target_transform(label)
                
            return img, label, img_path, gt_path

    cityscapes_root = os.path.join(cfg.paths.root, "Datasets", "Cityscapes") 
    
    base_dataset = LocalCityscapesDataset(
        root_dir=cityscapes_root,
        transform=Compose([
            ToTensor()
        ]),
        target_transform=Compose([PILToLongTensorWrap()]),
        subset='train'
    )
    
    outliers_dir = os.path.join(cfg.paths.root, "Datasets", "Outliers_COCO")
    if not os.path.isdir(outliers_dir):
        outliers_dir = os.path.join(cfg.paths.root, "Datasets", "COCO", "Outliers_COCO")
    if (not os.path.isdir(outliers_dir)) or (len(glob.glob(os.path.join(outliers_dir, "*.png"))) == 0):
        logger.warning(f"Outlier directory missing or empty: {outliers_dir}. OOD loss may stay near zero.")
    
    dataset = OutlierAugmentedDataset(
        base_dataset=base_dataset,
        outliers_dir=outliers_dir,
        ignore_index=19
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.solutions.training.batch_size,
        shuffle=True,
        num_workers=cfg.solutions.training.num_workers,
        persistent_workers=True,   # FIX: evita corruzione shared-memory tra worker e main process
    )
    
    # --- Resume or fresh train ---
    # Priority: resume_from (specific run folder) > fresh start
    import glob as _glob
    start_epoch = 1

    if resume_from:
        existing_ckpts = sorted(
            _glob.glob(os.path.join(ckpt_dir, "epoch_*_EoMT.pth")),
            key=lambda p: int(os.path.basename(p).split('_')[1])
        )
        if existing_ckpts:
            latest_ckpt = existing_ckpts[-1]
            start_epoch = int(os.path.basename(latest_ckpt).split('_')[1]) + 1
            logger.info(f"[resume_from] Caricamento checkpoint: {latest_ckpt} (prossima epoch: {start_epoch})")
            resume_sd = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(resume_sd, strict=True)
            logger.success(f"Checkpoint caricato — ripresa da epoch {start_epoch}")
        else:
            logger.warning(f"[resume_from] Nessun checkpoint trovato in {ckpt_dir} — training da zero.")
    else:
        logger.info("Training da zero (nuova run).")

    epochs = cfg.solutions.training.epochs
    for epoch in range(start_epoch, epochs + 1):
        logger.info(f"\n[bold green]=== Start Epoch {epoch}/{epochs} ===[/bold green]")
        
        avg_ce, avg_ood = train_epoch(model, dataloader, optimizer, loss_fn, device)
        
        logger.success(f"End Epoch {epoch} | Avg CE Loss: {avg_ce:.4f} | Avg OOD Loss: {avg_ood:.4f}")
        
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}_EoMT.pth")
        try:
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Checkpoint salvato: {ckpt_path}")
        except Exception as e:
            logger.error(f"Errore salvataggio checkpoint: {e}")

if __name__ == "__main__":
    run_finetuning()