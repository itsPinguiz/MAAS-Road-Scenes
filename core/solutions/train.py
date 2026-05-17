import os
import sys
import gc
from datetime import datetime

# Reduce CUDA memory fragmentation on long runs.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_SOLUTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SOLUTIONS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_EOMT_DIR = os.path.join(_ROOT, "third_party", "eomt")
if _EOMT_DIR not in sys.path:
    sys.path.insert(0, _EOMT_DIR)

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.utility.config_loader import cfg
from core.utility.logger import logger

from core.solutions.losses import CombinedFineTuningLoss
from core.solutions.dataset import OutlierAugmentedDataset

def build_model_eomt(checkpoint_path, device, num_classes=19, img_size=(1024, 1024)):
    """Build the EoMT architecture and load pretrained weights."""
    logger.info("Loading EoMT architecture...")
    from models.vit import ViT
    from models.eomt import EoMT
    from training.mask_classification_semantic import MaskClassificationSemantic

    encoder = ViT(
        img_size=img_size, 
        patch_size=16, 
        backbone_name="vit_base_patch14_reg4_dinov2",
        ckpt_path=checkpoint_path,
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

    logger.info(f"Loading pretrained weights from: {checkpoint_path}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("model.", "")
            clean_state_dict[new_k] = v
            
        model.load_state_dict(clean_state_dict, strict=True)
        logger.success("Original EoMT weights loaded successfully (strict=True).")
    except Exception as e:
        logger.error(f"Critical error while loading weights: {e}")
        raise e
        
    return model

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch: int):
    """Run one OOD fine-tuning epoch."""
    model.train()
    
    total_ce_loss = 0.0
    total_ood_loss = 0.0
    total_ood_ratio = 0.0
    batches_with_ood = 0
    is_cuda = device.type == "cuda"
    
    w_ce = cfg.solutions.training.ce_loss_weight
    target_w_ood = cfg.solutions.training.ood_loss_weight
    warmup_epochs = max(1, getattr(cfg.solutions.training, "ood_loss_warmup_epochs", 1))
    w_ood = target_w_ood * min(1.0, epoch / warmup_epochs)

    for batch_idx, data in enumerate(dataloader):
        
        aug_images, aug_masks, ood_masks = data
        
        aug_images = aug_images.to(device, non_blocking=is_cuda)
        aug_masks = aug_masks.to(device, non_blocking=is_cuda)
        ood_masks = ood_masks.to(device, non_blocking=is_cuda)
        ood_ratio = ood_masks.float().mean().item()
        total_ood_ratio += ood_ratio
        if ood_masks.any():
            batches_with_ood += 1
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=is_cuda):
            aug_images_for_model = aug_images * 255.0
            mask_logits_list, class_logits_list = model(aug_images_for_model)
            
            final_mask_logits = mask_logits_list[-1]
            final_class_logits = class_logits_list[-1]

            dense_logits_low_res = model.to_per_pixel_logits_semantic(
                final_mask_logits, final_class_logits
            )
            
            dense_logits = F.interpolate(
                dense_logits_low_res.float(), 
                size=(aug_images.shape[2], aug_images.shape[3]), 
                mode="bilinear", 
                align_corners=False
            )
            
            losses = loss_fn(dense_logits, aug_masks, ood_masks)
            
            loss_ce = losses["loss_ce"]
            loss_ood = losses["loss_ood"]
            
            loss = (w_ce * loss_ce) + (w_ood * loss_ood)
        
        loss.backward()
        
        # Keep ViT attention updates stable during fine-tuning.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_ce_loss += loss_ce.item()
        total_ood_loss += loss_ood.item()

        if batch_idx % 10 == 0:
            logger.info(
                f"Batch {batch_idx}/{len(dataloader)} | CE: {loss_ce.item():.4f} | "
                f"OOD: {loss_ood.item():.4f} | w_ood: {w_ood:.4f} | "
                f"OOD px: {ood_ratio * 100:.3f}%"
            )
            
        del mask_logits_list, class_logits_list, final_mask_logits, final_class_logits, dense_logits, losses, loss, loss_ce, loss_ood
        if is_cuda and batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        if batch_idx % 200 == 0:
            gc.collect()

    if len(dataloader) == 0:
        raise RuntimeError("Empty dataloader: check the Cityscapes path and dataset filters.")

    avg_ce = total_ce_loss / len(dataloader)
    avg_ood = total_ood_loss / len(dataloader)
    avg_ood_ratio = total_ood_ratio / len(dataloader)
    ood_batch_rate = batches_with_ood / len(dataloader)
    return avg_ce, avg_ood, avg_ood_ratio, ood_batch_rate

def run_finetuning():
    """Run the configured EoMT OOD fine-tuning job."""
    logger.info("[bold magenta]Starting OOD Fine-Tuning Pipeline[/bold magenta]")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Active device: [bold cyan]{device}[/bold cyan]", extra={"markup": True})
    
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
        logger.info(f"[resume_from] Resuming existing run: {ckpt_dir}")
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(ckpt_base_dir, run_timestamp)
        logger.info(f"New run checkpoint dir: {ckpt_dir}")

    os.makedirs(ckpt_dir, exist_ok=True)
    
    base_ckpt_path = getattr(cfg.paths.models, "eomt_base_checkpoint", cfg.paths.models.eomt_checkpoint)
    
    model = build_model_eomt(base_ckpt_path, device, num_classes=19, img_size=(1024, 1024))
    
    # Freeze DINOv2 features to reduce catastrophic forgetting.
    for param in model.network.encoder.parameters():
        param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    loss_fn = CombinedFineTuningLoss(
        ood_loss_type=cfg.solutions.training.ood_loss_type, 
        ignore_index=cfg.eval.ignore_index,
        ood_entropy_weight=getattr(cfg.solutions.training, "ood_entropy_weight", 1.0),
        ood_logit_norm_weight=getattr(cfg.solutions.training, "ood_logit_norm_weight", 0.05),
    ).to(device)
    
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
        """Convert a PIL label image to a long tensor."""

        def __call__(self, pic):
            return torch.from_numpy(np.array(pic)).long()

    class LocalCityscapesDataset:
        """Minimal Cityscapes trainId dataset used by the fine-tuning loop."""

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
                logger.error(f"Fatal error: no images found with pattern:\n{search_pattern}")
                raise RuntimeError(f"No Cityscapes images found: {search_pattern}")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            gt_path = self.labels[idx]
            
            try:
                img = Image.open(img_path).convert('RGB')
                label_raw_np = np.array(Image.open(gt_path))
            except OSError as e:
                logger.warning(
                    f"Unreadable file at '{img_path}' or '{gt_path}'. "
                    f"Skipping to the next index. Detail: {e}"
                )
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

    cityscapes_root = cfg.paths.datasets.cityscapes
    
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
        ignore_index=cfg.eval.ignore_index
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.solutions.training.batch_size,
        shuffle=True,
        num_workers=cfg.solutions.training.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.solutions.training.num_workers > 0,
    )
    
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
            logger.info(f"[resume_from] Loading checkpoint: {latest_ckpt} (next epoch: {start_epoch})")
            resume_sd = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(resume_sd, strict=True)
            logger.success(f"Checkpoint loaded; resuming from epoch {start_epoch}")
        else:
            logger.warning(f"[resume_from] No checkpoint found in {ckpt_dir}; training from scratch.")
    else:
        logger.info("Training from scratch.")

    epochs = cfg.solutions.training.epochs
    for epoch in range(start_epoch, epochs + 1):
        logger.info(f"\n[bold green]=== Start Epoch {epoch}/{epochs} ===[/bold green]")
        
        avg_ce, avg_ood, avg_ood_ratio, ood_batch_rate = train_epoch(model, dataloader, optimizer, loss_fn, device, epoch)
        
        logger.success(
            f"End Epoch {epoch} | Avg CE Loss: {avg_ce:.4f} | Avg OOD Loss: {avg_ood:.4f} | "
            f"Avg OOD px: {avg_ood_ratio * 100:.3f}% | OOD batches: {ood_batch_rate * 100:.1f}%"
        )
        
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}_EoMT.pth")
        try:
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")

if __name__ == "__main__":
    run_finetuning()
