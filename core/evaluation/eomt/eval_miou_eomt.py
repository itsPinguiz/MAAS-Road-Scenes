"""Evaluate EoMT mIoU on Cityscapes and optionally cache dense logits."""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_TP_EOMT = os.path.join(_ROOT, "third_party", "eomt")
if _TP_EOMT not in sys.path:
    sys.path.append(_TP_EOMT)

_TP_EVAL = os.path.join(_ROOT, "third_party", "eval")
if _TP_EVAL not in sys.path:
    sys.path.append(_TP_EVAL)

from core.utility.config_loader import cfg
import torch
from core.utility.runtime import get_device, print_device_health

DEVICE = get_device()

_CORE_EVAL_DIR = os.path.join(_ROOT, "core", "evaluation", "eval")
if _CORE_EVAL_DIR not in sys.path:
    sys.path.append(_CORE_EVAL_DIR)

import warnings
warnings.filterwarnings("ignore", ".*'network' is an instance.*")

from core.utility.logger import logger, console
from core.utility.eval_common import cityscapes_label_mapping
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchmetrics import JaccardIndex

from evalAnomaly_eomt import get_dense_logits

def load_eomt_model(ckpt_path):
    """Build the EoMT architecture and load a Lightning or plain state dict."""
    import torch
    from models.vit import ViT
    from models.eomt import EoMT
    from training.mask_classification_semantic import MaskClassificationSemantic
    
    img_size = (1024, 1024)
    num_classes = 19
    
    encoder = ViT(
        img_size=img_size, 
        patch_size=16, 
        backbone_name="vit_base_patch14_reg4_dinov2",
        ckpt_path=ckpt_path,
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

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error(f"Could not open checkpoint '{ckpt_path}': {e}")
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
        logger.success(f"Weights loaded successfully (strict={strict}): {ckpt_path}")
    except RuntimeError as e:
        logger.warning(f"strict=True failed: {e}\nRetrying with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    return model


# 19 Cityscapes classes plus one ignored class mapped to 19.
NUM_CLASSES = 20
IGNORE_INDEX = 19

def main():
    """Evaluate EoMT mIoU on Cityscapes and optionally save dense logits."""
    print_device_health(DEVICE)

    parser = ArgumentParser()
    parser.add_argument('--datadir', default=cfg.paths.datasets.cityscapes)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--ckpt_path', default=cfg.paths.models.eomt_checkpoint)
    parser.add_argument('--device', default=str(DEVICE), help='Device to use for computation')
    parser.add_argument('--quiet', action='store_true', help='Minimal output for bulk runs')
    parser.add_argument('--no-update-table', action='store_true', help='Do not write mIoU into TABLE.md')
    parser.add_argument('--save_logits', action='store_true', help='Save logits to disk')
    parser.add_argument('--crop-batch-size', type=int, default=2, help='Batch size for crop window inference')
    args = parser.parse_args()

    model = load_eomt_model(args.ckpt_path)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    import numpy as np

    image_paths = []
    for root, dirs, files in os.walk(args.datadir):
        if 'val' in root.split(os.sep): 
            for file in files:
                if file.endswith("leftImg8bit.png"):
                    image_paths.append(os.path.join(root, file))

    if len(image_paths) == 0:
        logger.error(f"No validation images found in {args.datadir}")
        return

    metric = JaccardIndex(task="multiclass", num_classes=19, ignore_index=255).to(device)

    valid_images_count = 0

    mapping_256 = cityscapes_label_mapping(ignore_index=255)

    valid_paths = []
    for img_path in image_paths:
        gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest') \
                          .replace('leftImg8bit', 'gtFine') \
                          .replace('.png', '_labelIds.png')
        if os.path.exists(gt_path):
            valid_paths.append((img_path, gt_path))
        else:
            logger.warning(f"Missing label for {img_path}\nExpected: {gt_path}")

    class CityscapesValDataset(Dataset):
        """Load Cityscapes validation images and mapped trainId labels."""

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
            
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            label_tensor = torch.from_numpy(label_mapped_np)
            return img_tensor, label_tensor, img_path

    dataset = CityscapesValDataset(valid_paths, mapping_256)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

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
                    for sample_logits, img_path in zip(dense_logits.cpu(), img_path_batch):
                        save_name = os.path.basename(img_path).replace(".png", ".pt")
                        torch.save(sample_logits.unsqueeze(0), os.path.join(save_dir, save_name))
                
            del dense_logits, preds, img_tensor, label_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()
            progress.advance(task_id)

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

    if not args.no_update_table:
        from core.utility.update_table import update_table_entry
        
        miou_str = f"{mIoU * 100:.2f}"
        for method in ['MSP', 'MaxLogit', 'Max Entropy', 'RbA']:
            update_table_entry(model="EoMT", method=method, miou=miou_str)
        
        logger.success("Updated TABLE.md with EoMT mIoU.")

if __name__ == '__main__':
    main()
