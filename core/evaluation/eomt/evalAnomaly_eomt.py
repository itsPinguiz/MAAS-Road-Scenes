import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TP_EOMT = os.path.join(_ROOT, "third_party", "eomt")
if _TP_EOMT not in sys.path:
    sys.path.append(_TP_EOMT)

from core.utility.config_loader import cfg
import torch
from core.utility.runtime import get_device

DEVICE = get_device()

import glob
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
from core.utility.eval_common import anomaly_scores_numpy
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from models.vit import ViT
from models.eomt import EoMT
from training.mask_classification_semantic import MaskClassificationSemantic

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def load_eomt_model(ckpt_path):
    """Build the EoMT architecture and load a Lightning or plain state dict."""
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


def get_dense_logits(model, img_tensor, crop_batch_size=2):
    """Run windowed EoMT inference and stitch dense logits back together."""
    crops, origins = model.window_imgs_semantic(img_tensor)
    
    final_mask_logits_list = []
    final_class_logits_list = []
    
    if not torch.is_tensor(crops):
        crops = torch.stack(crops)
        
    num_crops = crops.shape[0]
    
    is_cuda = img_tensor.device.type == "cuda"
    with torch.autocast(device_type=img_tensor.device.type, dtype=torch.float16, enabled=is_cuda):
        
        for i in range(0, num_crops, crop_batch_size):
            crop_batch = crops[i:i+crop_batch_size]
            
            mask_logits_per_layer, class_logits_per_layer = model(crop_batch)
            
            final_mask_logits_list.append(mask_logits_per_layer[-1].float())
            final_class_logits_list.append(class_logits_per_layer[-1].float())
            
            del mask_logits_per_layer, class_logits_per_layer
            if is_cuda:
                torch.cuda.empty_cache()
            
    final_mask_logits = torch.cat(final_mask_logits_list, dim=0)
    final_class_logits = torch.cat(final_class_logits_list, dim=0)
    
    final_mask_logits = F.interpolate(final_mask_logits, model.img_size, mode="bilinear")
    
    crop_logits = model.to_per_pixel_logits_semantic(
        final_mask_logits, final_class_logits
    )
    
    img_sizes = [img.shape[-2:] for img in img_tensor]
    dense_logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
    
    return dense_logits[0].unsqueeze(0), final_mask_logits, final_class_logits

from torch.utils.data import Dataset, DataLoader

class AnomalyDataset(Dataset):
    """Lazily load images as EoMT-ready tensors and keep source metadata."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img_pil = Image.open(path).convert('RGB')
        w, h = img_pil.size
        img_np_temp = np.array(img_pil)
        # Keep inputs in [0, 255]; LightningModule.forward scales by 1/255.
        img_tensor = torch.from_numpy(img_np_temp).permute(2, 0, 1).float()
        return img_tensor, path, w, h

def main():
    """Evaluate EoMT anomaly metrics for one dataset glob."""
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

            if osp.exists(save_path):
                dense_logits = torch.load(save_path, map_location=device)
            else:
                with torch.no_grad():
                    dense_logits, _, _ = get_dense_logits(model, img_tensor, crop_batch_size=args.crop_batch_size)
                    
                    if args.save_logits:
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(dense_logits.cpu(), save_path)
    
            score_maps = anomaly_scores_numpy(dense_logits, include_rba=True)
    
            pathGT = path.replace("images", "labels_masks")                
            if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")                
            if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")  
    
            try:
                mask_img = Image.open(pathGT)
                mask_img = mask_img.resize((w, h), Image.NEAREST)
                ood_gts = np.array(mask_img)
            except Exception as e:
                logger.error(f"Could not load mask for {pathGT}: {e}")
                progress.advance(task_id)
                continue
    
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
                for method, score in score_maps.items():
                    anomaly_scores_all[method].append(score)
                
            del dense_logits, score_maps, ood_gts, mask_img
            if img_tensor is not None: del img_tensor
            torch.cuda.empty_cache()
            progress.advance(task_id)
        
    if len(ood_gts_list) == 0:
        logger.warning("No valid evaluations found.")
        return
    
    ood_gts = np.concatenate([gt.flatten() for gt in ood_gts_list])
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    if not args.quiet:
        logger.info("\n=======================================")
        logger.info(f"Model:   EoMT")
        logger.info(f"Dataset: {args.dataset_name}")
        logger.info("---------------------------------------")
    
    for method in ['msp', 'maxlogit', 'maxentropy', 'rba']:
        anomaly_scores = np.concatenate([score.flatten() for score in anomaly_scores_all[method]])

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
            logger.info(
                f"[Metrics] Model: EoMT, Dataset: {args.dataset_name}, Method: {method_label}, "
                f"AuPRC: {prc_auc*100.0:.2f}, FPR95: {fpr*100.0:.2f}"
            )
        else:
            logger.info(f"Method:  {method_label}")
            logger.info(f"AuPRC:   {prc_auc*100.0:.2f}")
            logger.info(f"FPR95:   {fpr*100.0:.2f}")
            logger.info("---------------------------------------")
        
        from core.utility.update_table import update_table_entry
        update_table_entry(model="EoMT", method=method_label, dataset=args.dataset_name, miou='-', auprc=f"{prc_auc*100.0:.2f}", fpr95=f"{fpr*100.0:.2f}")
        
    if not args.quiet:
        logger.info("=======================================\n")

if __name__ == '__main__':
    main()
