"""Evaluate ERFNet mIoU on Cityscapes and optionally cache logits."""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TP_EVAL = os.path.join(_ROOT, "third_party", "eval")
if _TP_EVAL not in sys.path:
    sys.path.append(_TP_EVAL)

from core.utility.config_loader import cfg
import torch
from core.utility.runtime import get_device, print_device_health

DEVICE = get_device()

import os
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
import warnings

warnings.filterwarnings("ignore", ".*'network' is an instance.*")

from core.utility.logger import logger, console
from core.utility.eval_common import cityscapes_label_mapping, load_state_dict_flexible
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from erfnet import ERFNet
from transform import Relabel, ToLabel
from iouEval import iouEval

NUM_CLASSES = 20
IGNORE_INDEX = 19

input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, IGNORE_INDEX),
])

def main(args):
    """Evaluate ERFNet mIoU on Cityscapes and optionally save logits."""
    print_device_health(DEVICE)

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

    model = load_state_dict_flexible(model, torch.load(weightspath, map_location=device))
    model.eval()

    image_paths = []
    for root, dirs, files in os.walk(args.datadir):
        if 'val' in root.split(os.sep): 
            for file in files:
                if file.endswith("leftImg8bit.png"):
                    image_paths.append(os.path.join(root, file))

    if len(image_paths) == 0:
        logger.error(f"No validation images found in {args.datadir}")
        return

    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=IGNORE_INDEX)
    
    mapping_256 = cityscapes_label_mapping(ignore_index=255)

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

            img = Image.open(img_path).convert('RGB')
            
            label_raw_np = np.array(Image.open(gt_path))
            label_mapped_np = mapping_256[label_raw_np]
            gt = Image.fromarray(label_mapped_np)

            img_t = input_transform_cityscapes(img).unsqueeze(0).to(device)
            gt_t = target_transform_cityscapes(gt).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_t)
                
            pred = outputs.max(1)[1].unsqueeze(1).data
            
            gt_t[gt_t >= NUM_CLASSES] = IGNORE_INDEX
            pred[pred >= NUM_CLASSES] = IGNORE_INDEX
            
            iouEvalVal.addBatch(pred, gt_t)

            if args.save_logits:
                ckpt_name = os.path.splitext(os.path.basename(args.loadWeights))[0]
                save_dir = os.path.join("saved_logits", "erfnet", ckpt_name, "Cityscapes")
                os.makedirs(save_dir, exist_ok=True)
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
    for method in ['MSP', 'MaxLogit', 'Max Entropy']:
        update_table_entry(model="ERFNET", method=method, miou=miou_str)
    
    logger.success("Updated TABLE.md with ERFNet mIoU.")

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
