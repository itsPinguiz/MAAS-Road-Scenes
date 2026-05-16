import os
import sys
import argparse
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import random
from collections import defaultdict
import logging

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.utility.config_loader import cfg

DEFAULT_IMG_DIR = os.path.join(cfg.paths.root, "Datasets", "COCO", "val2017")
DEFAULT_ANN_FILE = os.path.join(
    cfg.paths.root,
    "Datasets",
    "COCO",
    "annotations_trainval2017",
    "annotations",
    "instances_val2017.json",
)
DEFAULT_OUTPUT_DIR = os.path.join(cfg.paths.root, "Datasets", "Outliers_COCO")

# COCO classes that make plausible road-scene anomalies.
VALID_CLASSES = [
    'dog', 'cat', 'cow', 'sheep', 'bear', 'horse',
    'backpack', 'umbrella', 'handbag', 'suitcase',
    'frisbee', 'sports ball', 'skateboard',
    'bottle', 'cup', 'bowl', 'banana', 'apple', 'orange',
    'chair', 'couch', 'potted plant', 'teddy bear',
]

MAX_SAMPLES_PER_CLASS = 150
MIN_ANN_AREA = 300

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    """Parse COCO cutout extraction arguments."""
    parser = argparse.ArgumentParser(description="Extract RGBA OOD cutouts from COCO instances.")
    parser.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    parser.add_argument("--ann-file", default=DEFAULT_ANN_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples-per-class", type=int, default=MAX_SAMPLES_PER_CLASS)
    parser.add_argument("--min-ann-area", type=int, default=MIN_ANN_AREA)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def main(
    img_dir=DEFAULT_IMG_DIR,
    ann_file=DEFAULT_ANN_FILE,
    output_dir=DEFAULT_OUTPUT_DIR,
    max_samples_per_class=MAX_SAMPLES_PER_CLASS,
    min_ann_area=MIN_ANN_AREA,
    seed=None,
):
    """Extract selected COCO object instances as transparent PNG cutouts."""
    if seed is not None:
        random.seed(seed)

    if not os.path.exists(ann_file) or not os.path.exists(img_dir):
        logger.error(f"Image directory or annotation file not found.\nIMG: {img_dir}\nANN: {ann_file}")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Initializing PyCocoTools...")
    
    coco = COCO(ann_file)
    
    class_ids = coco.getCatIds(catNms=VALID_CLASSES)
    categories = coco.loadCats(class_ids)
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    annotations_by_class = defaultdict(list)
    
    for cat_id in class_ids:
        ann_ids = coco.getAnnIds(catIds=cat_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann['area'] >= min_ann_area and len(ann.get('segmentation', [])) > 0:
                annotations_by_class[cat_id].append(ann)
    
    total_extracted = 0
    
    for cat_id, name in cat_id_to_name.items():
        anns_for_class = annotations_by_class[cat_id]
        
        if len(anns_for_class) > max_samples_per_class:
            anns_for_class = random.sample(anns_for_class, max_samples_per_class)
            
        logger.info(f"Class [{name}]: processing {len(anns_for_class)} instances...")
        
        for ann in anns_for_class:
            try:
                img_data = coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(img_dir, img_data['file_name'])
                
                if not os.path.exists(img_path):
                    continue
                
                img_rgb = np.array(Image.open(img_path).convert('RGB'))
                mask = coco.annToMask(ann)
                x_bbox, y_bbox, w_bbox, h_bbox = [int(v) for v in ann['bbox']]
                
                if w_bbox <= 0 or h_bbox <= 0:
                    continue
                    
                x_min = max(0, x_bbox)
                y_min = max(0, y_bbox)
                x_max = min(img_rgb.shape[1], x_bbox + w_bbox)
                y_max = min(img_rgb.shape[0], y_bbox + h_bbox)
                
                img_crop = img_rgb[y_min:y_max, x_min:x_max]
                mask_crop = mask[y_min:y_max, x_min:x_max]
                
                rgba_crop = np.zeros((img_crop.shape[0], img_crop.shape[1], 4), dtype=np.uint8)
                
                fg_indices = (mask_crop == 1)
                rgba_crop[fg_indices, 0:3] = img_crop[fg_indices]
                rgba_crop[fg_indices, 3] = 255
                
                result_img = Image.fromarray(rgba_crop, 'RGBA')
                
                save_path = os.path.join(output_dir, f"{name}_{ann['id']}.png")
                result_img.save(save_path, "PNG")
                total_extracted += 1

            except Exception as e:
                logger.debug(f"Annotation {ann['id']} failed: {e}")
                continue

    logger.info(f"OOD extraction complete. Generated {total_extracted} transparent cutouts.")
    logger.info(f"Results available in: {output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        output_dir=args.output_dir,
        max_samples_per_class=args.max_samples_per_class,
        min_ann_area=args.min_ann_area,
        seed=args.seed,
    )
