import os
import glob
import numpy as np
from PIL import Image

mapping_256 = np.ones(256, dtype=np.uint8) * 19
mapping = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}
for k, v in mapping.items():
    mapping_256[k] = v

root_dir = "/mnt/Shared-Data/Code/github/itsPinguiz/MAAS-Road-Scenes/Datasets/Cityscapes"
search_pattern = os.path.join(root_dir, "**", "train", "**", "*leftImg8bit.png")
images = sorted(glob.glob(search_pattern, recursive=True))

for i, img_path in enumerate(images):
    gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest') \
                      .replace('leftImg8bit', 'gtFine') \
                      .replace('.png', '_labelIds.png')
    try:
        label_raw_np = np.array(Image.open(gt_path))
        label_mapped_np = mapping_256[label_raw_np]
        min_val = label_mapped_np.min()
        max_val = label_mapped_np.max()
        if min_val < 0 or max_val > 19:
            print(f"Bad label found in {gt_path}: min={min_val}, max={max_val}")
    except Exception as e:
        print(f"Error reading {gt_path}: {e}")
print("Check complete.")
