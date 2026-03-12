import os
import glob
import subprocess
import re
import sys
import argparse

# Add parent directory to path to import the update table utility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from update_table import update_table_entry

erfnet_python = ".venv_eval/bin/python"
eomt_python = "../eomt/.venv_eomt/bin/python"
cityscapes_val_images = "../gtFine_trainvaltest/val/*/*_gtFine_labelIds.png" # Assuming structure of Cityscapes labels
import argparse

# The simplest way to evaluate both is to write a temporary sub-script that loads the model,
# processes the validation set using standard transforms, and computes mIoU, then call it for each model.
# But ERFNet already has eval_iou.py! Let's modify it or use it.
# Actually, eval_iou.py uses datadir. Let's create two dedicated sub-scripts to be completely safe, 
# or just a single flexible generic script and invoke it twice.

eval_miou_erfnet = """
import torch
import numpy as np
from PIL import Image
from erfnet import ERFNet
from transform import Relabel, ToLabel
from iouEval import iouEval
from torchvision.transforms import Compose, Resize, ToTensor
import glob

# Cityscapes 19 classes
NUM_CLASSES = 19
IGNORE_INDEX = 255

def main():
    model = ERFNet(20) # 20 originally, 19 + 1 ignored
    model = torch.nn.DataParallel(model).cuda()
    
    # Load weights
    weightspath = "../trained_models/erfnet_pretrained.pth"
    state_dict = torch.load(weightspath)
    
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
        else:
            own_state[name].copy_(param)

    model.eval()
    
    val_images = glob.glob("../leftImg8bit_trainvaltest/val/*/*.png")
    
    iouEvalVal = iouEval(20, ignoreIndex=19) # 19 is ignored index in eval_iou mapping usually
    
    transform_in = Compose([Resize(512, Image.BILINEAR), ToTensor()])
    transform_gt = Compose([Resize(512, Image.NEAREST), ToLabel(), Relabel(255, 19)])
    
    print("Evaluating ERFNet on", len(val_images), "images...")
    
    with torch.no_grad():
        for i, img_path in enumerate(val_images):
            gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest').replace('leftImg8bit.png', 'gtFine_labelIds.png')
            
            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path)
            
            img_t = transform_in(img).unsqueeze(0).cuda()
            gt_t = transform_gt(gt).unsqueeze(0).cuda()
            
            out = model(img_t)
            
            iouEvalVal.addBatch(out.max(1)[1].unsqueeze(1).data, gt_t)
            if i % 50 == 0:
                print(f"[{i}/{len(val_images)}]")
                
    iouVal, iou_classes = iouEvalVal.getIoU()
    print(f"ERFNet_mIoU_FINAL: {iouVal.item()*100:.2f}")

if __name__ == '__main__':
    main()
"""

eval_miou_eomt = """
import sys
sys.path.append('../eomt')
import torch
import numpy as np
from PIL import Image
import glob
from evalAnomaly_eomt import load_eomt_model, get_dense_logits
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

NUM_CLASSES = 19
IGNORE_INDEX = 255

# Standard Cityscapes mapping
mapping_20 = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
    14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
}
def map_labels(mask):
    temp = np.copy(mask)
    for k, v in mapping_20.items():
        temp[mask == k] = v
    return temp

def compute_miou(conf_matrix):
    with np.errstate(divide='ignore', invalid='ignore'):
        ious = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))
    valid_ious = ious[~np.isnan(ious)]
    return np.mean(valid_ious) * 100

def main():
    model = load_eomt_model("../trained_models/epoch_106-step_19902_eomt.ckpt")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    val_images = glob.glob("../leftImg8bit_trainvaltest/val/*/*.png")
    
    print("Evaluating EoMT on", len(val_images), "images...")
    
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    with torch.no_grad():
        for i, img_path in enumerate(val_images):
            gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest').replace('leftImg8bit.png', 'gtFine_labelIds.png')
            
            img_np = np.array(Image.open(img_path).convert('RGB'))
            gt_np = np.array(Image.open(gt_path))
            
            gt_mapped = map_labels(gt_np)
            
            # Predict
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            dense_logits, _, _ = get_dense_logits(model, img_tensor)
            
            preds = torch.argmax(dense_logits, dim=1).squeeze(0).cpu().numpy()
            
            # Resize pred if necessary
            # dense_logits reverted are already the original input shape.
            
            valid_mask = (gt_mapped != IGNORE_INDEX)
            
            if np.any(valid_mask):
                conf_matrix += confusion_matrix(
                    gt_mapped[valid_mask].flatten(),
                    preds[valid_mask].flatten(),
                    labels=np.arange(NUM_CLASSES)
                )
            
            if i % 50 == 0:
                print(f"[{i}/{len(val_images)}]")
                
    mIoU = compute_miou(conf_matrix)
    print(f"EoMT_mIoU_FINAL: {mIoU:.2f}")

if __name__ == '__main__':
    main()
"""

def write_and_run():
    with open("eval_miou_erfnet.py", "w") as f:
        f.write(eval_miou_erfnet)
        
    with open("eval_miou_eomt.py", "w") as f:
        f.write(eval_miou_eomt)
        
    print("Running ERFNet Evaluation...")
    erfnet_res = subprocess.run([erfnet_python, "eval_miou_erfnet.py"], capture_output=True, text=True)
    if "ERFNet_mIoU_FINAL" in erfnet_res.stdout:
        erfnet_miou = re.search(r"ERFNet_mIoU_FINAL: ([0-9.]+)", erfnet_res.stdout).group(1)
        print(f"ERFNet mIoU: {erfnet_miou}")
        
        # Update TABLE.md directly for all methods of ERFNET (since mIoU is dataset-wide, not method-dependent)
        for method in ['MSP', 'MaxLogit', 'Max Entropy']:
            update_table_entry(model="ERFNET", method=method, miou=erfnet_miou)
    else:
        print("ERFNet failed:")
        print(erfnet_res.stderr)
        
    print("Running EoMT Evaluation...")
    eomt_res = subprocess.run([eomt_python, "eval_miou_eomt.py"], capture_output=True, text=True)
    if "EoMT_mIoU_FINAL" in eomt_res.stdout:
        eomt_miou = re.search(r"EoMT_mIoU_FINAL: ([0-9.]+)", eomt_res.stdout).group(1)
        print(f"EoMT mIoU: {eomt_miou}")
        
        # Update TABLE.md directly for all methods of EoMT (since mIoU is dataset-wide, not method-dependent)
        for method in ['MSP', 'MaxLogit', 'Max Entropy']:
            update_table_entry(model="EoMT", method=method, miou=eomt_miou)
    else:
        print("EoMT failed:")
        print(eomt_res.stderr)

if __name__ == '__main__':
    write_and_run()
