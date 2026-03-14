import os
import torch
import numpy as np
import glob
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import sys
from argparse import ArgumentParser

# Import utility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from update_table_t import update_table_t_entry
from ood_metrics import fpr_at_95_tpr

def get_msp_score(logits, T, device):
    scaled_logits = logits / T
    probs = F.softmax(scaled_logits, dim=1)
    msp, _ = torch.max(probs, dim=1)
    return (1.0 - msp).squeeze().data.cpu().numpy()

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', required=True, choices=['ERFNET', 'EOMT'])
    parser.add_argument('--logits_dir', required=True)
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    TEMPS = [0.5, 0.75, 1.1]
    DATASETS = {
        'RoadAnomaly21': '../Datasets/SegmentMeIfYouCan/RoadAnomaly21/images/*.png',
        'RoadObsticle21': '../Datasets/SegmentMeIfYouCan/RoadObsticle21/images/*.webp',
        'Fishyscapes Lost & Found': '../Datasets/Fishyscapes/FS_LostFound_full/images/*.png',
        'Fishyscapes Static': '../Datasets/Fishyscapes/fs_static/images/*.jpg',
        'RoadAnomaly': '../Datasets/RoadAnomaly/images/*.jpg'
    }

    for ds_name, ds_pattern in DATASETS.items():
        ds_slug = ds_name.replace(" ", "_")
        current_logits_path = os.path.join(args.logits_dir, ds_slug)
        logit_files = glob.glob(os.path.join(current_logits_path, "*.pt"))
        
        if not logit_files:
            print(f"Saltando {ds_name}: Nessun logit trovato in {current_logits_path}")
            continue

        best_metrics = {"auprc": -1, "fpr95": 100, "t": None}

        for T in TEMPS:
            all_scores, all_gts = [], []
            
            for l_path in tqdm(logit_files, desc=f"{ds_name} (T={T})"):
                logits = torch.load(l_path, map_location=device).to(device)
                score = get_msp_score(logits, T, device)
                
                # Mapping preciso della GT
                base_name = os.path.basename(l_path).replace(".pt", "")
                img_ref_list = glob.glob(ds_pattern.replace("*", base_name))
                if not img_ref_list: continue
                
                gt_path = img_ref_list[0].replace("images", "labels_masks")
                if "RoadObsticle21" in gt_path: gt_path = gt_path.replace(".webp", ".png")
                if "fs_static" in gt_path: gt_path = gt_path.replace(".jpg", ".png")
                if "RoadAnomaly" in gt_path: gt_path = gt_path.replace(".jpg", ".png")

                gt_img = Image.open(gt_path).resize((score.shape[1], score.shape[0]), Image.NEAREST)
                gt_np = np.array(gt_img)

                # Mappature OOD
                if "RoadAnomaly" in ds_name: 
                    gt_np = np.where(gt_np==2, 1, gt_np)
                elif "LostAndFound" in ds_name:
                    gt_np = np.where(gt_np==0, 255, gt_np)
                    gt_np = np.where(gt_np==1, 0, gt_np)
                    gt_np = np.where((gt_np>1)&(gt_np<201), 1, gt_np)

                mask = (gt_np != 255)
                if np.any(mask):
                    all_scores.append(score[mask])
                    all_gts.append(gt_np[mask])

            if not all_gts: continue
            
            y_true = np.concatenate(all_gts)
            y_scores = np.concatenate(all_scores)
            
            auprc = average_precision_score(y_true, y_scores) * 100
            fpr95 = fpr_at_95_tpr(y_scores, y_true) * 100
            
            print(f" {ds_name} | T={T} -> AuPRC: {auprc:.2f}, FPR95: {fpr95:.2f}")
            update_table_t_entry(args.model, f"MSP (t = {T})", ds_name, f"{auprc:.2f}", f"{fpr95:.2f}")
            
            if auprc > best_metrics["auprc"]:
                best_metrics.update({"auprc": auprc, "fpr95": fpr95, "t": T})
        
        if best_metrics["t"] is not None:
            update_table_t_entry(args.model, "MSP (best t)", ds_name, 
                                 f"{best_metrics['auprc']:.2f}", f"{best_metrics['fpr95']:.2f}")

if __name__ == "__main__":
    main()