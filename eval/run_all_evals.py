import os
import subprocess
import re
import sys
import argparse
from tqdm import tqdm

# Add parent directory to path to import the update table utility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from update_table import update_table_entry

# Define datasets and their corresponding image paths
datasets = {
    'Fishyscapes Static': '../Datasets/Fishyscapes/fs_static/images/*.jpg',
    'Fishyscapes Lost & Found': '../Datasets/Fishyscapes/FS_LostFound_full/images/*.png',
    'RoadAnomaly': '../Datasets/RoadAnomaly/images/*.jpg',
    'RoadAnomaly21': '../Datasets/SegmentMeIfYouCan/RoadAnomaly21/images/*.png',
    'RoadObsticle21': '../Datasets/SegmentMeIfYouCan/RoadObsticle21/images/*.webp'
}

methods = ['msp', 'maxlogit', 'maxentropy']
model = 'ERFNet'

results = []

# Pre-calculate all combinations to feed into tqdm
combinations = [(d_name, d_path, m) for d_name, d_path in datasets.items() for m in methods]

print(f"Starting bulk evaluation for {model}...")

# Initialize tqdm progress bar
pbar = tqdm(combinations, desc="Evaluating", unit="eval")

for dataset_name, dataset_path, method in pbar:
    # Update progress bar description to show current dataset and method
    pbar.set_description(f"Eval: {dataset_name} [{method.upper()}]")
    
    cmd = [
        sys.executable, "evalAnomaly.py",
        "--input", dataset_path,
        "--method", method
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        # Use tqdm.write instead of print to avoid breaking the progress bar visual
        tqdm.write(f"\nError running eval on {dataset_name} with {method.upper()}:", file=sys.stderr)
        tqdm.write(result.stderr, file=sys.stderr)
        continue
        
    auprc = "N/A"
    fpr = "N/A"
    
    # Parse output for metrics
    for line in result.stdout.split('\n'):
        if "AUPRC score:" in line:
            match = re.search(r"AUPRC score:\s*([0-9.]+)", line)
            if match:
                auprc = match.group(1)
        if "FPR@TPR95:" in line:
            match = re.search(r"FPR@TPR95:\s*([0-9.]+)", line)
            if match:
                fpr = match.group(1)
                
    results.append({
        'Model': model,
        'Method': method.upper(),
        'Dataset': dataset_name,
        'mIoU': '-',
        'AuPRC': auprc,
        'FPR95': fpr
    })
    
    # Update the right side of the progress bar with the latest metrics
    pbar.set_postfix({'AUPRC': auprc, 'FPR95': fpr})
    
    # Update the main TABLE.md directly
    if auprc != "N/A" and fpr != "N/A":
        update_table_entry(model=model, method=method, dataset=dataset_name, miou='-', auprc=auprc, fpr95=fpr)

print(f"\nUpdated TABLE.md with the results for {model}.")