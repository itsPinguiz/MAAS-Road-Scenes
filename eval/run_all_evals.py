import os
import subprocess
import re
import sys
import argparse

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

print("Starting bulk evaluation...")
print(f"Total combinations to test: {len(datasets) * len(methods)}\n")

for dataset_name, dataset_path in datasets.items():
    for method in methods:
        print(f"Running evaluation for Dataset: {dataset_name}, Method: {method.upper()}...")
        cmd = [
            sys.executable, "evalAnomaly.py",
            "--input", dataset_path,
            "--method", method
        ]
        
        # We print stderr as it runs to show progress (evalAnomaly.py doesn't print much, but just in case)
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running eval on {dataset_name} with {method}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
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
        print(f"-> AUPRC: {auprc}, FPR95: {fpr}\n")
        
        # Update the main TABLE.md directly
        if auprc != "N/A" and fpr != "N/A":
            update_table_entry(model=model, method=method, dataset=dataset_name, miou='-', auprc=auprc, fpr95=fpr)

# Format output as Markdown Table based on user rules
markdown_table = "| Model | Method | Dataset | mIoU | AuPRC | FPR95 |\n"
markdown_table += "|---|---|---|---|---|---|\n"
for res in results:
    markdown_table += f"| {res['Model']} | {res['Method']} | {res['Dataset']} | {res['mIoU']} | {res['AuPRC']} | {res['FPR95']} |\n"

print("\n--- Final Evaluation Results ---\n")
print(markdown_table)

with open('full_results.txt', 'w') as f:
    f.write(markdown_table)

print("\nResults successfully saved to full_results.txt")
