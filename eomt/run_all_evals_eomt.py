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

methods = ['msp', 'maxlogit', 'maxentropy', 'rba']
model = 'EoMT'

results = []

print("Starting bulk evaluation for EoMT...")
print(f"Total combinations to test: {len(datasets) * len(methods)}\n")

for dataset_name, dataset_path in datasets.items():
    print(f"Running evaluation for Dataset: {dataset_name}...")
    
    cmd = [
        ".venv_eomt/bin/python", "evalAnomaly_eomt.py",
        "--input", dataset_path,
        "--save_logits",
        "--dataset_name", dataset_name.replace(" ", "_"),
        "--device", "cuda:0"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running eval on {dataset_name}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        continue
        
    auprc_dict = {m: "N/A" for m in methods}
    fpr_dict = {m: "N/A" for m in methods}
    
    # Parse output for metrics
    # Format expected log output: "  Method: MSP -> AUPRC score: 85.00, FPR@TPR95: 12.00"
    for line in result.stdout.split('\n'):
        match = re.search(r"Method:\s*([A-Za-z]+)\s*->\s*AUPRC score:\s*([0-9.]+),\s*FPR@TPR95:\s*([0-9.]+)", line)
        if match:
            method_parsed = match.group(1).lower()
            if method_parsed in methods:
                auprc_dict[method_parsed] = match.group(2)
                fpr_dict[method_parsed] = match.group(3)

        results.append({
            'Model': model,
            'Method': method.upper(),
            'Dataset': dataset_name,
            'mIoU': '-',
            'AuPRC': auprc_dict[method],
            'FPR95': fpr_dict[method]
        })
        print(f"-> Method: {method.upper()} -> AUPRC: {auprc_dict[method]}, FPR95: {fpr_dict[method]}")
        
        # Update TABLE.md directly if parsing succeeded
        if auprc_dict[method] != "N/A" and fpr_dict[method] != "N/A":
            # For logging purposes we use uppercase METHOD, except maxentropy which is 'Max Entropy' in table
            table_method = 'Max Entropy' if method == 'maxentropy' else method.upper()
            table_method = 'RbA' if method == 'rba' else table_method
            update_table_entry(model=model, method=table_method, dataset=dataset_name, miou='-', auprc=auprc_dict[method], fpr95=fpr_dict[method])
            
    print("\n")

# Format output as Markdown Table based on user rules
markdown_table = "| Model | Method | Dataset | mIoU | AuPRC | FPR95 |\n"
markdown_table += "|---|---|---|---|---|---|\n"
for res in results:
    markdown_table += f"| {res['Model']} | {res['Method']} | {res['Dataset']} | {res['mIoU']} | {res['AuPRC']} | {res['FPR95']} |\n"

print("\n--- Final Evaluation Results ---\n")
print(markdown_table)

with open('full_results_eomt.txt', 'w') as f:
    f.write(markdown_table)

print("\nResults successfully saved to full_results_eomt.txt")
