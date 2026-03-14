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

methods = ['msp', 'maxlogit', 'maxentropy', 'rba']
model = 'EoMT'

results = []

print(f"Starting bulk evaluation for {model}...")
print(f"Total datasets to test: {len(datasets)} (computing all {len(methods)} methods simultaneously)\n")

# Initialize tqdm progress bar over the datasets
pbar = tqdm(datasets.items(), desc="Evaluating EoMT", unit="dataset")

for dataset_name, dataset_path in pbar:
    # Update progress bar description
    pbar.set_description(f"Eval: {dataset_name}")
    
    cmd = [
        sys.executable, "evalAnomaly_eomt.py",
        "--input", dataset_path,
        "--save_logits",
        "--dataset_name", dataset_name.replace(" ", "_"),
        "--device", "cuda:0"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        tqdm.write(f"\nError running eval on {dataset_name}:", file=sys.stderr)
        tqdm.write(result.stderr, file=sys.stderr)
        continue
        
    auprc_dict = {m: "N/A" for m in methods}
    fpr_dict = {m: "N/A" for m in methods}
    
    # Parse output for metrics
    # Expected format: "Method: MSP -> AUPRC score: 85.00, FPR@TPR95: 12.00"
    for line in result.stdout.split('\n'):
        match = re.search(r"Method:\s*([A-Za-z]+)\s*->\s*AUPRC score:\s*([0-9.]+),\s*FPR@TPR95:\s*([0-9.]+)", line)
        if match:
            method_parsed = match.group(1).lower()
            if method_parsed in methods:
                auprc_dict[method_parsed] = match.group(2)
                fpr_dict[method_parsed] = match.group(3)

    # After parsing the output for this dataset, iterate through methods to update the table
    for method in methods:
        results.append({
            'Model': model,
            'Method': method.upper(),
            'Dataset': dataset_name,
            'mIoU': '-',
            'AuPRC': auprc_dict[method],
            'FPR95': fpr_dict[method]
        })
        
        # Update TABLE.md directly if parsing succeeded
        if auprc_dict[method] != "N/A" and fpr_dict[method] != "N/A":
            table_method = 'Max Entropy' if method == 'maxentropy' else method.upper()
            table_method = 'RbA' if method == 'rba' else table_method
            
            update_table_entry(model=model, method=table_method, dataset=dataset_name, miou='-', auprc=auprc_dict[method], fpr95=fpr_dict[method])
            
            # Print cleanly above the progress bar
            tqdm.write(f"-> {dataset_name} | {table_method}: AUPRC={auprc_dict[method]}, FPR95={fpr_dict[method]}")

print("\nUpdated TABLE.md with the results for EoMT.")