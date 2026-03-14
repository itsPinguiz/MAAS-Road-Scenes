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
        "python", "evalAnomaly_eomt.py",
        "--input", dataset_path,
        "--dataset_name", dataset_name,
        "--device", "cuda:0",
        "--quiet"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        tqdm.write(f"\nError running eval on {dataset_name}:", file=sys.stderr)
        tqdm.write(result.stderr, file=sys.stderr)
        continue
        
    auprc_dict = {m: "N/A" for m in methods}
    fpr_dict = {m: "N/A" for m in methods}
    
    # Parse output for metrics
    current_method = None
    for line in result.stdout.split('\n'):
        # Match "Method:  [NAME]" or "Method: [NAME]"
        method_match = re.search(r"Method:\s*([A-Za-z\s]+)", line)
        if method_match:
            current_method = method_match.group(1).strip().lower().replace(" ", "")
            # No need for explicit remapping if current_method is already in `methods`
            continue
            
        if current_method in methods:
            # Match "AuPRC:   [VALUE]"
            auprc_match = re.search(r"AuPRC:\s+([0-9.]+)", line)
            if auprc_match:
                auprc_dict[current_method] = auprc_match.group(1)
                
            # Match "FPR95:   [VALUE]"
            fpr_match = re.search(r"FPR95:\s+([0-9.]+)", line)
            if fpr_match:
                fpr_dict[current_method] = fpr_match.group(1)

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
        
        table_method = method.upper() if method != 'maxentropy' else 'MAX ENTROPY'
        
        # Update the right side of the progress bar with the latest metrics
        pbar.set_postfix({'DS': dataset_name, 'Method': table_method, 'AUPRC': auprc_dict[method]})
        
        # Update the main TABLE.md directly
        if auprc_dict[method] != "N/A" and fpr_dict[method] != "N/A":
            update_table_entry(model="EoMT", method=table_method, dataset=dataset_name, miou='-', auprc=auprc_dict[method], fpr95=fpr_dict[method])

print("\nUpdated TABLE.md with the results for EoMT.")