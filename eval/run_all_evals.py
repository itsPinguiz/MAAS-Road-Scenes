import os
import subprocess
import re
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from update_table import update_table_entry

datasets = {
    'Fishyscapes Static': '../Datasets/Fishyscapes/fs_static/images/*.jpg',
    'Fishyscapes Lost & Found': '../Datasets/Fishyscapes/FS_LostFound_full/images/*.png',
    'RoadAnomaly': '../Datasets/RoadAnomaly/images/*.jpg',
    'RoadAnomaly21': '../Datasets/SegmentMeIfYouCan/RoadAnomaly21/images/*.png',
    'RoadObsticle21': '../Datasets/SegmentMeIfYouCan/RoadObsticle21/images/*.webp'
}

methods = ['msp', 'maxlogit', 'maxentropy']
model = 'ERFNet'

print(f"Starting bulk evaluation for {model}...")
print(f"Total datasets to test: {len(datasets)} (computing all {len(methods)} methods simultaneously)\n")

pbar = tqdm(datasets.items(), desc="Evaluating ERFNet", unit="dataset")

for dataset_name, dataset_path in pbar:
    pbar.set_description(f"Eval: {dataset_name}")
    
    cmd = [
        sys.executable, "evalAnomaly.py",
        "--input", dataset_path,
        "--quiet"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        tqdm.write(f"\nError running eval on {dataset_name}:", file=sys.stderr)
        tqdm.write(result.stderr, file=sys.stderr)
        continue
        
    auprc_dict = {m: "N/A" for m in methods}
    fpr_dict = {m: "N/A" for m in methods}
    
    for line in result.stdout.split('\n'):
        match = re.search(r"Method:\s*([A-Za-z]+)\s*->\s*AUPRC score:\s*([0-9.]+),\s*FPR@TPR95:\s*([0-9.]+)", line)
        if match:
            method_parsed = match.group(1).lower()
            if method_parsed in methods:
                auprc_dict[method_parsed] = match.group(2)
                fpr_dict[method_parsed] = match.group(3)

    for method in methods:
        if auprc_dict[method] != "N/A" and fpr_dict[method] != "N/A":
            table_method = 'Max Entropy' if method == 'maxentropy' else method.upper()
            update_table_entry(model="ERFNET", method=table_method, dataset=dataset_name, miou='-', auprc=auprc_dict[method], fpr95=fpr_dict[method])
            tqdm.write(f"-> {dataset_name} | {table_method}: AUPRC={auprc_dict[method]}, FPR95={fpr_dict[method]}")

print("\nUpdated TABLE.md with the results for ERFNET.")