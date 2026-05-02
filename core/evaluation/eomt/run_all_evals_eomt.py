# --- ENVIRONMENT SETUP BLOCK ---
import os
import sys

# Add project root to path so core.* is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.utility.config_loader import cfg

# Unified Device Logic
import torch
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# GPU Health Check
def print_gpu_health(device):
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        details = f"[bold]Hardware environment:[/bold] {device.type.upper()}\n"
        if device.type == 'cuda':
            details += f"CUDA Device: {torch.cuda.get_device_name(device)}\n"
            vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            details += f"Available VRAM: {vram:.2f} GB"
        elif device.type == 'mps':
            details += "Apple Silicon (MPS) detected."
        else:
            details += "[yellow]Running on CPU. Performance will be limited.[/yellow]"
        console.print(Panel(details, title="[bold blue]GPU Health Check[/bold blue]", border_style="blue", expand=False))
    except ImportError:
        pass

print_gpu_health(DEVICE)
# --- END SETUP BLOCK ---

import os
import subprocess
import re
import sys
import argparse
import warnings

warnings.filterwarnings("ignore", ".*'network' is an instance.*")

# Add parent directory to path to import the update table utility
from core.utility.update_table import update_table_entry
from core.utility.logger import logger, console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Define datasets and their corresponding image paths
datasets = {
    'Fishyscapes Static':      cfg.paths.datasets.fishyscapes_static,
    'Fishyscapes Lost & Found': cfg.paths.datasets.fishyscapes_lost_found,
    'RoadAnomaly':             cfg.paths.datasets.road_anomaly,
    'RoadAnomaly21':           cfg.paths.datasets.road_anomaly21,
    'RoadObsticle21':          cfg.paths.datasets.road_obstacle21,
}

methods = ['msp', 'maxlogit', 'maxentropy', 'rba']
model = 'EoMT'

results = []

logger.info(f"Starting bulk evaluation for {model}...")
logger.info(f"Total datasets to test: {len(datasets)} (computing all {len(methods)} methods simultaneously)")

with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    console=console
) as progress:
    task_id = progress.add_task("Evaluating EoMT", total=len(datasets))
    
    for dataset_name, dataset_path in datasets.items():
        progress.update(task_id, description=f"Eval: {dataset_name}")
        
        cmd = [
            sys.executable, "evalAnomaly_eomt.py",
            "--input", dataset_path,
            "--dataset_name", dataset_name,
            "--device", "cuda:0",
            "--quiet",
            "--save_logits",
            "--ckpt_path", os.path.join(_ROOT, cfg.paths.models.eomt_checkpoint)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error running eval on {dataset_name}:\n{result.stderr}")
            progress.advance(task_id)
            continue
            
        auprc_dict = {m: "N/A" for m in methods}
        fpr_dict = {m: "N/A" for m in methods}
        
        # Parse output for metrics
        for line in result.stdout.split('\n'):
            # Match "[Metrics] Model: EoMT, Dataset: [NAME], Method: [NAME], AuPRC: [VALUE], FPR95: [VALUE]"
            match = re.search(r"Method:\s*([^,]+),\s*AuPRC:\s*([0-9.]+),\s*FPR95:\s*([0-9.]+)", line)
            if match:
                method_parsed = match.group(1).strip().lower().replace(" ", "")
                # Account for "maxentropy" vs "max entropy"
                if method_parsed == 'maxentropy':
                    method_parsed = 'maxentropy'
                    
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
            
            if method == 'maxentropy':
                table_method = 'MAX ENTROPY'
            elif method == 'rba':
                table_method = 'Rba'
            else:
                table_method = method.upper()
            
            # Update the right side of the progress bar with the latest metrics inside description
            progress.update(task_id, description=f"Eval: {dataset_name} | {table_method}: AUPRC={auprc_dict[method]}")
            
            # Update the main TABLE.md directly
            if auprc_dict[method] != "N/A" and fpr_dict[method] != "N/A":
                update_table_entry(model="EoMT", method=table_method, dataset=dataset_name, miou='-', auprc=auprc_dict[method], fpr95=fpr_dict[method])

        progress.advance(task_id)

logger.success("Updated TABLE.md with the results for EoMT.")