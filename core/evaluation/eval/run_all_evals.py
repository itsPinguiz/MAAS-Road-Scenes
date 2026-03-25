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

from core.utility.update_table import update_table_entry
from core.utility.logger import logger, console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

datasets = {
    'Fishyscapes Static':      cfg.paths.datasets.fishyscapes_static,
    'Fishyscapes Lost & Found': cfg.paths.datasets.fishyscapes_lost_found,
    'RoadAnomaly':             cfg.paths.datasets.road_anomaly,
    'RoadAnomaly21':           cfg.paths.datasets.road_anomaly21,
    'RoadObsticle21':          cfg.paths.datasets.road_obsticle21,
}

methods = ['msp', 'maxlogit', 'maxentropy']
model = 'ERFNet'

logger.info(f"Starting bulk evaluation for {model}...")
logger.info(f"Total datasets to test: {len(datasets)} (computing all {len(methods)} methods simultaneously)")

with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    console=console
) as progress:
    task_id = progress.add_task("Evaluating ERFNet", total=len(datasets))

    for dataset_name, dataset_path in datasets.items():
        progress.update(task_id, description=f"Eval: {dataset_name}")
        
        cmd = [
            sys.executable, "evalAnomaly.py",
            "--input", dataset_path,
            "--quiet",
            "--save_logits",
            "--dataset_name", dataset_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error running eval on {dataset_name}:\n{result.stderr}")
            progress.advance(task_id)
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
                logger.info(f"-> {dataset_name} | {table_method}: AUPRC={auprc_dict[method]}, FPR95={fpr_dict[method]}")

        progress.advance(task_id)

logger.success("Updated TABLE.md with the results for ERFNET.")