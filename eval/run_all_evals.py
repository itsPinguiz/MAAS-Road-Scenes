# --- ENVIRONMENT SETUP BLOCK ---
import os
import sys

# 1. Conditional Environment Detection
IS_COLAB = 'google.colab' in sys.modules

# 2. Hybrid Pathing
if IS_COLAB:
    BASE_PATH = '/content/drive/MyDrive/Project'
    # Optional: Automatically install dependencies if on Colab
    import subprocess
    print("Checking requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', os.path.join(BASE_PATH, 'requirements.txt')])
    except Exception as e:
        print(f"Warning: Could not install requirements: {e}")
else:
    BASE_PATH = '.'

def resolve_path(relative_path):
    """ Helper to resolve paths consistently between environments. """
    if IS_COLAB and relative_path.startswith('../'):
        relative_path = relative_path.lstrip('../')
    return os.path.join(BASE_PATH, relative_path)

# 3. Unified Device Logic
import torch
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# 4. GPU Health Check
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from update_table import update_table_entry
from logger import logger, console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

datasets = {
    'Fishyscapes Static': resolve_path('../Datasets/Fishyscapes/fs_static/images/*.jpg'),
    'Fishyscapes Lost & Found': resolve_path('../Datasets/Fishyscapes/FS_LostFound_full/images/*.png'),
    'RoadAnomaly': resolve_path('../Datasets/RoadAnomaly/images/*.jpg'),
    'RoadAnomaly21': resolve_path('../Datasets/SegmentMeIfYouCan/RoadAnomaly21/images/*.png'),
    'RoadObsticle21': resolve_path('../Datasets/SegmentMeIfYouCan/RoadObsticle21/images/*.webp')
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