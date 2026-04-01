"""
run_full_analysis_pipeline.py
=============================

Master orchestration script to automate the execution of all 3 fine-grained analysis pillars:
  1. eval_semantics.py
  2. eval_objects.py
  3. eval_resolution_attention.py
  
over multiple models and datasets. Collects plots into a structured output directory.

USAGE:
  python core/analysis/run_full_analysis_pipeline.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich import box

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.utility.config_loader import cfg
from core.utility.logger import logger, console

# --- Configuration ---
MODELS = {
    "ERFNet": {
        "logits_base": cfg.paths.logits.erfnet
    },
    "EoMT": {
        "logits_base": cfg.paths.logits.eomt
    }
}

DATASETS = [
    {
        "name": "Fishyscapes_Static",
        "type": "fs_static",
        "images_glob": cfg.paths.datasets.fishyscapes_static,
        "logits_folder": "Fishyscapes_Static"
    },
    {
        "name": "Fishyscapes_Lost_Found",
        "type": "lost_found",
        "images_glob": cfg.paths.datasets.fishyscapes_lost_found,
        "logits_folder": "Fishyscapes_Lost_&_Found"
    },
    {
        "name": "RoadAnomaly",
        "type": "road_anomaly",
        "images_glob": cfg.paths.datasets.road_anomaly,
        "logits_folder": "RoadAnomaly"
    },
    {
        "name": "RoadAnomaly21",
        "type": "road_anomaly21",
        "images_glob": cfg.paths.datasets.road_anomaly21,
        "logits_folder": "RoadAnomaly21"
    },
    {
        "name": "RoadObsticle21",
        "type": "road_obstacle21", 
        "images_glob": cfg.paths.datasets.road_obstacle21,
        "logits_folder": "RoadObsticle21"
    }
]


def run_command(cmd, task_name, env=None):
    """Execute a shell command with a rich spinner and elapsed time logging."""
    start_time = time.perf_counter()
    with console.status(f"[bold cyan]Running {task_name}...", spinner="dots"):
        try:
            res = subprocess.run(
                cmd, 
                check=True, 
                env=env, 
                capture_output=True, 
                text=True
            )
            elapsed = time.perf_counter() - start_time
            logger.success(f"Successfully finished {task_name} in {elapsed:.2f}s")
            return True
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Task '{task_name}' failed after {elapsed:.2f}s with exit code {e.returncode}")
            
            error_output = e.stderr if e.stderr else e.stdout
            if not error_output:
                error_output = "No output captured."
            
            console.print(Panel(error_output, title=f"Error Log: {task_name}", border_style="red"))
            return False
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"[bold red]Task '{task_name}' failed with exception after {elapsed:.2f}s: {e}[/]")
            return False


def main():
    root_dir = Path(_ROOT)
    
    # Fallback sicuro nel caso in cui cfg.analysis.reports_dir non sia ancora nel config.yml
    reports_dir_path = getattr(getattr(cfg, "analysis", None), "reports_dir", "results/analysis_reports")
    reports_dir = root_dir / reports_dir_path
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    summary_lines = ["# Fine-Grained Analysis Summary\n"]
    summary_lines.append("Execution report across all datasets and models.\n")
    summary_lines.append("| Model | Dataset | Task 1: Semantics | Task 2: Objects | Task 3: Res/Attn |")
    summary_lines.append("|-------|---------|-------------------|-----------------|------------------|")
    
    script_semantics = root_dir / "core" / "analysis" / "eval_semantics.py"
    script_objects = root_dir / "core" / "analysis" / "eval_objects.py"
    script_res_attn = root_dir / "core" / "analysis" / "eval_resolution_attention.py"

    sys_env = os.environ.copy()
    
    # FORZATURA: Usa SEMPRE l'interprete Python corrente (quello che ha cv2 e seaborn)
    python_exec = sys.executable

    for model_name, model_info in MODELS.items():
            
        for d in DATASETS:
            dataset_name = d["name"]
            dataset_type = d["type"]
            images_glob = d["images_glob"] 
            
            logits_dir = root_dir / model_info["logits_base"] / d["logits_folder"]

            out_dir = reports_dir / model_name / dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\n[bold magenta]Evaluating {model_name} on {dataset_name}[/bold magenta]")
            
            # Base arguments for Task 1 and 2
            base_args = [
                str(python_exec),
                "", # placeholder for script path
                "--logits_dir", str(logits_dir),
                "--images_glob", str(d["images_glob"]),
                "--dataset_type", dataset_type,
                "--model_name", model_name,
                "--out_dir", str(out_dir)
            ]
            
            # --- TASK 1 ---
            cmd1 = list(base_args)
            cmd1[1] = str(script_semantics)
            ok1 = run_command(cmd1, f"Semantics ({model_name} / {dataset_name})", sys_env)
            
            # --- TASK 2 ---
            cmd2 = list(base_args)
            cmd2[1] = str(script_objects)
            ok2 = run_command(cmd2, f"Objects ({model_name} / {dataset_name})", sys_env)
            
            # --- TASK 3 ---
            cmd3 = [
                str(python_exec),
                str(script_res_attn),
                "--images_glob", str(d["images_glob"]),
                "--dataset_type", dataset_type,
                "--out_dir", str(out_dir)
            ]
            
            # Specific execution rule for Task 3 based on the model:
            if model_name == "ERFNet":
                cmd3.append("--run_resolution_eval")
            elif model_name == "EoMT":
                cmd3.append("--run_attention_eval")

            ok3 = run_command(cmd3, f"Res/Attn ({model_name} / {dataset_name})", sys_env)
            
            res1 = "✅" if ok1 else "❌"
            res2 = "✅" if ok2 else "❌"
            res3 = "✅" if ok3 else "❌"
            
            summary_lines.append(f"| {model_name} | {dataset_name} | {res1} | {res2} | {res3} |")
            
    # Write Final Summary Markdown
    summary_path = reports_dir / "SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        f.write("\n\n*Plots and extracted results are available in the corresponding model and dataset subdirectories.*")
    
    logger.info(f"\n[bold green]Full analysis pipeline completed! Summary saved to {summary_path}[/]")

if __name__ == "__main__":
    main()