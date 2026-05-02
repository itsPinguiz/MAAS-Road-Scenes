import os
import sys
import time
import subprocess
from rich.panel import Panel

# Add project root to path so core.* is importable
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EVAL_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.utility.logger import logger, console
from core.utility.config_loader import cfg

def run_script(venv_python, script_name, working_dir, args_list=None):
    """Runs a python script using a specific virtual environment's python executable."""
    if args_list is None:
        args_list = []
        
    if not os.path.exists(venv_python):
        logger.error(f"Could not find Python in {venv_python}")
        sys.exit(1)

    command = [venv_python, script_name] + args_list
    command_str = " ".join(command)
    logger.info(f"({working_dir}) Launching: {command_str}")
    
    start_time = time.perf_counter()
    
    # Use rich.status to animate the execution instead of dumping logs
    with console.status(f"[bold cyan]Evaluating {script_name}...", spinner="dots"):
        try:
            result = subprocess.run(
                command, 
                cwd=working_dir,
                capture_output=True, # Prevent raw prints from cluttering screen
                text=True,
                check=True
            )
            elapsed = time.perf_counter() - start_time
            logger.success(f"Successfully finished {script_name} in {elapsed:.2f}s")
            
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Pipeline failed! {script_name} crashed with exit code {e.returncode} after {elapsed:.2f}s.")
            
            # Print the captured error log inside a red rich panel
            error_output = e.stderr if e.stderr else e.stdout
            if not error_output:
                error_output = "No output captured."
            
            console.print(Panel(error_output, title=f"Error Log: {script_name}", border_style="red"))
            sys.exit(e.returncode)

def main():
    total_start = time.perf_counter()
    logger.info("[bold magenta]Starting ML Master Pipeline[/bold magenta]")
    
    # --- PATH SETUP (from config.yml) ---
    root_dir   = cfg.paths.root
    eval_dir   = os.path.join(root_dir, "core", "evaluation", "eval")
    eomt_dir   = os.path.join(root_dir, "core", "evaluation", "eomt")

    eval_venv_python = cfg.paths.venvs.eval_python
    eomt_venv_python = cfg.paths.venvs.eomt_python

    # --- STEP 1: EVAL (ERFNET) ---
    if getattr(cfg.pipeline, "run_erfnet", True):
        logger.info("\n[bold]Step 1: Running ERFNet Evaluations...[/bold]")
        run_script(eval_venv_python, "eval_iou.py", eval_dir, ["--save_logits"])
        run_script(eval_venv_python, "run_all_evals.py", eval_dir)

    # --- STEP 2: EOMT ---
    if getattr(cfg.pipeline, "run_eomt", True):
        logger.info("\n[bold]Step 2: Running EoMT Evaluations...[/bold]")
        run_script(eomt_venv_python, "eval_miou_eomt.py", eomt_dir, ["--save_logits"])
        run_script(eomt_venv_python, "run_all_evals_eomt.py", eomt_dir)

    # --- STEP 3: TEMPERATURE (ERFNET) ---
    if getattr(cfg.pipeline, "run_erfnet", True):
        logger.info("\n[bold]Step 3: Calculating Temperature metrics (ERFNet)...[/bold]")
        run_script(
            eval_venv_python, 
            "eval_temperature.py", 
            os.path.join(root_dir, "core", "evaluation"), 
            ["--model", "ERFNET", "--logits_dir", cfg.paths.logits.erfnet]
        )

    # --- STEP 4: TEMPERATURE (EOMT) ---
    if getattr(cfg.pipeline, "run_eomt", True):
        logger.info("\n[bold]Step 4: Calculating Temperature metrics (EoMT)...[/bold]")
        run_script(
            eomt_venv_python, 
            "eval_temperature.py", 
            os.path.join(root_dir, "core", "evaluation"), 
            ["--model", "EOMT", "--logits_dir", cfg.paths.logits.eomt]
        )
    
    # --- FINAL TIMER ---
    total_elapsed = time.perf_counter() - total_start
    completion_msg = (
        f"All evaluations completed successfully!\n"
        f"Total Pipeline Runtime: [bold cyan]{total_elapsed:.2f} seconds[/bold cyan]"
    )
    console.print()
    console.print(Panel(completion_msg, title="[bold green]Pipeline Complete[/bold green]", border_style="green", expand=False))

if __name__ == "__main__":
    main()