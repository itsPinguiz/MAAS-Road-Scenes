import subprocess
import os
import sys
import time
from rich.panel import Panel
from logger import logger, console

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
    
    # --- PATH SETUP ---
    # We get the absolute path to the root directory first
    root_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Define the absolute paths to your venv python executables
    eval_dir = os.path.join(root_dir, "eval")
    eval_venv_python = os.path.join(eval_dir, ".venv_eval", "bin", "python")
    
    eomt_dir = os.path.join(root_dir, "eomt")
    eomt_venv_python = os.path.join(eomt_dir, ".venv_eomt", "bin", "python")

    # --- STEP 1: EVAL (ERFNET) ---
    logger.info("\n[bold]Step 1: Running ERFNet Evaluations...[/bold]")
    run_script(eval_venv_python, "eval_iou.py", eval_dir)
    run_script(eval_venv_python, "run_all_evals.py", eval_dir)

    # --- STEP 2: EOMT ---
    logger.info("\n[bold]Step 2: Running EoMT Evaluations...[/bold]")
    run_script(eomt_venv_python, "eval_miou_eomt.py", eomt_dir)
    run_script(eomt_venv_python, "run_all_evals_eomt.py", eomt_dir)

    # --- STEP 3: TEMPERATURE (ERFNET) ---
    logger.info("\n[bold]Step 3: Calculating Temperature metrics (ERFNet)...[/bold]")
    erfnet_logits = os.path.join(eval_dir, "saved_logits", "erfnet")
    run_script(
        eval_venv_python, 
        "eval_temperature.py", 
        root_dir, 
        ["--model", "ERFNET", "--logits_dir", erfnet_logits]
    )

    # --- STEP 4: TEMPERATURE (EOMT) ---
    logger.info("\n[bold]Step 4: Calculating Temperature metrics (EoMT)...[/bold]")
    eomt_logits = os.path.join(eomt_dir, "saved_logits", "eomt")
    run_script(
        eomt_venv_python, 
        "eval_temperature.py", 
        root_dir, 
        ["--model", "EOMT", "--logits_dir", eomt_logits]
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