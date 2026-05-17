"""Run the project evaluation pipeline as ordered subprocess stages."""

import os
import sys
import time
import subprocess
from dataclasses import dataclass
from rich.panel import Panel

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EVAL_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.utility.logger import logger, console
from core.utility.config_loader import cfg


@dataclass(frozen=True)
class PipelineStage:
    """One subprocess stage in the evaluation pipeline."""

    name: str
    venv_python: str
    script_name: str
    working_dir: str
    args: list[str]
    enabled: bool = True


def run_script(stage: PipelineStage):
    """Run one pipeline stage and show captured errors in a Rich panel."""
    if not stage.enabled:
        logger.info(f"Skipping {stage.name}")
        return

    venv_python = stage.venv_python
    script_name = stage.script_name
    working_dir = stage.working_dir

    if not os.path.exists(venv_python):
        logger.error(f"Could not find Python in {venv_python}")
        sys.exit(1)

    command = [venv_python, script_name] + stage.args
    command_str = " ".join(command)
    logger.info(f"[bold]{stage.name}[/bold]")
    logger.info(f"({working_dir}) Launching: {command_str}")
    
    start_time = time.perf_counter()
    
    with console.status(f"[bold cyan]Running {stage.name}...", spinner="dots"):
        try:
            result = subprocess.run(
                command, 
                cwd=working_dir,
                capture_output=True,
                text=True,
                check=True
            )
            elapsed = time.perf_counter() - start_time
            logger.success(f"Successfully finished {stage.name} in {elapsed:.2f}s")
            
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                f"Pipeline failed! {stage.name} crashed with exit code "
                f"{e.returncode} after {elapsed:.2f}s."
            )
            
            error_output = e.stderr if e.stderr else e.stdout
            if not error_output:
                error_output = "No output captured."
            
            console.print(Panel(error_output, title=f"Error Log: {stage.name}", border_style="red"))
            sys.exit(e.returncode)


def build_stages(root_dir: str) -> list[PipelineStage]:
    """Declare the full evaluation flow from config values."""
    eval_dir = os.path.join(root_dir, "core", "evaluation", "eval")
    eomt_dir = os.path.join(root_dir, "core", "evaluation", "eomt")
    evaluation_dir = os.path.join(root_dir, "core", "evaluation")
    run_erfnet = getattr(cfg.pipeline, "run_erfnet", True)
    run_eomt = getattr(cfg.pipeline, "run_eomt", True)
    save_logits_args = ["--save_logits"] if getattr(cfg.eval, "save_logits", False) else []

    return [
        PipelineStage(
            name="1. ERFNet Cityscapes mIoU + logits",
            venv_python=cfg.paths.venvs.eval_python,
            script_name="eval_iou.py",
            working_dir=eval_dir,
            args=save_logits_args,
            enabled=run_erfnet,
        ),
        PipelineStage(
            name="2. ERFNet OOD datasets",
            venv_python=cfg.paths.venvs.eval_python,
            script_name="run_all_evals.py",
            working_dir=eval_dir,
            args=[],
            enabled=run_erfnet,
        ),
        PipelineStage(
            name="3. EoMT Cityscapes mIoU + logits",
            venv_python=cfg.paths.venvs.eomt_python,
            script_name="eval_miou_eomt.py",
            working_dir=eomt_dir,
            args=save_logits_args,
            enabled=run_eomt,
        ),
        PipelineStage(
            name="4. EoMT OOD datasets",
            venv_python=cfg.paths.venvs.eomt_python,
            script_name="run_all_evals_eomt.py",
            working_dir=eomt_dir,
            args=[],
            enabled=run_eomt,
        ),
        PipelineStage(
            name="5. ERFNet temperature sweep",
            venv_python=cfg.paths.venvs.eval_python,
            script_name="eval_temperature.py",
            working_dir=evaluation_dir,
            args=["--model", "ERFNET", "--logits_dir", cfg.paths.logits.erfnet],
            enabled=run_erfnet,
        ),
        PipelineStage(
            name="6. EoMT temperature sweep",
            venv_python=cfg.paths.venvs.eomt_python,
            script_name="eval_temperature.py",
            working_dir=evaluation_dir,
            args=["--model", "EOMT", "--logits_dir", cfg.paths.logits.eomt],
            enabled=run_eomt,
        ),
    ]


def main():
    """Run the configured evaluation pipeline stages."""
    total_start = time.perf_counter()
    logger.info("[bold magenta]Starting ML Master Pipeline[/bold magenta]")
    
    for stage in build_stages(cfg.paths.root):
        run_script(stage)
    
    total_elapsed = time.perf_counter() - total_start
    completion_msg = (
        f"All evaluations completed successfully!\n"
        f"Total Pipeline Runtime: [bold cyan]{total_elapsed:.2f} seconds[/bold cyan]"
    )
    console.print()
    console.print(
        Panel(
            completion_msg,
            title="[bold green]Pipeline Complete[/bold green]",
            border_style="green",
            expand=False,
        )
    )

if __name__ == "__main__":
    main()
