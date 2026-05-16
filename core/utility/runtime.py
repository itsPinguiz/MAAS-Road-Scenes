"""Small runtime helpers shared by command-line scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def add_paths(*paths: str | Path, append: bool = False) -> None:
    """Add existing import roots to ``sys.path`` once."""
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            if append:
                sys.path.append(path_str)
            else:
                sys.path.insert(0, path_str)


def get_device() -> torch.device:
    """Return the best available compute device for local inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_health(device: torch.device) -> None:
    """Print a compact hardware summary when Rich is installed."""
    try:
        from rich.console import Console
        from rich.panel import Panel
    except ImportError:
        return

    details = f"[bold]Hardware environment:[/bold] {device.type.upper()}\n"
    if device.type == "cuda":
        details += f"CUDA Device: {torch.cuda.get_device_name(device)}\n"
        vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        details += f"Available VRAM: {vram:.2f} GB"
    elif device.type == "mps":
        details += "Apple Silicon (MPS) detected."
    else:
        details += "[yellow]Running on CPU. Performance will be limited.[/yellow]"

    Console().print(
        Panel(details, title="[bold blue]GPU Health Check[/bold blue]", border_style="blue", expand=False)
    )
