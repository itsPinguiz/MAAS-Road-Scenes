"""
config_loader.py
----------------
Loads config.yml from the project root and exposes a fully-resolved `cfg`
object. All relative paths are converted to absolute paths at import time,
so every script gets consistent references regardless of its cwd.

Usage:
    from config_loader import cfg
    print(cfg.paths.datasets.cityscapes)
    print(cfg.eval.temperatures)
"""

import os
import sys
import yaml
from types import SimpleNamespace

# Always points to the project root.
_UTILITY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_UTILITY_DIR))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yml")


def _load() -> SimpleNamespace:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Use the configured Drive root when running inside Colab.
    is_colab = "google.colab" in sys.modules
    root = raw["paths"]["colab_root"] if is_colab else _PROJECT_ROOT

    def abs_path(p: str) -> str:
        """Return an absolute path, resolving relative paths against root."""
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))

    p = raw["paths"]

    ns_paths = SimpleNamespace(
        root=root,
        datasets=SimpleNamespace(**{k: abs_path(v) for k, v in p["datasets"].items()}),
        models=SimpleNamespace(**{k: abs_path(v) for k, v in p["models"].items()}),
        logits=SimpleNamespace(**{k: abs_path(v) for k, v in p["logits"].items()}),
        tables=SimpleNamespace(**{k: abs_path(v) for k, v in p["tables"].items()}),
        venvs=SimpleNamespace(**{k: abs_path(v) for k, v in p["venvs"].items()}),
    )

    def dict_to_ns(d):
        """Recursively convert dictionaries into SimpleNamespace objects."""
        if not isinstance(d, dict):
            return d
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})

    ns_eval = dict_to_ns(raw["eval"])
    ns_analysis = dict_to_ns(raw["analysis"])
    ns_solutions = dict_to_ns(raw.get("solutions", {}))
    ns_pipeline = dict_to_ns(raw.get("pipeline", {}))

    return SimpleNamespace(
        paths=ns_paths,
        eval=ns_eval,
        analysis=ns_analysis,
        solutions=ns_solutions,
        pipeline=ns_pipeline,
        is_colab=is_colab,
    )


# Single shared instance; import this in every script.
cfg = _load()
