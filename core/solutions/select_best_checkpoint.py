"""Evaluate solution checkpoints and rank the best EoMT fine-tuning epoch."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.utility.config_loader import cfg
from core.utility.logger import logger


DATASETS = {
    "Fishyscapes Static": cfg.paths.datasets.fishyscapes_static,
    "Fishyscapes Lost & Found": cfg.paths.datasets.fishyscapes_lost_found,
    "RoadAnomaly": cfg.paths.datasets.road_anomaly,
    "RoadAnomaly21": cfg.paths.datasets.road_anomaly21,
    "RoadObsticle21": cfg.paths.datasets.road_obstacle21,
}

METHODS = ("msp", "maxlogit", "maxentropy", "rba")
METRIC_VALUE_RE = r"(?:nan|[+-]?inf|[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?)"


@dataclass
class CheckpointResult:
    """All metrics collected for one checkpoint."""

    checkpoint: Path
    epoch: int
    miou: float | None = None
    metrics: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    score: float | None = None

    @property
    def mean_auprc(self) -> float:
        values = [v[0] if math.isfinite(v[0]) else 0.0 for v in self.metrics.values()]
        return sum(values) / len(values) if values else float("nan")

    @property
    def mean_fpr95(self) -> float:
        values = [v[1] if math.isfinite(v[1]) else 100.0 for v in self.metrics.values()]
        return sum(values) / len(values) if values else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank EoMT fine-tuning checkpoints by validation metrics.")
    parser.add_argument("run_dir", help="Directory containing epoch_*_EoMT.pth checkpoints")
    parser.add_argument("--python", default=sys.executable, help="Python executable with EoMT dependencies")
    parser.add_argument("--device", default="cuda", help="Device passed to eval scripts")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--skip-miou", action="store_true", help="Only rank OOD metrics")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--crop-batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=None, help="Evaluate only the first N checkpoints")
    parser.add_argument("--miou-baseline", type=float, default=None, help="Reference mIoU for penalty, e.g. original model mIoU")
    parser.add_argument("--miou-penalty", type=float, default=2.0, help="Penalty multiplier for mIoU drop below baseline")
    parser.add_argument("--fpr-weight", type=float, default=1.0, help="Penalty multiplier for mean FPR95")
    parser.add_argument("--output-dir", default=None, help="Defaults to results/checkpoint_selection/<run_dir_name>")
    return parser.parse_args()


def epoch_from_path(path: Path) -> int:
    match = re.search(r"epoch_(\d+)_EoMT\.pth$", path.name)
    if not match:
        raise ValueError(f"Could not parse epoch from {path}")
    return int(match.group(1))


def find_checkpoints(run_dir: Path, max_epochs: int | None) -> list[Path]:
    checkpoints = sorted(
        run_dir.glob("epoch_*_EoMT.pth"),
        key=epoch_from_path,
    )
    if max_epochs is not None:
        checkpoints = checkpoints[:max_epochs]
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_*_EoMT.pth checkpoints found in {run_dir}")
    return checkpoints


def run_command(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            f"{' '.join(command)}\n\n{output}"
        )
    return output


def evaluate_miou(args: argparse.Namespace, ckpt: Path) -> float:
    eomt_dir = Path(cfg.paths.root) / "core" / "evaluation" / "eomt"
    command = [
        args.python,
        "eval_miou_eomt.py",
        "--ckpt_path",
        str(ckpt),
        "--device",
        args.device,
        "--quiet",
        "--no-update-table",
        "--num-workers",
        str(args.num_workers),
        "--crop-batch-size",
        str(args.crop_batch_size),
    ]
    output = run_command(command, cwd=eomt_dir)
    match = re.search(rf"mIoU:\s*({METRIC_VALUE_RE})%", output, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"Could not parse mIoU from output for {ckpt}:\n{output}")
    return float(match.group(1))


def evaluate_ood(args: argparse.Namespace, ckpt: Path, dataset_name: str) -> dict[str, tuple[float, float]]:
    eomt_dir = Path(cfg.paths.root) / "core" / "evaluation" / "eomt"
    command = [
        args.python,
        "evalAnomaly_eomt.py",
        "--ckpt_path",
        str(ckpt),
        "--input",
        DATASETS[dataset_name],
        "--dataset_name",
        dataset_name,
        "--device",
        args.device,
        "--quiet",
        "--no-update-table",
        "--num-workers",
        str(args.num_workers),
        "--crop-batch-size",
        str(args.crop_batch_size),
    ]
    output = run_command(command, cwd=eomt_dir)
    parsed: dict[str, tuple[float, float]] = {}
    pattern = re.compile(
        rf"Method:\s*([^,]+),\s*AuPRC:\s*({METRIC_VALUE_RE}),\s*FPR95:\s*({METRIC_VALUE_RE})",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(output):
        method = match.group(1).strip().lower().replace(" ", "")
        parsed[method] = (float(match.group(2)), float(match.group(3)))
    missing = [method for method in args.methods if method not in parsed]
    if missing:
        logger.warning(
            f"Missing metrics for {missing} on {dataset_name} / {ckpt}; "
            "using AuPRC=nan, FPR95=100.0 for ranking."
        )
        for method in missing:
            parsed[method] = (float("nan"), 100.0)
    return parsed


def compute_score(result: CheckpointResult, args: argparse.Namespace) -> float:
    score = result.mean_auprc - (args.fpr_weight * result.mean_fpr95)
    if args.miou_baseline is not None and result.miou is not None:
        miou_drop = max(0.0, args.miou_baseline - result.miou)
        score -= args.miou_penalty * miou_drop
    return score


def write_reports(results: list[CheckpointResult], output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "checkpoint_selection.csv"
    md_path = output_dir / "checkpoint_selection.md"

    rows = []
    for result in sorted(results, key=lambda r: r.score if r.score is not None else -math.inf, reverse=True):
        row = {
            "rank": len(rows) + 1,
            "epoch": result.epoch,
            "checkpoint": str(result.checkpoint),
            "score": f"{result.score:.4f}" if result.score is not None else "nan",
            "miou": f"{result.miou:.2f}" if result.miou is not None else "",
            "mean_auprc": f"{result.mean_auprc:.2f}",
            "mean_fpr95": f"{result.mean_fpr95:.2f}",
        }
        for dataset in args.datasets:
            for method in args.methods:
                values = result.metrics.get((dataset, method))
                if values:
                    row[f"{dataset}/{method}/auprc"] = f"{values[0]:.2f}"
                    row[f"{dataset}/{method}/fpr95"] = f"{values[1]:.2f}"
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["rank", "epoch", "checkpoint", "score", "miou", "mean_auprc", "mean_fpr95"]
    fieldnames = preferred + [field for field in fieldnames if field not in preferred]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Checkpoint Selection\n\n")
        f.write(f"Run directory: `{args.run_dir}`\n\n")
        f.write("| Rank | Epoch | Score | mIoU | Mean AuPRC | Mean FPR95 | Checkpoint |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['rank']} | {row['epoch']} | {row['score']} | {row['miou']} | "
                f"{row['mean_auprc']} | {row['mean_fpr95']} | `{row['checkpoint']}` |\n"
            )

    logger.success(f"Wrote selection CSV: {csv_path}")
    logger.success(f"Wrote selection report: {md_path}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path(cfg.paths.root) / run_dir
    args.run_dir = str(run_dir)

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.paths.root) / "results" / "checkpoint_selection" / run_dir.name
    checkpoints = find_checkpoints(run_dir, args.max_epochs)
    logger.info(f"Evaluating {len(checkpoints)} checkpoints from {run_dir}")

    results: list[CheckpointResult] = []
    for ckpt in checkpoints:
        result = CheckpointResult(checkpoint=ckpt, epoch=epoch_from_path(ckpt))
        logger.info(f"[epoch {result.epoch}] Evaluating {ckpt.name}")

        if not args.skip_miou:
            result.miou = evaluate_miou(args, ckpt)
            logger.info(f"[epoch {result.epoch}] mIoU={result.miou:.2f}")

        for dataset in args.datasets:
            dataset_metrics = evaluate_ood(args, ckpt, dataset)
            for method in args.methods:
                result.metrics[(dataset, method)] = dataset_metrics[method]
            logger.info(f"[epoch {result.epoch}] {dataset} done")

        result.score = compute_score(result, args)
        logger.info(
            f"[epoch {result.epoch}] score={result.score:.2f}, "
            f"mean AuPRC={result.mean_auprc:.2f}, mean FPR95={result.mean_fpr95:.2f}"
        )
        results.append(result)
        write_reports(results, output_dir, args)

    best = max(results, key=lambda r: r.score if r.score is not None else -math.inf)
    logger.success(f"Best checkpoint: epoch {best.epoch} | score={best.score:.2f} | {best.checkpoint}")


if __name__ == "__main__":
    main()
