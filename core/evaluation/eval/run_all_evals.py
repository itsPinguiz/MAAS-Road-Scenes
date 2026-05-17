import os
import re
import subprocess
import sys
import warnings

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.utility.config_loader import cfg
from core.utility.runtime import get_device, print_device_health

DEVICE = get_device()

warnings.filterwarnings("ignore", ".*'network' is an instance.*")

from core.utility.update_table import update_table_entry
from core.utility.logger import logger, console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

DATASETS = {
    'Fishyscapes Static':      cfg.paths.datasets.fishyscapes_static,
    'Fishyscapes Lost & Found': cfg.paths.datasets.fishyscapes_lost_found,
    'RoadAnomaly':             cfg.paths.datasets.road_anomaly,
    'RoadAnomaly21':           cfg.paths.datasets.road_anomaly21,
    'RoadObsticle21':          cfg.paths.datasets.road_obstacle21,
}

METHODS = ['msp', 'maxlogit', 'maxentropy']
MODEL = 'ERFNet'


def main():
    """Run ERFNet anomaly evaluations for all configured OOD datasets."""
    print_device_health(DEVICE)
    logger.info(f"Starting bulk evaluation for {MODEL}...")
    logger.info(f"Total datasets to test: {len(DATASETS)} (computing all {len(METHODS)} methods simultaneously)")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("Evaluating ERFNet", total=len(DATASETS))

        for dataset_name, dataset_path in DATASETS.items():
            progress.update(task_id, description=f"Eval: {dataset_name}")

            cmd = [
                sys.executable, "evalAnomaly.py",
                "--input", dataset_path,
                "--quiet",
                "--dataset_name", dataset_name,
                "--loadWeights", cfg.paths.models.erfnet_weights,
            ]
            if getattr(cfg.eval, "save_logits", False):
                cmd.append("--save_logits")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Error running eval on {dataset_name}:\n{result.stderr}")
                progress.advance(task_id)
                continue

            auprc_dict = {m: "N/A" for m in METHODS}
            fpr_dict = {m: "N/A" for m in METHODS}

            for line in result.stdout.split('\n'):
                match = re.search(r"Method:\s*([A-Za-z]+)\s*->\s*AUPRC score:\s*([0-9.]+),\s*FPR@TPR95:\s*([0-9.]+)", line)
                if match:
                    method_parsed = match.group(1).lower()
                    if method_parsed in METHODS:
                        auprc_dict[method_parsed] = match.group(2)
                        fpr_dict[method_parsed] = match.group(3)

            for method in METHODS:
                if auprc_dict[method] != "N/A" and fpr_dict[method] != "N/A":
                    table_method = 'Max Entropy' if method == 'maxentropy' else method.upper()
                    update_table_entry(model="ERFNET", method=table_method, dataset=dataset_name, miou='-', auprc=auprc_dict[method], fpr95=fpr_dict[method])
                    logger.info(f"-> {dataset_name} | {table_method}: AUPRC={auprc_dict[method]}, FPR95={fpr_dict[method]}")

            progress.advance(task_id)

    logger.success("Updated TABLE.md with the results for ERFNET.")


if __name__ == "__main__":
    main()
