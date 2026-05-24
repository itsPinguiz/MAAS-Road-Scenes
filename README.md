<div align="center">
  <img src="assets/banner.png" alt="MAAS-Road-Scenes Banner" width="800">

  # MAAS-Road-Scenes

  **Multi-Armed Adversarial Selection for Road Scene Anomaly Detection**

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

## Overview

MAAS-Road-Scenes is an evaluation and analysis workspace for semantic segmentation
anomaly detection in road scenes. It compares ERFNet and EoMT
(Encoder-only Mask Transformer) on common out-of-distribution road benchmarks,
tracks calibration behavior with temperature sweeps, and provides fine-grained
analysis of where models fail: boundaries, object scale, depth bands, semantic
confusions, resolution trade-offs, and attention behavior.

The project is organized around a config-driven pipeline in `config/config.yml`.
Most scripts resolve paths from that file, so experiments can be rerun locally or
in Colab without editing individual scripts.

## What Is Included

- ERFNet and EoMT evaluation wrappers for Cityscapes mIoU and OOD benchmarks.
- OOD metrics for MSP, MaxLogit, Max Entropy, and RbA where supported.
- Temperature scaling sweeps for calibration analysis.
- Logit-saving support for offline fine-grained analysis.
- Analysis scripts for semantic boundaries, anomaly object size, depth bands,
  things/stuff false positives, resolution/FPS, and EoMT attention overlays.
- EoMT fine-tuning utilities with COCO outlier cutout pasting and checkpoint
  selection.
- Markdown result tables and analysis reports under `results/`.

## Repository Layout

```text
MAAS-Road-Scenes/
├── assets/                  # README banner and project media
├── config/config.yml         # Central paths, flags, and experiment settings
├── core/
│   ├── analysis/             # Fine-grained analysis scripts
│   ├── evaluation/           # Evaluation pipeline and model-specific wrappers
│   ├── solutions/            # Outlier augmentation, fine-tuning, checkpoint ranking
│   └── utility/              # Config loader, logging, table updates
├── third_party/
│   ├── eval/                 # ERFNet evaluation code and requirements
│   └── eomt/                 # EoMT code, configs, docs, and requirements
├── results/                  # Generated tables, reports, plots, checkpoint rankings
├── Datasets/                 # Local datasets; not intended for source control
└── checkpoints/              # Local model weights; not intended for source control
```

## Setup

This project uses separate environments because ERFNet and EoMT have different
dependency stacks. The paths below match the defaults in `config/config.yml`.

```bash
git clone https://github.com/itsPinguiz/MAAS-Road-Scenes.git
cd MAAS-Road-Scenes

# ERFNet/general evaluation environment
python -m venv core/evaluation/eval/.venv_eval
core/evaluation/eval/.venv_eval/bin/python -m pip install --upgrade pip
core/evaluation/eval/.venv_eval/bin/python -m pip install \
  -r third_party/eval/requirements.txt \
  -r core/requirements.txt

# EoMT environment
python -m venv core/evaluation/eomt/.venv_eomt
core/evaluation/eomt/.venv_eomt/bin/python -m pip install --upgrade pip
core/evaluation/eomt/.venv_eomt/bin/python -m pip install \
  -r third_party/eomt/requirements.txt \
  -r core/requirements.txt
```

PyTorch wheels are CUDA-specific. If the pinned versions in the requirements do
not match your machine, install the correct PyTorch build first from the official
PyTorch selector, then install the remaining requirements.

## Data And Checkpoints

Configure all local paths in `config/config.yml`. By default the project expects:

```text
Datasets/Cityscapes
Datasets/Fishyscapes/fs_static/images/*.jpg
Datasets/Fishyscapes/FS_LostFound_full/images/*.png
Datasets/RoadAnomaly/images/*.jpg
Datasets/SegmentMeIfYouCan/RoadAnomaly21/images/*.png
Datasets/SegmentMeIfYouCan/RoadObsticle21/images/*.webp
checkpoints/erfnet_pretrained.pth
checkpoints/eomt_cityscapes.bin
checkpoints/solutions/<run>/epoch_*_EoMT.pth
```

The current config points EoMT evaluation at:

```text
checkpoints/solutions/20260516_130935/epoch_3_EoMT.pth
```

Set `pipeline.run_erfnet` and `pipeline.run_eomt` in `config/config.yml` to choose
which model families are evaluated.

## Run Evaluation

From the repository root:

```bash
core/evaluation/eval/.venv_eval/bin/python core/evaluation/run_pipeline.py
```

The pipeline runs the enabled stages in order:

1. ERFNet Cityscapes mIoU and optional logits.
2. ERFNet OOD datasets.
3. EoMT Cityscapes mIoU and optional logits.
4. EoMT OOD datasets.
5. ERFNet temperature sweep.
6. EoMT temperature sweep.

To generate logits for offline analysis, set this in `config/config.yml` before
running the pipeline:

```yaml
eval:
  save_logits: true
```

Result tables are written to the configured table paths, currently
`results/TABLE.md` and `results/TABLE_T.md`. Additional snapshots are kept under
`results/evaluation/`.

## Fine-Grained Analysis

After saving logits, run the full analysis pipeline:

```bash
core/evaluation/eomt/.venv_eomt/bin/python core/analysis/run_full_analysis_pipeline.py
```

Outputs are written to `results/analysis_reports/`, with one subdirectory per
model and dataset plus a summary at `results/analysis_reports/SUMMARY.md`.

Individual analysis scripts can also be run directly:

```bash
core/evaluation/eomt/.venv_eomt/bin/python core/analysis/eval_semantics.py \
  --logits_dir core/evaluation/eomt/saved_logits/eomt/<checkpoint>/<dataset> \
  --images_glob "Datasets/Fishyscapes/fs_static/images/*.jpg" \
  --dataset_type fs_static \
  --model_name EoMT

core/evaluation/eomt/.venv_eomt/bin/python core/analysis/eval_objects.py \
  --logits_dir core/evaluation/eomt/saved_logits/eomt/<checkpoint>/<dataset> \
  --images_glob "Datasets/Fishyscapes/fs_static/images/*.jpg" \
  --dataset_type fs_static \
  --model_name EoMT

core/evaluation/eomt/.venv_eomt/bin/python core/analysis/eval_resolution_attention.py \
  --images_glob "Datasets/Fishyscapes/fs_static/images/*.jpg" \
  --dataset_type fs_static \
  --run_attention_eval \
  --model_weights checkpoints/solutions/<run>/epoch_<n>_EoMT.pth
```

## Fine-Tuning Workflow

The `core/solutions` package contains the current mitigation experiments:
COCO outlier extraction, EoMT fine-tuning with pasted OOD objects, and checkpoint
ranking.

Prepare transparent COCO cutouts:

```bash
core/evaluation/eomt/.venv_eomt/bin/python core/solutions/prepare_coco_outliers.py \
  --img-dir Datasets/COCO/val2017 \
  --ann-file Datasets/COCO/annotations_trainval2017/annotations/instances_val2017.json \
  --output-dir Datasets/Outliers_COCO
```

Run the configured EoMT fine-tuning job:

```bash
core/evaluation/eomt/.venv_eomt/bin/python core/solutions/train.py
```

Rank checkpoints from a run:

```bash
core/evaluation/eomt/.venv_eomt/bin/python core/solutions/select_best_checkpoint.py \
  checkpoints/solutions/<run> \
  --python core/evaluation/eomt/.venv_eomt/bin/python \
  --device cuda \
  --miou-baseline 60.27
```

Selection reports are written under `results/checkpoint_selection/`.

## Results

Generated artifacts live in:

- `results/TABLE.md` - current main evaluation table.
- `results/evaluation/new/TABLE.md` - latest stored evaluation snapshot.
- `results/evaluation/new/TABLE_T.md` - latest stored temperature snapshot.
- `results/analysis_reports/SUMMARY.md` - fine-grained analysis run summary.
- `results/checkpoint_selection/` - checkpoint ranking CSV/Markdown reports.

The repository keeps generated Markdown tables and plots for traceability. Large
datasets and model checkpoints should remain local.

## Credits

This project builds on the following open-source research code and papers:

**ERFNet**

Eduardo Romera, Jose M. Alvarez, Luis M. Bergasa, and Roberto Arroyo.
*"ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation"*,
IEEE Transactions on Intelligent Transportation Systems, 2017.

**EoMT (Encoder-only Mask Transformer)**

Tommie Kerssies, Niccolo Cavagnero, Alexander Hermans, Narges Norouzi,
Giuseppe Averta, Gijs Dubbelman, and Daan de Geus.
*"Encoder-only Mask Transformer for Image Segmentation"*, CVPR 2024.
Original repository: https://github.com/tue-mps/eomt

## License

This repository is released under the MIT License. See `LICENSE` for details.
