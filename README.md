<div align="center">
  <img src="assets/banner.png" alt="MAAS-Road-Scenes Banner" width="800">

  # MAAS-Road-Scenes
  
  **Multi-Armed Adversarial Selection for Road Scene Anomaly Detection**

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  ---
</div>

## Overview

**MAAS-Road-Scenes** is a comprehensive evaluation pipeline designed for robust semantic segmentation and anomaly detection in road environments. It integrates state-of-the-art models like **ERFNet** and **EoMT** (Encoder-only Mask Transformer) to evaluate their performance across diverse out-of-distribution (OOD) datasets.

The project focuses on automated benchmarking, temperature scaling analysis, and modular evaluation across multiple datasets, providing a unified interface for researchers to assess model reliability in safety-critical autonomous driving scenarios.

## Key Features

- **Multi-Model Support**: Native integration of **ERFNet** and **EoMT**.
- **Automated Pipeline**: End-to-end evaluation from inference to results table generation.
- **Temperature Scaling**: Integrated tools for OOD detection calibration via temperature scaling.
- **Extensive Datasets**: Pre-configured support for Cityscapes, Fishyscapes, RoadAnomaly, and more.
- **Markdown Reporting**: Automatic generation of summary tables in `results/`.

---

## Project Structure

```text
MAAS-Road-Scenes/
├── assets/             # Project media and banner
├── config/             # YAML configuration files
├── core/               # Core logic and pipeline scripts
│   ├── evaluation/     # Main evaluation suite (run_pipeline.py)
│   └── utility/        # Shared utilities (logger, config loader)
├── Datasets/           # Symbolic links or data folders (Cityscapes, etc.)
├── results/            # Automatically generated performance tables
├── third_party/        # Integrated researchers' codebases (ERFNet, EoMT)
└── checkpoints/        # Model weights and checkpoints
```

---

##  Getting Started

### 1. Prerequisites

This project uses **two distinct virtual environments** to manage dependencies between different model implementations (ERFNet and EoMT).

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/itsPinguiz/MAAS-Road-Scenes.git
cd MAAS-Road-Scenes

# Setup Virtual Environments (Example using venv)
# 1. Environment for general evaluation (ERFNet)
python -m venv core/evaluation/eval/.venv_eval
source core/evaluation/eval/.venv_eval/bin/python -m pip install -r core/evaluation/eval/requirements.txt

# 2. Environment for EoMT
python -m venv core/evaluation/eomt/.venv_eomt
source core/evaluation/eomt/.venv_eomt/bin/python -m pip install -r core/evaluation/eomt/requirements.txt
```

### 3. Usage

To run the entire evaluation pipeline, simply execute the `run_pipeline.py` script. This will orchestrate the full suite of tests across all configured models and datasets.

```bash
python core/evaluation/run_pipeline.py
```

Results will be saved as formatted tables in:
- `results/TABLE.md`: Main performance metrics.
- `results/TABLE_T.md`: Temperature scaling analysis.

---

##  Credits & Citations

This project integrates and builds upon several incredible open-source contributions. We extend our gratitude to the authors of the following papers:

### **ERFNet**
> **Eduardo Romera, José M. Álvarez, Luis M. Bergasa, and Roberto Arroyo.**  
> *"ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation"*  
> IEEE Transactions on Intelligent Transportation Systems (T-ITS), 2017.

### **EoMT (Encoder-only Mask Transformer)**
> **Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Gijs Dubbelman, and Daan de Geus.**  
> *"Encoder-only Mask Transformer for Image Segmentation"*  
> CVPR 2024. [[Original Repo](https://github.com/tue-mps/eomt)]

---
