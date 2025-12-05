# Utilizziamo l'immagine base NVIDIA con CUDA 11.7 e CUDNN 8 su Ubuntu 22.04.
# La tag 'devel' è CRUCIALE: contiene nvcc e gli header necessari per compilare
# estensioni custom di PyTorch (come quelle in Detectron2).
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Impostiamo variabili d'ambiente per evitare interattività durante l'installazione apt
# e per configurare correttamente la compilazione CUDA.
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

# Installazione delle dipendenze di sistema.
# git, wget, curl: tool base.
# python3-dev, pip: ambiente Python.
# libgl1, libglib2.0-0: dipendenze grafiche per OpenCV (cv2).
# ninja-build: sistema di build veloce richiesto da PyTorch/Detectron2.
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    unzip \
    ca-certificates \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Aggiornamento di pip
RUN pip install --upgrade pip

# Installazione di PyTorch, TorchVision e TorchAudio.
# Specifichiamo l'URL dell'indice CUDA 11.7.
RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Pre-installazione di numpy e ninja per sicurezza
RUN pip install numpy ninja

# Installazione di Detectron2.
# FIX APPLICATO: --no-build-isolation
# Questo flag forza pip a usare le librerie installate nel sistema (incluso torch)
# invece di creare un ambiente temporaneo isolato vuoto.
RUN python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# Installazione delle librerie Python aggiuntive.
# timm: PyTorch Image Models (necessario per backbone ViT).
# einops: Operazioni su tensori (usato in molti Transformer).
# wandb: Experiment tracking.
RUN pip install \
    timm \
    scikit-learn \
    scikit-image \
    opencv-python \
    matplotlib \
    tqdm \
    pyyaml \
    h5py \
    wandb \
    tensorboard \
    einops \
    yacs \
    termcolor \
    shapely \
    submitit \
    natten

# Creazione della directory di lavoro
WORKDIR /workspace

# Copia del file requirements.txt (per eventuali dipendenze future o versioni specifiche)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Configurazione del comando di default
CMD ["/bin/bash"]