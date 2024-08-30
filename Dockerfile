# Not well tested yet

FROM nvcr.io/nvidia/pytorch:24.02-py3 as base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sudo \
    htop \
    git \
    wget \
    tmux \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir transformers sentencepiece

RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
cd flash-attention && cd csrc/fused_dense_lib && pip install . && cd ../layer_norm && pip install .