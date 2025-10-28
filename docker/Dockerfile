# Dockerfile: python3 + CUDA 12.8
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    torch numpy --index-url https://download.pytorch.org/whl/cu128 && \
    rm -rf /var/lib/apt/lists/*

COPY train.py /workspace/train.py
CMD ["python3", "/workspace/train.py"]
