# =============================================================================
# OpenProcessor segmenter — promptable segmentation service (SAM 3).
# =============================================================================
# Serves the detection cascade's segmenter leg: the worker
# (scripts/curation/worker/client.py) POSTs a crop + a text prompt and
# gets candidate boxes back. See ./README.md for the wire contract,
# tuning knobs and GPU sizing, and ./NOTICE for attribution.
#
# Build:
#   docker compose --profile segmenter build segmenter
#   docker compose --profile segmenter build \
#     --build-arg SAM3_SHA=<sha> segmenter      # re-pin upstream SAM 3
#
# Run (needs a GPU and a HuggingFace token for the gated weights):
#   OP_SEGMENTER_URL=http://segmenter:8000 \
#     docker compose --profile curation --profile segmenter up -d
# =============================================================================

# -- Stage 1: fetch upstream SAM 3 ------------------------------------------
# Isolated so neither git nor the clone's history reaches the runtime
# image. Pin to a known-good upstream SHA for reproducible rebuilds;
# `main` is the default only because upstream publishes no releases.
FROM alpine/git:2.47.2 AS sam3-src

ARG SAM3_SHA=main
WORKDIR /src
RUN git clone https://github.com/facebookresearch/sam3.git sam3
WORKDIR /src/sam3
RUN git checkout "$SAM3_SHA" && rm -rf .git

# -- Stage 2: runtime --------------------------------------------------------
# No -slim base is available for a CUDA + cuDNN + PyTorch runtime; this
# is the smallest official image that carries the stack the model needs.
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

LABEL org.opencontainers.image.title="OpenProcessor Segmenter" \
      org.opencontainers.image.description="Promptable segmentation (SAM 3) for the detection cascade" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"

# libgl1/libglib: OpenCV runtime. gcc/g++: torch.compile's inductor
# backend builds C extensions on the first call, so SEGMENTER_COMPILE=1
# fails with "Failed to find C compiler" without them.
# hadolint ignore=DL3008
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=sam3-src /src/sam3 /opt/sam3

# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /opt/sam3 \
    && pip install --no-cache-dir \
        'fastapi==0.115.4' \
        'uvicorn[standard]==0.32.0' \
        'pydantic>=2.7,<3' \
        'numpy>=1.26,<2.0' \
        'pillow>=10.0' \
        'opencv-python-headless>=4.10' \
        'scikit-image>=0.24' \
        'huggingface-hub>=0.24' \
        'einops>=0.8' \
        'pycocotools>=2.0.7'

RUN groupadd -r appuser && \
    useradd -r -g appuser -G video -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser/.cache

WORKDIR /app
COPY --chown=appuser:appuser main.py sam3_backend.py /app/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SEGMENTER_DEVICE=cuda:0 \
    SEGMENTER_PORT=8000 \
    SEGMENTER_LOG_LEVEL=info \
    HF_HOME=/home/appuser/.cache/huggingface \
    TORCH_HOME=/home/appuser/.cache/torch

USER appuser

EXPOSE 8000

# Long start period: the first boot downloads weights and builds the
# processor pool. /health answers 200 with loaded=false while that runs,
# so probe the flag, not just the socket.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8000/health | grep -q '"loaded":true' || exit 1

CMD ["python", "main.py"]
