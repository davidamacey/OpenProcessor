# =============================================================================
# OpenProcessor FastAPI Service
# Multi-stage build with non-root user and production defaults
# =============================================================================
# Builds the yolo-api FastAPI service for visual AI inference.
# docker-compose.yml overrides CMD with environment-specific worker counts.
#
# VOLUME MOUNTS (for development):
#   - ./src:/app/src              (hot reload)
#   - ./pytorch_models:/app/pytorch_models
#   - ./VERSION:/app/VERSION:ro
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build - Install Python dependencies with compilation tools
# -----------------------------------------------------------------------------
FROM python:3.13-slim-trixie AS builder

WORKDIR /build

# Install build dependencies (isolated to this stage).
# apt-get upgrade pulls Debian point-release security fixes the base tag
# may lag behind.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --user --no-cache-dir --upgrade pip \
    && pip install --user --no-cache-dir --no-warn-script-location \
        --extra-index-url https://pypi.nvidia.com \
        -r requirements.txt \
    && pip install --user --no-cache-dir --no-warn-script-location --no-deps \
        "perception_models @ git+https://github.com/facebookresearch/perception_models.git@3e352cca660658d4b5c90f42a7808b11469e4c66"

# Isolated YOLO11 EfficientNMS export toolchain (ultralytics pinned <8.4 for
# the vendored end2end patch; CPU torch — see requirements-export-y11.txt).
# export/export_models.py re-execs into this venv automatically.
COPY requirements-export-y11.txt .

RUN python -m venv /opt/venv-y11 \
    && /opt/venv-y11/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv-y11/bin/pip install --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com \
        -r requirements-export-y11.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal image with only runtime dependencies
# -----------------------------------------------------------------------------
FROM python:3.13-slim-trixie

LABEL org.opencontainers.image.title="OpenProcessor FastAPI Service" \
      org.opencontainers.image.description="Visual AI API with object detection, face recognition, embeddings, and OCR" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.authors="OpenProcessor Contributors" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor" \
      org.opencontainers.image.documentation="https://github.com/davidamacey/OpenProcessor/blob/main/README.md"

# Build identity: the commit this image was built from, baked in so every
# manifest's code_versions.api_sha reflects the running image rather than a
# dev checkout's live `git rev-parse` fallback (docker-compose.yml passes
# this as a build arg; empty stays "unknown" via _code_sha()). A runtime
# OP_BUILD_SHA still overrides -- see env.template.
ARG OP_BUILD_SHA=""
ENV OP_BUILD_SHA=${OP_BUILD_SHA}
LABEL org.opencontainers.image.revision=${OP_BUILD_SHA}

# Runtime-only system packages (no build tools); upgrade first for
# Debian point-release security fixes.
# procps -> pgrep, used by the curation worker services' healthchecks
# (docker-compose.yml `profiles: [curation]`) — this image is shared
# between yolo-api and those long-running worker processes.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    curl \
    jq \
    procps \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Non-root user with video group for GPU access
RUN groupadd -r appuser && \
    useradd -r -g appuser -G video -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app /app/logs && \
    chown -R appuser:appuser /app && \
    mkdir -p /home/appuser/.cache/huggingface \
             /home/appuser/.cache/torch && \
    chown -R appuser:appuser /home/appuser/.cache && \
    # CurationConfig.state_dir default and the shared /jobs dir (auto-label,
    # training, scores, select, viz job files) -- created + owned by appuser
    # here so a fresh named Docker volume mounted over either path on first
    # `up` inherits appuser ownership (Docker copies an empty named volume's
    # ownership from the image directory it's mounted over).
    mkdir -p /var/lib/openprocessor /jobs && \
    chown -R appuser:appuser /var/lib/openprocessor /jobs

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Pinned YOLO11 export venv (see requirements-export-y11.txt)
COPY --from=builder --chown=appuser:appuser /opt/venv-y11 /opt/venv-y11

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TORCH_HOME=/home/appuser/.cache/torch

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser VERSION ./VERSION

USER appuser

EXPOSE 8000

# /live = process liveness only. /health now probes dependencies (Triton,
# OpenSearch) and returns 503 while any is down — gating the container's
# health on it would mark yolo-api unhealthy (and cascade via
# depends_on: service_healthy) whenever a *downstream* dep degrades.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/live || exit 1

# Production defaults (docker-compose.yml overrides workers, backlog, etc.)
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "16", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--backlog", "4096", \
     "--limit-concurrency", "512", \
     "--timeout-keep-alive", "75", \
     "--timeout-graceful-shutdown", "30", \
     "--access-log", \
     "--log-level", "info"]
