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

# Install build dependencies (isolated to this stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
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
        -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal image with only runtime dependencies
# -----------------------------------------------------------------------------
FROM python:3.13-slim-trixie

LABEL org.opencontainers.image.title="OpenProcessor FastAPI Service" \
      org.opencontainers.image.description="Visual AI API with object detection, face recognition, embeddings, and OCR" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.authors="OpenProcessor Contributors" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor" \
      org.opencontainers.image.documentation="https://github.com/davidamacey/OpenProcessor/blob/main/README.md"

# Runtime-only system packages (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
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
    chown -R appuser:appuser /home/appuser/.cache

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

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

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

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
