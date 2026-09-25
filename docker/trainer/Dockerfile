# =============================================================================
# Curation trainer -- YOLO training service for the API's job-file protocol.
# =============================================================================
# Watches the shared jobs volume for *.job.json files and runs each through
# Ultralytics. The API half of the protocol lives in
# src/services/training/jobs.py; this image implements the trainer half
# (docker/trainer/trainer.py). Nothing domain-specific is baked in -- the
# dataset, class subset, hyperparameters and augmentation all arrive in
# job.json, and deployment paths arrive as OP_* env vars (env.template).
#
# Multi-stage + slim base + non-root, matching this repo's other images
# (./Dockerfile, docker/evaluator/Dockerfile). Torch comes from the PyTorch
# CUDA wheel index, which bundles the CUDA runtime libraries -- so this stays
# on a python:*-slim base rather than a multi-GB nvidia/cuda one. The host
# still needs an NVIDIA driver and the container toolkit; GPU access is
# granted by compose (`deploy.resources.reservations.devices`).
#
# Runtime mounts (docker-compose.yml, profile `training`):
#   /jobs                                -> job + status + manifest files
#   /var/lib/openprocessor/training_runs -> Ultralytics run outputs
#   the export root                      -> frozen dataset exports (read-only)
#
# IMPORTANT: the runs dir must be mounted at the SAME absolute path in this
# container and in yolo-api. The trainer records `checkpoint_path` verbatim in
# status.json, and promote resolves that exact string from the API container.
#
# Build:
#   docker compose --profile training build curation-trainer
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: build -- resolve every wheel into a relocatable --user tree
# -----------------------------------------------------------------------------
FROM python:3.12-slim-trixie AS builder

WORKDIR /build

# git is needed to install Ultralytics from a pinned commit.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA wheel index. Override for a different CUDA minor
# (--build-arg TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124).
ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128

# Ultralytics is pinned to a commit rather than a release tag: the training
# recipe (and the MuSGD optimizer path) moves between releases, and a run's
# manifest records this SHA so the image can be rebuilt at the same point in
# time. Override with --build-arg ULTRALYTICS_SHA=<sha>.
ARG ULTRALYTICS_SHA=8f5d355cd05b91503e7bf62429681dea4fa4b004

COPY docker/trainer/requirements.txt requirements.txt

# hadolint ignore=DL3013
RUN pip install --user --no-cache-dir --upgrade pip \
    && pip install --user --no-cache-dir --no-warn-script-location \
        --extra-index-url "${TORCH_CUDA_INDEX}" \
        -r requirements.txt \
    && pip install --user --no-cache-dir --no-warn-script-location \
        "ultralytics @ git+https://github.com/ultralytics/ultralytics.git@${ULTRALYTICS_SHA}"

# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim-trixie

LABEL org.opencontainers.image.title="OpenProcessor Curation Trainer" \
      org.opencontainers.image.description="YOLO training service driven by the curation job-file protocol" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"

# Build identity: the commit this trainer image was built from. Read by
# job_protocol.build_lineage() as code_versions.trainer_sha (preferred over
# the API's submit-time trainer_image_revision observation, since this is
# the image actually running the code) -- see docker-compose.yml build.args.
ARG OP_BUILD_SHA=""
ENV OP_BUILD_SHA=${OP_BUILD_SHA}
LABEL org.opencontainers.image.revision=${OP_BUILD_SHA}

ARG ULTRALYTICS_SHA=8f5d355cd05b91503e7bf62429681dea4fa4b004
# Promoted from ARG to ENV so it is readable at *runtime*
# (`os.environ['ULTRALYTICS_SHA']`) and lands in each run's manifest under
# code_versions.ultralytics_sha. An ARG alone only exists at build time.
ENV ULTRALYTICS_SHA=${ULTRALYTICS_SHA}

# Runtime-only system packages: cv2/Pillow image libs, curl+jq for probes,
# procps for the healthcheck's pgrep.
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

# Non-root user, uid 1000 with the video group -- same as this repo's other
# images so bind-mounted volumes keep consistent ownership on the host.
RUN groupadd -r appuser \
    && useradd -r -g appuser -G video -u 1000 -m -s /bin/bash appuser \
    && mkdir -p /app /jobs /var/lib/openprocessor/training_runs \
                /home/appuser/.cache/torch /home/appuser/.config/Ultralytics \
    && chown -R appuser:appuser /app /jobs /var/lib/openprocessor /home/appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TORCH_HOME=/home/appuser/.cache/torch \
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics \
    OP_TRAIN_JOBS_DIR=/jobs \
    OP_TRAIN_RUNS_ROOT=/var/lib/openprocessor/training_runs

# Flat module layout: /app is on PYTHONPATH, so `python -m trainer` resolves
# trainer.py and its siblings as top-level modules.
COPY --chown=appuser:appuser docker/trainer/*.py ./

USER appuser

# The watcher is healthy while it is running and can see the jobs volume.
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD test -d "$OP_TRAIN_JOBS_DIR" && pgrep -f trainer.py > /dev/null || exit 1

ENTRYPOINT ["python", "-m", "trainer"]
CMD ["--watch", "/jobs"]
