# =============================================================================
# Curation trainer -- YOLO training worker: the consumer of the `gpu_train` queue.
# =============================================================================
# A Celery worker (one pool child per run, `--max-tasks-per-child=1`, so GPU memory is released
# when a run ends) that runs the `train` job kind through Ultralytics. The API half is
# src/services/training/jobs.py; this image carries the job runtime (src/tasks/runtime) and the
# database layer (src/db, src/config, src/domain) and no web or OpenSearch code. Nothing
# domain-specific is baked in -- the dataset, class subset, hyperparameters and augmentation all
# arrive in the run's spec (training_runs.spec), and deployment paths and the queue and database
# URLs arrive as OP_* env vars (env.template).
#
# Multi-stage + slim base + non-root, matching this repo's other images
# (./Dockerfile, which holds the api, export and evaluator targets). Torch comes from the PyTorch
# CUDA wheel index, which bundles the CUDA runtime libraries -- so this stays
# on a python:*-slim base rather than a multi-GB nvidia/cuda one. The host
# still needs an NVIDIA driver and the container toolkit; GPU access is
# granted by compose (`deploy.resources.reservations.devices`).
#
# Runtime mounts (docker-compose.yml, profile `training`):
#   /var/lib/openprocessor/training_runs -> Ultralytics run outputs
#   the projects data root               -> run logs and artifacts (storage root `projects`)
#   the export root                      -> frozen dataset exports (read-only)
#
# IMPORTANT: the runs dir must be mounted at the SAME absolute path in this
# container and in api. The trainer records `checkpoint_path` verbatim in
# the run's progress and manifest, and promote resolves that exact string from the API container.
#
# Build:
#   docker compose --profile training build curation-trainer
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: build -- resolve every wheel into a relocatable --user tree
# -----------------------------------------------------------------------------
FROM python:3.12-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS builder

WORKDIR /build

# git is needed to install Ultralytics from a pinned commit; libpq-dev is what the C build of
# psycopg (psycopg[c], like the api image) compiles against.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA wheel index. Override for a different CUDA minor
# (--build-arg TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130).
ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu132

# Ultralytics is pinned to a commit rather than a release tag: the training
# recipe (and the MuSGD optimizer path) moves between releases, and a run's
# manifest records this SHA so the image can be rebuilt at the same point in
# time. Override with --build-arg ULTRALYTICS_SHA=<sha>.
ARG ULTRALYTICS_SHA=8f5d355cd05b91503e7bf62429681dea4fa4b004

COPY docker/trainer/requirements.txt requirements.txt

# The base image's own pip does the install; no pip is copied into the runtime stage.
RUN pip install --user --no-cache-dir --no-warn-script-location \
        --extra-index-url "${TORCH_CUDA_INDEX}" \
        -r requirements.txt \
    && pip install --user --no-cache-dir --no-warn-script-location \
        "ultralytics @ git+https://github.com/ultralytics/ultralytics.git@${ULTRALYTICS_SHA}" \
    && rm -rf /root/.local/lib/python3.12/site-packages/pip \
        /root/.local/lib/python3.12/site-packages/pip-*.dist-info \
        /root/.local/bin/pip /root/.local/bin/pip3 /root/.local/bin/pip3.12

# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

LABEL org.opencontainers.image.title="OpenProcessor Curation Trainer" \
      org.opencontainers.image.description="YOLO training worker for the gpu_train job queue" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"

# Runtime-only system packages: cv2/Pillow image libs, curl+jq for probes, libpq5 for psycopg.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    curl \
    jq \
    libpq5 \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# No package manager in the runtime image (Trivy flags the vendored libraries of even the
# newest pip, and nothing here installs packages at run time). The builder's user-site copy is
# stripped in the builder stage.
RUN rm -rf /usr/local/lib/python3.12/site-packages/pip \
        /usr/local/lib/python3.12/site-packages/pip-*.dist-info \
        /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.12

# Non-root user, uid 1000 with the video group -- same as this repo's other
# images so bind-mounted volumes keep consistent ownership on the host.
RUN groupadd -r appuser \
    && useradd -r -g appuser -G video -u 1000 -m -s /bin/bash appuser \
    && mkdir -p /app /var/lib/openprocessor/training_runs \
                /home/appuser/.cache/torch /home/appuser/.config/Ultralytics \
    && chown -R appuser:appuser /app /var/lib/openprocessor /home/appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TORCH_HOME=/home/appuser/.cache/torch \
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics \
    OP_TRAIN_RUNS_ROOT=/var/lib/openprocessor/training_runs

# Flat module layout: /app is on PYTHONPATH, so `celery -A celery_tasks:app` resolves
# celery_tasks.py and its siblings as top-level modules.
COPY --chown=appuser:appuser docker/trainer/*.py ./
# The job runtime and what it imports; no web, OpenSearch or service code
# (tests/trainer/test_runtime_import_minimal.py). `src` is a namespace package: no top __init__.
COPY --chown=appuser:appuser src/tasks/__init__.py ./src/tasks/__init__.py
COPY --chown=appuser:appuser src/tasks/runtime/ ./src/tasks/runtime/
COPY --chown=appuser:appuser src/clients/__init__.py ./src/clients/__init__.py
COPY --chown=appuser:appuser src/clients/storage/ ./src/clients/storage/
COPY --chown=appuser:appuser src/db/ ./src/db/
COPY --chown=appuser:appuser src/config/ ./src/config/
COPY --chown=appuser:appuser src/domain/ ./src/domain/
COPY --chown=appuser:appuser src/utils/ ./src/utils/
# The augmentation preset catalog is shared with the API (which validates and
# serves it); copied flat so augment.py imports it as a top-level module.
COPY --chown=appuser:appuser src/services/training/augmentation_presets.py ./augmentation_presets.py

USER appuser

# Healthy while the worker answers a control ping on its own node name.
HEALTHCHECK --interval=60s --timeout=15s --start-period=60s --retries=3 \
    CMD ["sh", "-c", "celery -A celery_tasks:app inspect ping --timeout 10 -d \"celery@$(hostname)\" > /dev/null || exit 1"]

# Concurrency 1 and a fresh child per run: a run holds the card and releases its memory on exit.
ENTRYPOINT ["celery", "-A", "celery_tasks:app", "worker", "-Q", "gpu_train", "--pool=prefork", "--concurrency=1", "--max-tasks-per-child=1"]

# Build identity, last so a new commit rewrites only metadata and never a dependency layer.
# Read by job_manifest.build_lineage() as code_versions.trainer_sha (preferred over the API's
# submit-time trainer_image_revision observation, since this is the image actually running the
# code) -- see docker-compose.yml build.args.
ARG OP_BUILD_SHA=""
ENV OP_BUILD_SHA=${OP_BUILD_SHA}
LABEL org.opencontainers.image.revision=${OP_BUILD_SHA}

ARG ULTRALYTICS_SHA=8f5d355cd05b91503e7bf62429681dea4fa4b004
# Promoted from ARG to ENV so it is readable at *runtime*
# (`os.environ['ULTRALYTICS_SHA']`) and lands in each run's manifest under
# code_versions.ultralytics_sha. An ARG alone only exists at build time.
ENV ULTRALYTICS_SHA=${ULTRALYTICS_SHA}
