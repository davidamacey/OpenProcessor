# =============================================================================
# OpenProcessor images built from the repository root
# =============================================================================
# One build graph, three published targets that share the same runtime layers:
#
#   api        FastAPI service and the curation workers (the LAST stage, so an untargeted
#              `docker build .` still produces the API)
#   export     the engine-build toolchain: TensorRT, ModelOpt and the
#              model-repo seed. `openprocessor models install` and `export` run in it.
#   evaluator  the on-demand bake-off runner
#
#   base ----------------------------> runtime-base --+--> export    (+ builder-export delta)
#   builder -> builder-export (delta)                  +--> evaluator (+ builder-eval delta)
#           \-> builder-eval  (delta)                  +--> api
#
# export and evaluator add only their own packages as a file-level delta over the runtime
# site-packages (docker/buildlib/site_delta.sh), so a host that pulls all three downloads the
# runtime set once. OP_BUILD_SHA is declared at the very end of each target: a commit changes
# only the last metadata layer, never a multi-GB layer's cache key.
#
# VOLUME MOUNTS (for development):
#   - ./src:/app/src              (hot reload)
#   - ./pytorch_models:/app/pytorch_models
#   - ./VERSION:/app/VERSION:ro
# =============================================================================

# -----------------------------------------------------------------------------
# base: the Debian runtime shared by every target
# -----------------------------------------------------------------------------
FROM python:3.13-slim-trixie@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f AS base

# Runtime-only system packages (no build tools); upgrade first for Debian point-release
# security fixes.
# procps -> pgrep, used by the curation worker services' healthchecks
# (docker-compose.yml `profiles: [curation]`) -- the api image is shared with those workers.
# postgresql-client -> pg_dump, used by the periodic worker's db_backup tick (src/services/periodic/db_backup.py)
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    curl \
    jq \
    procps \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    libpq5 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# pg_dump must be the server's major version (it refuses a newer server): Debian trixie ships 17,
# the compose Postgres is 18, so the client comes from the PGDG apt repository
# (https://www.postgresql.org/download/linux/debian/; apt reads the .asc key directly).
RUN curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc -o /usr/share/keyrings/pgdg.asc \
    && echo "deb [signed-by=/usr/share/keyrings/pgdg.asc] https://apt.postgresql.org/pub/repos/apt trixie-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client-18 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# No package manager in the runtime images: even the newest pip vendors
# msgpack / setuptools / urllib3 builds that Trivy flags (HIGH), and nothing
# here installs packages at run time. Removed from the base image's system
# site-packages; the builder's user site never contains one.
RUN rm -rf /usr/local/lib/python3.13/site-packages/pip \
        /usr/local/lib/python3.13/site-packages/pip-*.dist-info \
        /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.13

# Non-root user with video group for GPU access
RUN groupadd -r appuser && \
    useradd -r -g appuser -G video -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app /app/logs && \
    chown -R appuser:appuser /app && \
    mkdir -p /home/appuser/.cache/huggingface \
             /home/appuser/.cache/torch && \
    chown -R appuser:appuser /home/appuser/.cache && \
    # CurationConfig.state_dir default, created + owned by appuser here so a fresh
    # named Docker volume mounted over it on first `up` inherits appuser ownership
    # (Docker copies an empty named volume's ownership from the image directory).
    mkdir -p /var/lib/openprocessor && \
    chown -R appuser:appuser /var/lib/openprocessor

WORKDIR /app

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TORCH_HOME=/home/appuser/.cache/torch

# -----------------------------------------------------------------------------
# builder: the runtime dependency set, with compilers (never reaches an image)
# -----------------------------------------------------------------------------
FROM python:3.13-slim-trixie@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /build

# Build dependencies (isolated to this stage). apt-get upgrade pulls Debian point-release
# security fixes the base tag may lag behind.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    make \
    git \
    pkg-config \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --user --no-cache-dir --no-warn-script-location \
        --extra-index-url https://pypi.nvidia.com \
        -r requirements.txt \
    && pip install --user --no-cache-dir --no-warn-script-location --no-deps \
        "perception_models @ git+https://github.com/facebookresearch/perception_models.git@3e352cca660658d4b5c90f42a7808b11469e4c66"

# The exact runtime versions, as pip constraints for the extra dependency sets below. The
# direct-URL line cannot be a constraint; it is already installed.
RUN pip freeze --user | grep -v '^perception' | LC_ALL=C sort > /tmp/runtime.freeze \
    && rm -rf /root/.local/lib/python3.13/site-packages/pip \
        /root/.local/lib/python3.13/site-packages/pip-*.dist-info \
        /root/.local/bin/pip /root/.local/bin/pip3 /root/.local/bin/pip3.13

COPY docker/buildlib/ /usr/local/share/op-build/

# -----------------------------------------------------------------------------
# builder-export: TensorRT + ModelOpt on top of the runtime set
# -----------------------------------------------------------------------------
FROM builder AS builder-export

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY requirements-export.txt ./

# Under the runtime freeze nothing in the runtime set can move; the superset check and the
# delta script fail the build if it did anyway.
RUN bash /usr/local/share/op-build/site_delta.sh snapshot /root/.local /tmp/pre.sha \
    && pip install --user --no-cache-dir --no-warn-script-location \
        -c /tmp/runtime.freeze \
        --extra-index-url https://pypi.nvidia.com \
        -r requirements-export.txt \
    && pip freeze --user | grep -v '^perception' | LC_ALL=C sort > /tmp/export.freeze \
    && missing="$(LC_ALL=C comm -23 /tmp/runtime.freeze /tmp/export.freeze)" \
    && { [ -z "$missing" ] || { echo "runtime packages changed:" "$missing" >&2; exit 1; }; } \
    && rm -rf /root/.local/lib/python3.13/site-packages/pip \
        /root/.local/lib/python3.13/site-packages/pip-*.dist-info \
        /root/.local/bin/pip /root/.local/bin/pip3 /root/.local/bin/pip3.13 \
    && bash /usr/local/share/op-build/site_delta.sh delta /root/.local /tmp/pre.sha /delta/export

# -----------------------------------------------------------------------------
# builder-eval: the bake-off harness extras on top of the runtime set
# -----------------------------------------------------------------------------
FROM builder AS builder-eval

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY scripts/curation/bakeoff/requirements.txt bakeoff-requirements.txt

RUN bash /usr/local/share/op-build/site_delta.sh snapshot /root/.local /tmp/pre.sha \
    && pip install --user --no-cache-dir --no-warn-script-location \
        -c /tmp/runtime.freeze \
        -r bakeoff-requirements.txt \
    && pip freeze --user | grep -v '^perception' | LC_ALL=C sort > /tmp/eval.freeze \
    && missing="$(LC_ALL=C comm -23 /tmp/runtime.freeze /tmp/eval.freeze)" \
    && { [ -z "$missing" ] || { echo "runtime packages changed:" "$missing" >&2; exit 1; }; } \
    && rm -rf /root/.local/lib/python3.13/site-packages/pip \
        /root/.local/lib/python3.13/site-packages/pip-*.dist-info \
        /root/.local/bin/pip /root/.local/bin/pip3 /root/.local/bin/pip3.13 \
    && bash /usr/local/share/op-build/site_delta.sh delta /root/.local /tmp/pre.sha /delta/eval

# -----------------------------------------------------------------------------
# runtime-base: base + the runtime site-packages (the layers all three targets share)
# -----------------------------------------------------------------------------
FROM base AS runtime-base

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# -----------------------------------------------------------------------------
# export: the engine-build toolchain (compose service model-export, profile setup)
# -----------------------------------------------------------------------------
FROM runtime-base AS export

COPY --from=builder-export --chown=appuser:appuser /delta/export/ /home/appuser/.local/

COPY --chown=appuser:appuser src/ ./src/
# What `openprocessor models install` and `sample coco` run in this image: the export scripts,
# the public COCO sample fetcher and the shell libraries model_setup.sh sources (config.sh and
# what it sources, download.sh). tests/test_dockerfile_hygiene.py checks this set against the
# imports.
COPY --chown=appuser:appuser export/ ./export/
COPY --chown=appuser:appuser scripts/datasets/ ./scripts/datasets/
COPY --chown=appuser:appuser scripts/lib/colors.sh scripts/lib/config.sh scripts/lib/config_models.sh \
     scripts/lib/download.sh scripts/lib/env_generator.sh scripts/lib/gpu.sh \
     scripts/lib/opensearch_heap.sh ./scripts/lib/
COPY --chown=appuser:appuser VERSION ./VERSION
COPY --chown=appuser:appuser config_templates/ ./config_templates/
# Model-repo seed: Triton needs config.pbtxt/labels.txt/OCR dicts/ocr_pipeline's model.py
# before any engine is exported. The release builds from the committed tree and .dockerignore
# excludes the large *.plan/*.onnx/*.pt/*.engine payloads, so this ships only the small tracked
# config tree. Seeded into ./models on first use by sync_model_seed.py.
COPY --chown=appuser:appuser models/ /opt/openprocessor/model_repo_seed/

USER appuser

CMD ["python", "-m", "export.preflight"]

LABEL org.opencontainers.image.title="OpenProcessor Model Export" \
      org.opencontainers.image.description="TensorRT engine build and model export toolchain" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.authors="OpenProcessor Contributors" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor" \
      org.opencontainers.image.documentation="https://github.com/davidamacey/OpenProcessor/blob/main/README.md"
# Build identity: the commit this image was built from (docker-compose.yml passes it as a
# build arg). Last, so a new commit rewrites only metadata. A runtime OP_BUILD_SHA still
# overrides -- see env.template.
ARG OP_BUILD_SHA=""
ENV OP_BUILD_SHA=${OP_BUILD_SHA}
LABEL org.opencontainers.image.revision=${OP_BUILD_SHA}

# -----------------------------------------------------------------------------
# evaluator: on-demand detector comparison (scripts/curation/bakeoff/bakeoff_runner.py)
# -----------------------------------------------------------------------------
# The consumer of queue gpu_eval: it scores each candidate detector of a bake-off job against a
# frozen test split and writes matrix.json/comparison.json for the review UI.
FROM runtime-base AS evaluator

COPY --from=builder-eval --chown=appuser:appuser /delta/eval/ /home/appuser/.local/

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/curation/ ./scripts/curation/
# scripts/curation/bakeoff/profile.py reads opt-in baseline profiles under the repo's examples/
# tree (path given explicitly via a job's baselines_path) -- bake it in so a non-checkout
# deploy doesn't need a bind mount just to use that opt-in feature.
COPY --chown=appuser:appuser examples/ ./examples/

USER appuser

ENV PYTHONPATH=/app
HEALTHCHECK --interval=60s --timeout=15s --start-period=60s --retries=3 \
    CMD ["sh", "-c", "celery -A scripts.curation.bakeoff.bakeoff_tasks inspect ping --timeout 10 -d \"celery@$(hostname)\" > /dev/null || exit 1"]

ENTRYPOINT ["celery", "-A", "scripts.curation.bakeoff.bakeoff_tasks", "worker", "-Q", "gpu_eval", "--pool=prefork", "--concurrency=1", "--max-tasks-per-child=5"]

LABEL org.opencontainers.image.title="OpenProcessor Curation Evaluator" \
      org.opencontainers.image.description="On-demand detector bake-off harness runner" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"
# Build identity: see the export target.
ARG OP_BUILD_SHA=""
ENV OP_BUILD_SHA=${OP_BUILD_SHA}
LABEL org.opencontainers.image.revision=${OP_BUILD_SHA}

# -----------------------------------------------------------------------------
# api: the FastAPI service and the curation workers (keep this the LAST stage)
# -----------------------------------------------------------------------------
# docker-compose.yml overrides CMD with environment-specific worker counts.
FROM runtime-base AS api

COPY --chown=appuser:appuser src/ ./src/
# Only what runs inside the container: the curation workers and the CLIs operators run with
# `docker compose exec api python -m scripts.curation...`, the public COCO sample fetcher, and
# the shell libraries the API container sources (config.sh and what it sources, download.sh).
# tests/test_dockerfile_hygiene.py checks this set against the imports.
COPY --chown=appuser:appuser scripts/curation/ ./scripts/curation/
# Run by compose services of this image: db-migrate (scripts.db.migrate reads alembic.ini),
# the Celery pool healthcheck and the Flower config. tests/test_dockerfile_hygiene.py derives
# this set from docker-compose.yml.
COPY --chown=appuser:appuser scripts/db/ ./scripts/db/
COPY --chown=appuser:appuser scripts/healthcheck/ ./scripts/healthcheck/
COPY --chown=appuser:appuser scripts/flower/ ./scripts/flower/
COPY --chown=appuser:appuser alembic.ini ./alembic.ini
COPY --chown=appuser:appuser scripts/datasets/ ./scripts/datasets/
COPY --chown=appuser:appuser scripts/lib/colors.sh scripts/lib/config.sh scripts/lib/config_models.sh \
     scripts/lib/download.sh scripts/lib/env_generator.sh scripts/lib/gpu.sh \
     scripts/lib/opensearch_heap.sh ./scripts/lib/
COPY --chown=appuser:appuser VERSION ./VERSION
COPY --chown=appuser:appuser config_templates/ ./config_templates/
# docker-compose.yml is deploy-safe (no bind mount of ./examples), so the examples are baked in
# or every OP_REGION_PROFILE_PATH example under /app/examples 404s in an installed
# (non-checkout) deployment. docker-compose.dev.yml still bind-mounts it for hot-reload.
COPY --chown=appuser:appuser examples/ ./examples/

USER appuser

EXPOSE 8000

# /live = process liveness only. /health now probes dependencies (Triton,
# OpenSearch) and returns 503 while any is down -- gating the container's
# health on it would mark api unhealthy (and cascade via
# depends_on: service_healthy) whenever a *downstream* dep degrades.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["sh", "-c", "curl -f http://localhost:8000/live || exit 1"]

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

LABEL org.opencontainers.image.title="OpenProcessor FastAPI Service" \
      org.opencontainers.image.description="Visual AI API with object detection, face recognition, embeddings, and OCR" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.authors="OpenProcessor Contributors" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor" \
      org.opencontainers.image.documentation="https://github.com/davidamacey/OpenProcessor/blob/main/README.md"
# Build identity: the commit this image was built from, baked in so every manifest's
# code_versions.api_sha reflects the running image rather than a dev checkout's live
# `git rev-parse` fallback (docker-compose.yml passes this as a build arg; empty stays
# "unknown" via _code_sha()). A runtime OP_BUILD_SHA still overrides -- see env.template.
ARG OP_BUILD_SHA=""
ENV OP_BUILD_SHA=${OP_BUILD_SHA}
LABEL org.opencontainers.image.revision=${OP_BUILD_SHA}
