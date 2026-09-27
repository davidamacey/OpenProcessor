# =============================================================================
# Curation bake-off evaluator — on-demand detector comparison container.
# =============================================================================
# Runs scripts/curation/bakeoff/bakeoff_runner.py, the entrypoint for the
# ~4,000-LOC bake-off harness (src/routers/curation/bakeoff.py drops job
# specs; this container scores each candidate detector against a frozen
# test split and writes matrix.json/comparison.json for the review UI).
#
# Standalone image (no shared trainer base image exists in this repo):
# installs the same detection-stack pins as the core API image
# (requirements.txt) plus the harness-only extras
# (scripts/curation/bakeoff/requirements.txt).
#
# Build:
#   docker compose --profile curation build curation-evaluator
#
# Run on demand (watch mode; scans every projects/*/bakeoff_jobs under the state dir):
#   docker compose --profile curation run --rm curation-evaluator
# One-shot job:
#   docker compose --profile curation run --rm curation-evaluator \
#     python -m scripts.curation.bakeoff.bakeoff_runner --job /eval_jobs/x.job.json
#
# WAIVER (installer plan §3.4 release checklist, Wave 0): still a single
# `FROM`, not multi-stage. build-essential/cmake/pkg-config stay in the
# final image to compile wheel deps at `pip install` time, which the
# other images' builder-stage split avoids. Deferred: Wave 0 scope is the
# compose/mount split, not image-size hardening, and this image is
# on-demand only (`docker compose ... run --rm`), never a long-running
# service. Track the multi-stage conversion as release-checklist debt
# (§3.4) rather than doing it silently here.
# =============================================================================

FROM python:3.13-slim-trixie

LABEL org.opencontainers.image.title="OpenProcessor Curation Evaluator" \
      org.opencontainers.image.description="On-demand detector bake-off harness runner" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"

# Build identity: the commit this evaluator image was built from -- see
# docker-compose.yml build.args and the same block in the root Dockerfile.
ARG OP_BUILD_SHA=""
ENV OP_BUILD_SHA=${OP_BUILD_SHA}
LABEL org.opencontainers.image.revision=${OP_BUILD_SHA}

# hadolint ignore=DL3008
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    pkg-config \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r appuser && \
    useradd -r -g appuser -G video -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser/.cache

WORKDIR /app

COPY requirements.txt requirements.txt
COPY scripts/curation/bakeoff/requirements.txt bakeoff-requirements.txt

# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
        -r requirements.txt \
    && pip install --no-cache-dir -r bakeoff-requirements.txt \
    && pip install --no-cache-dir mlflow

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/home/appuser/.cache/huggingface \
    TORCH_HOME=/home/appuser/.cache/torch

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
# scripts/curation/bakeoff/profile.py reads opt-in baseline profiles under
# the repo's examples/ tree (path given explicitly via a job's
# baselines_path) -- bake it in so a non-checkout deploy doesn't need a
# bind mount just to use that opt-in feature. Not mounted separately in
# docker-compose.yml today, matching current behavior.
COPY --chown=appuser:appuser examples/ ./examples/

USER appuser

# Watch root; the runner also scans the sibling projects/*/bakeoff_jobs
# dirs the API enqueues into.
ENV OP_BAKEOFF_JOBS_DIR=/var/lib/openprocessor/bakeoff_jobs
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD test -d "$OP_BAKEOFF_JOBS_DIR" || exit 1

ENTRYPOINT ["python", "-m", "scripts.curation.bakeoff.bakeoff_runner"]
CMD ["--watch", "/var/lib/openprocessor/bakeoff_jobs"]
