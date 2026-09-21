# =============================================================================
# Curation bake-off evaluator — on-demand detector comparison container.
# =============================================================================
# Runs scripts/curation/bakeoff/bakeoff_runner.py, the entrypoint for the
# ~4,000-LOC bake-off harness (src/routers/curation/bakeoff.py drops job
# specs; this container scores each candidate detector against a frozen
# test split and writes matrix.json/comparison.json for the review UI).
#
# Standalone image (no shared trainer base image exists in this repo yet
# — see docs/design/oss_main_completion_plan.md Wave 4 / D6): installs the
# same detection-stack pins as the core API image (requirements.txt) plus
# the harness-only extras (scripts/curation/bakeoff/requirements.txt).
#
# Build:
#   docker compose --profile curation build curation-evaluator
#
# Run on demand (watch mode, matches OP_BAKEOFF_JOBS_DIR in env.template):
#   docker compose --profile curation run --rm curation-evaluator
# One-shot job:
#   docker compose --profile curation run --rm curation-evaluator \
#     python -m scripts.curation.bakeoff.bakeoff_runner --job /eval_jobs/x.job.json
# =============================================================================

FROM python:3.13-slim-trixie

LABEL org.opencontainers.image.title="OpenProcessor Curation Evaluator" \
      org.opencontainers.image.description="On-demand detector bake-off harness runner" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"

# hadolint ignore=DL3008
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    curl \
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

USER appuser

# Default job-watch directory — must match OP_BAKEOFF_JOBS_DIR
# (env.template) when the compose service mounts a different path.
ENV OP_BAKEOFF_JOBS_DIR=/var/lib/openprocessor/bakeoff_jobs
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD test -d "$OP_BAKEOFF_JOBS_DIR" || exit 1

ENTRYPOINT ["python", "-m", "scripts.curation.bakeoff.bakeoff_runner"]
CMD ["--watch", "/var/lib/openprocessor/bakeoff_jobs"]
