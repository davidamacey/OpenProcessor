"""Bulk-ingest defaults and bounds, served to clients by ``GET /ingest/bulk/config``.

The defaults come from the measured request-bound regime of the ingest path
(``docs/PERFORMANCE.md``, probe 1: 8 images per batch with 16 in flight was the
best row, throughput flat from 8 in flight on), so a run keeps the GPU fed
without holding more decoded images than the process admission gate allows.
"""

from __future__ import annotations

from src.config.env import env_str


DEFAULT_LOOPBACK_URL = 'http://127.0.0.1:8000'

DEFAULT_BATCH_SIZE = 16
DEFAULT_CONCURRENCY = 16
MAX_BATCH_SIZE = 64
MAX_CONCURRENCY = 32
# Files one claim takes from the queue (``FOR UPDATE SKIP LOCKED``).
CLAIM_BATCH = 500
DEFAULT_EXTENSIONS: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
# A run whose detector or datastore is down fails every batch; stop after this many
# in a row instead of recording the whole source as failed.
MAX_CONSECUTIVE_FAILED_BATCHES = 5
PROGRESS_WRITE_INTERVAL_S = 1.0
METRICS_SAMPLE_INTERVAL_S = 5.0
MAX_SENDER_CONNECTIONS = 64
# One batch of the largest size on a loaded host; the route answers when the whole batch is stored.
BATCH_REQUEST_TIMEOUT_S = 600.0
RATE_WINDOW_S = 30.0


def loopback_url() -> str:
    """Where a run sends its batches: the API's own address (``OP_API_LOOPBACK_URL``, the ``api``
    service name in a worker container)."""
    return env_str('OP_API_LOOPBACK_URL') or DEFAULT_LOOPBACK_URL
