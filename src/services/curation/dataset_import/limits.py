"""Dataset-import limits and directories, read from the environment at call
time (so a test can ``monkeypatch.setenv`` around one call)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.env import env_int, env_path


if TYPE_CHECKING:
    from pathlib import Path


def imports_base_dir() -> Path:
    return env_path('OP_DATASET_IMPORTS_DIR', '/jobs/imports')


def import_chunk_size() -> int:
    return env_int('OP_DATASET_IMPORT_CHUNK', 64, minimum=1)


def import_max_pending() -> int:
    return env_int('OP_DATASET_IMPORT_MAX_PENDING', 2000, minimum=1)


def import_max_failed_chunks() -> int:
    return env_int('OP_DATASET_IMPORT_MAX_FAILED_CHUNKS', 5, minimum=1)


def preview_max_files() -> int:
    return env_int('OP_DATASET_PREVIEW_MAX_FILES', 500_000, minimum=1)


def upload_max_bytes() -> int:
    return env_int('OP_DATASET_UPLOAD_MAX_BYTES', 2 * 1024**3, minimum=1)


def upload_max_files() -> int:
    return env_int('OP_DATASET_UPLOAD_MAX_FILES', 200_000, minimum=1)


def upload_ttl_hours() -> int:
    return env_int('OP_DATASET_UPLOAD_TTL_H', 72, minimum=1)


def reprocess_sync_max() -> int:
    return env_int('OP_REPROCESS_SYNC_MAX', 20, minimum=1)


def open_vocab_sweep_interval_s() -> int:
    """Seconds between sweeps of images left ``pending`` by the ingest-time
    open-vocabulary pass; ``0`` turns the sweeper off."""
    return env_int('OP_OPEN_VOCAB_SWEEP_S', 120, minimum=0)


def open_vocab_stale_after_s() -> int:
    """Age of an ``open_vocab_status: pending`` stamp after which the sweeper
    treats the drain as orphaned (an API restart) and reclaims it. Default 240 s
    = two sweep ticks: a live drain of a 64-image ingest batch (a few seconds per
    image at the default concurrency) finishes well inside it, and recovery after
    a restart takes 4 to 6 minutes instead of 10 to 12. Floor 30 s."""
    return max(30, env_int('OP_OPEN_VOCAB_STALE_S', 240, minimum=0))


def open_vocab_concurrency() -> int:
    """Images of the full-image SAM 3 pass in flight at once (each fans out
    one segmenter call per target, so calls in flight <= this x targets)."""
    return env_int('OP_OPEN_VOCAB_CONCURRENCY', 4, minimum=1)


# Per-file read caps for the small text files a dataset is described by. A
# label file, a ``data.yaml`` or an annotation JSON larger than these is
# refused, not parsed: the scan holds one in memory.
MAX_YAML_BYTES = 1 * 1024**2
MAX_LABEL_FILE_BYTES = 8 * 1024**2
# ``json`` parses the whole file into Python objects, several times its size, so
# this cap is what bounds memory per request (128 MiB is on the order of a
# million annotations).
MAX_COCO_JSON_BYTES = 128 * 1024**2
MAX_MANIFEST_BYTES = 16 * 1024**2
# A real ``data.yaml`` is a few thousand nodes and a few levels deep.
MAX_YAML_NODES = 50_000
MAX_YAML_DEPTH = 32


__all__ = [
    'MAX_COCO_JSON_BYTES',
    'MAX_LABEL_FILE_BYTES',
    'MAX_MANIFEST_BYTES',
    'MAX_YAML_BYTES',
    'MAX_YAML_DEPTH',
    'MAX_YAML_NODES',
    'import_chunk_size',
    'import_max_failed_chunks',
    'import_max_pending',
    'imports_base_dir',
    'preview_max_files',
    'reprocess_sync_max',
    'upload_max_bytes',
    'upload_max_files',
    'upload_ttl_hours',
]
