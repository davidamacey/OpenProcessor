"""Dataset-import limits and directories, read from the environment at call
time (so a test can ``monkeypatch.setenv`` around one call)."""

from __future__ import annotations

import os
from pathlib import Path


def _positive_int(raw: str | None, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def imports_base_dir() -> Path:
    return Path(os.environ.get('OP_DATASET_IMPORTS_DIR', '/jobs/imports'))


def import_chunk_size() -> int:
    return _positive_int(os.environ.get('OP_DATASET_IMPORT_CHUNK'), 64)


def import_max_pending() -> int:
    return _positive_int(os.environ.get('OP_DATASET_IMPORT_MAX_PENDING'), 2000)


def import_max_failed_chunks() -> int:
    return _positive_int(os.environ.get('OP_DATASET_IMPORT_MAX_FAILED_CHUNKS'), 5)


def preview_max_files() -> int:
    return _positive_int(os.environ.get('OP_DATASET_PREVIEW_MAX_FILES'), 500_000)


def upload_max_bytes() -> int:
    return _positive_int(os.environ.get('OP_DATASET_UPLOAD_MAX_BYTES'), 2 * 1024**3)


def upload_max_files() -> int:
    return _positive_int(os.environ.get('OP_DATASET_UPLOAD_MAX_FILES'), 200_000)


def upload_ttl_hours() -> int:
    return _positive_int(os.environ.get('OP_DATASET_UPLOAD_TTL_H'), 72)


def reprocess_sync_max() -> int:
    return _positive_int(os.environ.get('OP_REPROCESS_SYNC_MAX'), 20)


def open_vocab_sweep_interval_s() -> int:
    """Seconds between sweeps of images left ``pending`` by the ingest-time
    open-vocabulary pass; ``0`` turns the sweeper off."""
    raw = os.environ.get('OP_OPEN_VOCAB_SWEEP_S')
    if raw is None or not raw.strip():
        return 120
    try:
        return max(0, int(raw))
    except ValueError:
        return 120


def open_vocab_concurrency() -> int:
    """Images of the full-image SAM 3 pass in flight at once (each fans out
    one segmenter call per target, so calls in flight <= this x targets)."""
    return _positive_int(os.environ.get('OP_OPEN_VOCAB_CONCURRENCY'), 4)


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
