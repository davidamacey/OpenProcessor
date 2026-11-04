"""
Model Export Service (API side).

The API image carries no TensorRT, so ``POST /models/upload`` and
``POST /models/{name}/export`` do not run the export themselves. They validate the model,
write a task, and queue a job file for the ``model-export-worker`` service (the export image,
``export/export_worker.py``), which runs the export, loads the engine into Triton and writes
the task status back. The protocol, under the export task dir (``EXPORT_TASK_DIR``, default
``/jobs/model_export``, the shared jobs volume):

- ``queue/<task_id>.job.json``    written here (atomic temp+rename), claimed by the worker
- ``claimed/<task_id>.job.json``  the worker's claim (atomic rename out of ``queue/``)
- ``<task_id>.json``              the task status the status routes serve, written by both sides
- ``worker.heartbeat``            touched by the worker every few seconds

Fail closed: a missing or stale heartbeat means no worker will run the job, so the routes
refuse with :class:`ExportWorkerUnavailableError` (HTTP 503 naming the service) before any task or
file is created. The task file is the source of truth; ``export_tasks`` is only a read-through
cache that every read refreshes from disk. Predates the curation subsystem and isn't
curation-specific, so its directory setting lives on the base ``Settings`` class
(``export_task_dir``) rather than ``CurationConfig``.
"""

import json
import logging
import re
import shutil
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.env import env_path
from src.schemas.models import ExportStatus
from src.utils.atomic_write import atomic_write_json


logger = logging.getLogger(__name__)

# Paths
PYTORCH_MODELS_DIR = Path('/app/pytorch_models')
TRITON_MODELS_DIR = Path('/app/models')

WORKER_SERVICE = 'model-export-worker'
QUEUE_SUBDIR = 'queue'
CLAIMED_SUBDIR = 'claimed'
HEARTBEAT_FILE = 'worker.heartbeat'
# The worker touches its heartbeat every 2 s from a thread, so even a multi-minute engine build
# keeps it fresh; 15 s tolerates a loaded host.
HEARTBEAT_MAX_AGE_S = 15.0

# Read-through in-memory cache; the on-disk file per task id is the source of truth (see module
# docstring) and every read refreshes this dict from it.
export_tasks: dict[str, dict[str, Any]] = {}


def _task_dir() -> Path:
    """Resolved fresh each call (env var wins over ``Settings``) so tests
    can override with ``monkeypatch.setenv`` + ``tmp_path`` without
    needing to bust ``get_settings()``'s ``lru_cache`` — same convention
    the curation job modules use for their own state directories."""
    from src.config.settings import get_settings

    return env_path('EXPORT_TASK_DIR', get_settings().export_task_dir)


def _task_file(task_id: str) -> Path:
    return _task_dir() / f'{task_id}.json'


def _ensure_task_dir() -> None:
    _task_dir().mkdir(parents=True, exist_ok=True)


def persist_task(task_id: str) -> None:
    """Atomically write ``export_tasks[task_id]`` to disk. No-op if the
    task isn't in the in-memory cache (shouldn't happen for any caller
    in this module, but keeps this safe to call defensively)."""
    task = export_tasks.get(task_id)
    if task is None:
        return
    _ensure_task_dir()
    atomic_write_json(_task_file(task_id), task)


def _load_task_file(path: Path) -> dict[str, Any] | None:
    try:
        typed: dict[str, Any] | None = json.loads(path.read_text())
        return typed
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning('export task file unreadable: path=%s error=%s', path, exc)
        return None


def load_tasks_from_disk() -> int:
    """Rebuild the in-memory ``export_tasks`` cache from
    ``<export_task_dir>/*.json``. Simulates what a fresh process sees on
    startup — the file is the source of truth, the dict is a cache.

    Returns the number of tasks loaded. Safe to call when the directory
    doesn't exist yet (a fresh deployment that has never exported a
    model) — returns 0 rather than raising.
    """
    directory = _task_dir()
    if not directory.is_dir():
        return 0
    loaded = 0
    for path in sorted(directory.glob('*.json')):
        task = _load_task_file(path)
        if task is None or 'task_id' not in task:
            continue
        export_tasks[task['task_id']] = task
        loaded += 1
    return loaded


def validate_pytorch_model(file_path: Path) -> tuple[bool, str, dict[str, Any]]:
    """
    Validate uploaded file is a valid YOLO11 detection model.

    Returns:
        Tuple of (is_valid, error_message, model_info)
    """
    try:
        import torch
        from ultralytics.nn.tasks import _SafeLoad, torch_safe_load

        # torch_safe_load quietly drops safe_only when this torch cannot restrict
        # unpickling; refuse instead of falling back to an unrestricted load.
        if not _SafeLoad.SUPPORTED:
            return False, 'Restricted model loading is unavailable in this build', {}

        # Check file size
        file_size = file_path.stat().st_size
        if file_size < 1000:
            return False, 'File too small to be a valid model', {}
        if file_size > 500 * 1024 * 1024:
            return False, 'File too large (max 500MB)', {}

        # Restricted load: torch.load(weights_only=True) plus the Ultralytics
        # allow-list of model classes, requested per call (not via the
        # ULTRALYTICS_SAFE_LOAD env var alone). Anything else in the file is refused.
        ckpt, _ = torch_safe_load(file_path, safe_only=True)
        model = ckpt.get('model') if isinstance(ckpt, dict) else None
        if model is None or not hasattr(model, 'names'):
            return False, 'Not an Ultralytics model checkpoint', {}
        names = model.names

        model_info = {
            'task': getattr(model, 'task', 'detect'),
            'num_classes': len(names),
            'class_names': list(names.values()) if isinstance(names, dict) else [],
            # End-to-end (NMS-free) architectures — YOLO26 — export through
            # the native toolchain; classic YOLO11-family models go through
            # the EfficientNMS toolchain. run_export() routes on this flag.
            'end2end': bool(getattr(model, 'end2end', False)),
        }

        if model_info['task'] != 'detect':
            return False, f'Only detection models supported, got: {model_info["task"]}', model_info

        del model
        torch.cuda.empty_cache()

        return True, 'Valid YOLO detection model', model_info

    except Exception as e:
        return False, f'Validation error: {e}', {}


TRITON_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9_]{0,63}$')
"""The name becomes a file under pytorch_models/, a model directory and the worker's
``pt:name:batch`` argument, so it is a slug or nothing."""


def check_triton_name(name: str) -> str:
    """Return ``name`` if it is a Triton model slug, else raise ValueError."""
    if not TRITON_NAME_RE.fullmatch(name):
        msg = (
            f'Invalid model name {name!r}: use 1-64 lowercase letters, digits or underscores, '
            'starting with a letter or digit'
        )
        raise ValueError(msg)
    return name


def generate_triton_name(filename: str, custom_name: str | None = None) -> str:
    """Triton model name from ``custom_name`` or the file name, normalised then checked."""
    if custom_name:
        return check_triton_name(custom_name.lower().replace('-', '_').replace(' ', '_'))

    stem = Path(filename).stem.lower().replace('-', '_').replace(' ', '_')

    # Remove common suffixes
    for suffix in ['_best', '_last', '_final']:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return check_triton_name(stem)


def create_export_task(
    filename: str,
    triton_name: str,
    model_info: dict[str, Any],
    formats: list[str],
) -> str:
    """Create a new export task and return task_id."""
    task_id = str(uuid.uuid4())[:8]
    now = datetime.now(UTC)

    export_tasks[task_id] = {
        'task_id': task_id,
        'model_name': Path(filename).stem,
        'triton_name': triton_name,
        'status': ExportStatus.PENDING,
        'progress': 0.0,
        'current_step': 'Queued for export',
        'message': 'Export starting...',
        'created_at': now,
        'started_at': None,
        'completed_at': None,
        'error': None,
        'formats_completed': [],
        'formats_pending': formats,
        'num_classes': model_info.get('num_classes'),
        'class_names': model_info.get('class_names'),
        'end2end': model_info.get('end2end', False),
        'triton_loaded': False,
    }
    persist_task(task_id)

    return task_id


def get_export_task(task_id: str) -> dict[str, Any] | None:
    """Get an export task by ID, always from its file: the worker process updates it."""
    task = _load_task_file(_task_file(task_id))
    if task is not None:
        export_tasks[task_id] = task
    return task


def list_export_tasks() -> list[dict[str, Any]]:
    """List all export tasks, read-through from disk.

    Reconciles the in-memory cache against the directory first so a
    task written by an earlier process (or another worker) shows up
    without requiring an explicit :func:`load_tasks_from_disk` call.
    """
    load_tasks_from_disk()
    return list(export_tasks.values())


# --- worker hand-off ------------------------------------------------------------------------


class ExportWorkerUnavailableError(RuntimeError):
    """No model-export worker is running, so a queued job would never be picked up."""


def queue_dir() -> Path:
    return _task_dir() / QUEUE_SUBDIR


def claimed_dir() -> Path:
    return _task_dir() / CLAIMED_SUBDIR


def heartbeat_path() -> Path:
    return _task_dir() / HEARTBEAT_FILE


def export_worker_alive() -> bool:
    """True when the worker's heartbeat file was touched within ``HEARTBEAT_MAX_AGE_S``."""
    try:
        age = time.time() - heartbeat_path().stat().st_mtime
    except OSError:
        return False
    return age <= HEARTBEAT_MAX_AGE_S


def require_export_worker() -> None:
    if not export_worker_alive():
        raise ExportWorkerUnavailableError(
            f'The model export worker is not running: the {WORKER_SERVICE} service has not '
            'reported in. Engine builds run in the export image, not in the API. Start it with: '
            f'docker compose --profile export up -d {WORKER_SERVICE}'
        )


def enqueue_export_job(
    task_id: str,
    *,
    pt_file: Path,
    triton_name: str,
    max_batch: int,
    formats: list[str],
    normalize_boxes: bool,
    auto_load: bool,
) -> None:
    """Queue the job file for the worker (atomic temp+rename into ``queue/``)."""
    check_triton_name(triton_name)
    task = export_tasks[task_id]
    job = {
        'task_id': task_id,
        'pt_file': str(pt_file),
        'triton_name': triton_name,
        'max_batch': max_batch,
        'formats': formats,
        'normalize_boxes': normalize_boxes,
        'auto_load': auto_load,
        'end2end': bool(task.get('end2end', False)),
    }
    queue = queue_dir()
    queue.mkdir(parents=True, exist_ok=True)
    final = queue / f'{task_id}.job.json'
    atomic_write_json(final, job, default=None)


async def save_uploaded_file(
    content: bytes,
    triton_name: str,
) -> tuple[Path, dict[str, Any]]:
    """
    Save uploaded file and validate it.

    Returns:
        Tuple of (saved_path, model_info)

    Raises:
        ValueError: If validation fails
    """
    PYTORCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save to temp file first for validation
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        is_valid, error_msg, model_info = validate_pytorch_model(tmp_path)
        if not is_valid:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(error_msg)

        # Move to final location
        final_path = PYTORCH_MODELS_DIR / f'{triton_name}.pt'
        shutil.move(tmp_path, final_path)

        return final_path, model_info

    except ValueError:
        raise
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f'Failed to save file: {e}') from e
