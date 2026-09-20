"""
Model Export Service.

Handles YOLO model validation, export task management, and TensorRT export execution.

Task-state durability: each task's dict is also written to
``<export_task_dir>/<task_id>.json`` via the same atomic temp+rename
pattern the curation job modules use (see
``src.services.curation.job_reconcile``). ``export_tasks`` remains a
plain in-memory dict — it's the hot read path for the status endpoints —
but the file is now the source of truth: :func:`load_tasks_from_disk`
rebuilds it from the directory, and :func:`reconcile_orphaned_export_tasks`
(called once at startup) repairs any task left in a non-terminal status
by a process that no longer exists. Predates the curation subsystem and
isn't curation-specific, so its directory setting lives on the base
``Settings`` class (``export_task_dir``) rather than ``CurationConfig``.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.schemas.models import ExportStatus
from src.services.triton_control import TritonControlService


logger = logging.getLogger(__name__)

# Paths
PYTORCH_MODELS_DIR = Path('/app/pytorch_models')
TRITON_MODELS_DIR = Path('/app/models')
# Legacy YOLO11-family toolchain (EfficientNMS; re-execs into /opt/venv-y11)
EXPORT_SCRIPT = Path('/app/export/export_models.py')
# Native NMS-free toolchain for end2end architectures (YOLO26)
EXPORT_SCRIPT_Y26 = Path('/app/export/export_yolo26.py')

# Read-through in-memory cache. The on-disk file per task id is the
# source of truth (see module docstring) — this dict just avoids a
# directory read on every status poll within a single process lifetime.
export_tasks: dict[str, dict[str, Any]] = {}


def _task_dir() -> Path:
    """Resolved fresh each call (env var wins over ``Settings``) so tests
    can override with ``monkeypatch.setenv`` + ``tmp_path`` without
    needing to bust ``get_settings()``'s ``lru_cache`` — same convention
    the curation job modules use for their own state directories."""
    from src.config.settings import get_settings

    return Path(os.environ.get('EXPORT_TASK_DIR', str(get_settings().export_task_dir)))


def _task_file(task_id: str) -> Path:
    return _task_dir() / f'{task_id}.json'


def _ensure_task_dir() -> None:
    _task_dir().mkdir(parents=True, exist_ok=True)


def _persist_task(task_id: str) -> None:
    """Atomically write ``export_tasks[task_id]`` to disk. No-op if the
    task isn't in the in-memory cache (shouldn't happen for any caller
    in this module, but keeps this safe to call defensively)."""
    task = export_tasks.get(task_id)
    if task is None:
        return
    _ensure_task_dir()
    tmp = _task_file(task_id).with_suffix('.tmp')
    tmp.write_text(json.dumps(task, default=str))
    tmp.replace(_task_file(task_id))


def _load_task_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
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


def reconcile_orphaned_export_tasks() -> int:
    """Startup-only repair: any task file left in a non-terminal status
    (``PENDING``/``VALIDATING``/``EXPORTING``/``LOADING``) was written by
    a process that no longer exists — unlike the curation job modules,
    export tasks run as a ``BackgroundTasks`` coroutine inline in this
    same process with no separate worker to still be alive, so at
    startup (before any new task is created) a non-terminal status on
    disk is unconditionally orphaned. No heartbeat file exists for this
    task type, so there is nothing to disambiguate — this is simpler
    than the curation jobs' heartbeat-staleness check by necessity.

    Marks each as :attr:`ExportStatus.FAILED` (there is no dedicated
    'interrupted' member on :class:`ExportStatus` — it's a frozen wire
    enum shared with the frontend via ``ExportTaskResponse``/
    ``ExportTaskStatus``, so reusing the existing terminal value avoids
    widening that contract for this fix). Returns the number reconciled.
    """
    load_tasks_from_disk()
    non_terminal = {
        ExportStatus.PENDING,
        ExportStatus.VALIDATING,
        ExportStatus.EXPORTING,
        ExportStatus.LOADING,
    }
    reconciled = 0
    for task_id, task in list(export_tasks.items()):
        if task.get('status') not in non_terminal:
            continue
        task['status'] = ExportStatus.FAILED
        task['error'] = task.get('error') or 'export task interrupted by a service restart'
        task['message'] = f'Export failed: {task["error"]}'
        task['completed_at'] = task.get('completed_at') or datetime.now(UTC)
        _persist_task(task_id)
        reconciled += 1
    return reconciled


def validate_pytorch_model(file_path: Path) -> tuple[bool, str, dict[str, Any]]:
    """
    Validate uploaded file is a valid YOLO11 detection model.

    Returns:
        Tuple of (is_valid, error_message, model_info)
    """
    try:
        import torch
        from ultralytics import YOLO

        # Check file size
        file_size = file_path.stat().st_size
        if file_size < 1000:
            return False, 'File too small to be a valid model', {}
        if file_size > 500 * 1024 * 1024:
            return False, 'File too large (max 500MB)', {}

        # Load and validate model
        model = YOLO(str(file_path))

        model_info = {
            'task': getattr(model, 'task', 'detect'),
            'num_classes': len(model.names) if hasattr(model, 'names') else 0,
            'class_names': list(model.names.values()) if isinstance(model.names, dict) else [],
            # End-to-end (NMS-free) architectures — YOLO26 — export through
            # the native toolchain; classic YOLO11-family models go through
            # the EfficientNMS toolchain. run_export() routes on this flag.
            'end2end': bool(getattr(getattr(model, 'model', None), 'end2end', False)),
        }

        if model_info['task'] != 'detect':
            return False, f'Only detection models supported, got: {model_info["task"]}', model_info

        del model
        torch.cuda.empty_cache()

        return True, 'Valid YOLO detection model', model_info

    except Exception as e:
        return False, f'Validation error: {e}', {}


def generate_triton_name(filename: str, custom_name: str | None = None) -> str:
    """Generate Triton-compatible model name from filename."""
    if custom_name:
        return custom_name.lower().replace('-', '_').replace(' ', '_')

    stem = Path(filename).stem.lower().replace('-', '_').replace(' ', '_')

    # Remove common suffixes
    for suffix in ['_best', '_last', '_final']:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


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
    _persist_task(task_id)

    return task_id


def get_export_task(task_id: str) -> dict[str, Any] | None:
    """Get export task by ID.

    Read-through: falls back to the on-disk file (and repopulates the
    cache) on a cache miss, so a task created by a process that has
    since restarted is still queryable — the whole point of Gap 2.
    """
    task = export_tasks.get(task_id)
    if task is not None:
        return task
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


class StepTimer:
    """Context manager for tracking step timing in export pipeline."""

    def __init__(self, step_times: dict[str, float], step_name: str):
        self.step_times = step_times
        self.step_name = step_name
        self.start: float = 0

    def __enter__(self):
        from time import perf_counter

        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        from time import perf_counter

        self.step_times[self.step_name] = round(perf_counter() - self.start, 2)


async def run_export(
    task_id: str,
    pt_file: Path,
    triton_name: str,
    max_batch: int,
    formats: list[str],
    normalize_boxes: bool,
    auto_load: bool,
) -> None:
    """
    Run the export process in background.

    Updates task status throughout the process.
    Tracks timing for each step to provide user feedback.
    """
    from time import perf_counter

    task = export_tasks[task_id]
    step_times: dict[str, float] = {}
    total_start = perf_counter()

    try:
        task['status'] = ExportStatus.EXPORTING
        task['started_at'] = datetime.now(UTC)
        task['current_step'] = 'Starting export'
        task['progress'] = 10.0
        task['step_times'] = step_times
        _persist_task(task_id)

        # Build export command — route by architecture family.
        if task.get('end2end'):
            # YOLO26 (natively NMS-free): native toolchain; any TRT-flavored
            # format request maps to its single fused 'trt' output. Labels +
            # config.pbtxt are always generated by this exporter.
            y26_formats = (
                ['trt'] if any(f.startswith(('trt', 'all')) for f in formats) else ['onnx']
            )
            cmd = [
                'python',
                str(EXPORT_SCRIPT_Y26),
                '--custom-model',
                f'{pt_file}:{triton_name}:{max_batch}',
                '--formats',
                *y26_formats,
            ]
        else:
            cmd = [
                'python',
                str(EXPORT_SCRIPT),
                '--custom-model',
                f'{pt_file}:{triton_name}:{max_batch}',
                '--formats',
                *formats,
                '--save-labels',
                '--generate-config',
            ]
            if normalize_boxes:
                cmd.append('--normalize-boxes')

        logger.info(f'[{task_id}] Running: {" ".join(cmd)}')
        task['current_step'] = f'Exporting: {", ".join(formats)}'
        task['progress'] = 20.0

        # Run export subprocess with timing
        with StepTimer(step_times, 'total_export'):
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd='/app',
            )

            output_lines = []
            current_phase = 'initialization'
            phase_start = perf_counter()

            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line_str = line.decode('utf-8', errors='replace').strip()
                output_lines.append(line_str)

                # Update progress and track step timing based on output
                if 'ONNX' in line_str and 'Export' in line_str:
                    if current_phase != 'onnx_export':
                        step_times[current_phase] = round(perf_counter() - phase_start, 2)
                        current_phase = 'onnx_export'
                        phase_start = perf_counter()
                    task['progress'] = 40.0
                    task['current_step'] = 'Exporting ONNX End2End'
                elif 'TensorRT' in line_str or 'Building' in line_str:
                    if current_phase != 'tensorrt_build':
                        step_times[current_phase] = round(perf_counter() - phase_start, 2)
                        current_phase = 'tensorrt_build'
                        phase_start = perf_counter()
                    task['progress'] = 60.0
                    task['current_step'] = 'Building TensorRT engine (this takes 2-5 min)'
                elif 'Engine saved' in line_str:
                    step_times[current_phase] = round(perf_counter() - phase_start, 2)
                    current_phase = 'finalization'
                    phase_start = perf_counter()
                    task['progress'] = 85.0
                    task['current_step'] = 'Finalizing export'

                # Update step_times in real-time, persisted so a status
                # poll (or a restart) after this point sees real progress
                # rather than the 'Starting export' snapshot.
                task['step_times'] = step_times
                _persist_task(task_id)

            await process.wait()

            # Record final phase time
            step_times[current_phase] = round(perf_counter() - phase_start, 2)

        if process.returncode != 0:
            error_output = '\n'.join(output_lines[-20:])
            raise RuntimeError(f'Export failed:\n{error_output}')

        task['progress'] = 90.0
        task['formats_completed'] = formats
        task['formats_pending'] = []
        _persist_task(task_id)

        # Auto-load into Triton with timing
        if auto_load:
            task['status'] = ExportStatus.LOADING
            task['current_step'] = 'Loading into Triton'
            _persist_task(task_id)

            with StepTimer(step_times, 'triton_load'):
                triton = TritonControlService()
                models_to_load = []

                if task.get('end2end'):
                    # Native NMS-free exporter writes a single fused engine
                    models_to_load.append(f'{triton_name}_trt')
                else:
                    if 'trt' in formats or 'all' in formats:
                        models_to_load.append(f'{triton_name}_trt')
                    if 'trt_end2end' in formats or 'all' in formats:
                        models_to_load.append(f'{triton_name}_trt_end2end')

                for model in models_to_load:
                    success, msg = await triton.load_model(model)
                    if success:
                        task['triton_loaded'] = True
                        logger.info(f'[{task_id}] Loaded {model}')
                    else:
                        logger.warning(f'[{task_id}] Failed to load {model}: {msg}')

        task['status'] = ExportStatus.COMPLETED
        task['completed_at'] = datetime.now(UTC)
        task['progress'] = 100.0
        task['current_step'] = 'Complete'

        # Calculate total duration
        total_duration = round(perf_counter() - total_start, 2)
        task['export_duration_seconds'] = total_duration
        task['step_times'] = step_times
        task['message'] = f'Model {triton_name} exported successfully in {total_duration:.1f}s'

        logger.info(f'[{task_id}] Export completed in {total_duration:.1f}s')
        logger.info(f'[{task_id}] Step times: {step_times}')
        _persist_task(task_id)

    except Exception as e:
        logger.exception(f'[{task_id}] Export failed')
        task['status'] = ExportStatus.FAILED
        task['error'] = str(e)
        task['message'] = f'Export failed: {e}'
        task['completed_at'] = datetime.now(UTC)
        task['export_duration_seconds'] = round(perf_counter() - total_start, 2)
        _persist_task(task_id)


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
