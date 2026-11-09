"""What an export job runs: the export subprocess command and the engines it produces.

The Celery handler of the ``gpu_export`` queue is ``export/export_tasks.py``. Every head kind goes
through the one script ``export/ultralytics_export.py``; the head kind only decides the name of the
engine it builds (``src/domain/model_families.py::engine_name``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.domain.model_families import HEAD_KINDS, engine_name


EXPORT_DIR = Path(__file__).resolve().parent
APP_DIR = EXPORT_DIR.parent
EXPORT_SCRIPT = EXPORT_DIR / 'ultralytics_export.py'


def _head_kind(job: dict[str, Any]) -> str:
    """The job's head kind. A job queued before head kinds existed carries none: fail closed."""
    kind = job.get('head_kind')
    if kind not in HEAD_KINDS:
        raise ValueError(
            'this export job has no head kind (it was queued by an older version): '
            're-upload the model'
        )
    return str(kind)


def build_command(job: dict[str, Any]) -> list[str]:
    """The export subprocess for a job. The script writes labels and ``config.pbtxt`` itself and,
    for a re-export, replaces a differing config: the config must match the engine it serves."""
    _head_kind(job)
    return [
        sys.executable,
        str(EXPORT_SCRIPT),
        '--custom-model',
        f'{job["pt_file"]}:{job["triton_name"]}:{job["max_batch"]}',
        '--formats',
        *job['formats'],
        '--overwrite-config',
    ]


def engines_to_load(job: dict[str, Any]) -> list[str]:
    """The Triton models the job builds: one engine for ``trt`` or ``all``, none for ``onnx``."""
    kind = _head_kind(job)
    formats = job['formats']
    if 'trt' in formats or 'all' in formats:
        return [engine_name(job['triton_name'], kind)]
    return []
