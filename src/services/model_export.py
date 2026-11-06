"""
Model Export Service (API side): the validation and naming rules of an uploaded model.

The API image carries no TensorRT, so ``POST /models/upload`` and
``POST /models/{name}/export`` do not run the export themselves. They validate the model here and
start a ``model_export`` job (``src/services/model_export_jobs.py``) that the ``gpu_export``
consumer, the export image's ``export/celery_tasks.py``, runs. The job row is the task and its
status; nothing is kept in a process or in a file.
"""

import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.services.checkpoint_inspect import YOLOV5_ORIGINAL_MESSAGE, head_kind, is_original_yolov5


logger = logging.getLogger(__name__)

PYTORCH_MODELS_DIR = Path('/app/pytorch_models')


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
        try:
            ckpt, _ = torch_safe_load(file_path, safe_only=True)
        except TypeError:
            if is_original_yolov5(file_path):
                return False, YOLOV5_ORIGINAL_MESSAGE, {}
            raise
        model = ckpt.get('model') if isinstance(ckpt, dict) else None
        if model is None or not hasattr(model, 'names'):
            return False, 'Not an Ultralytics model checkpoint', {}
        names = model.names

        model_info = {
            'task': getattr(model, 'task', 'detect'),
            'num_classes': len(names),
            'class_names': list(names.values()) if isinstance(names, dict) else [],
            # The NMS-free head (YOLO26, YOLOv10) exports through the native toolchain; the
            # other heads go through the EfficientNMS toolchain. The export worker routes on it.
            'head_kind': head_kind(model),
            'ultralytics_version': ckpt.get('version'),
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


class ExportWorkerUnavailableError(RuntimeError):
    """No model-export worker is running, so a started job would never be picked up."""


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
