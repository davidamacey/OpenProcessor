"""
Model Export Service (API side): the validation and naming rules of an uploaded model.

The API image carries no TensorRT, so ``POST /models/upload`` and
``POST /models/{name}/export`` do not run the export themselves. They validate the model here and
start a ``model_export`` job (``src/services/model_export_jobs.py``) that the ``gpu_export``
consumer, the export image's ``export/export_tasks.py``, runs. The job row is the task and its
status; nothing is kept in a process or in a file.
"""

import logging
import pickle  # nosec B403 - only names pickle.UnpicklingError, never loads
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.domain.model_families import head_kind_of
from src.services.checkpoint_inspect import YOLOV5_ORIGINAL_MESSAGE, is_original_yolov5


logger = logging.getLogger(__name__)

PYTORCH_MODELS_DIR = Path('/app/pytorch_models')

# What reading an untrusted checkpoint can raise: a damaged archive, a refused or unknown pickle
# class, a missing attribute. Each one rejects the upload with its message.
_UNREADABLE_CHECKPOINT = (
    OSError,
    EOFError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    AssertionError,
    ImportError,
    pickle.UnpicklingError,
    zipfile.BadZipFile,
)


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
            'head_kind': head_kind_of(model.model[-1]),
            'ultralytics_version': ckpt.get('version'),
        }

        if model_info['task'] != 'detect':
            return False, f'Only detection models supported, got: {model_info["task"]}', model_info

        del model
        torch.cuda.empty_cache()

        return True, 'Valid YOLO detection model', model_info

    except _UNREADABLE_CHECKPOINT as e:
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


@dataclass(frozen=True)
class StagedUpload:
    """A validated upload waiting beside its final path; ``place`` makes it the stored weights."""

    path: Path
    final_path: Path
    model_info: dict[str, Any]

    def place(self) -> None:
        """Atomic: the weights a running build reads are never half-written."""
        self.path.replace(self.final_path)

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


async def stage_uploaded_file(content: bytes, triton_name: str) -> StagedUpload:
    """
    Validate an uploaded file and keep it next to its final path without replacing anything.

    The caller places it only once the export job exists (the job start is what answers 409
    for a running build of the same name).

    Raises:
        ValueError: If validation fails
    """
    PYTORCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Same directory as the final path (so ``place`` is a rename); a leading dot keeps it out of
    # the ``{name}*.pt`` lookup of the export route.
    with tempfile.NamedTemporaryFile(
        delete=False, dir=PYTORCH_MODELS_DIR, prefix='.upload-', suffix='.pt'
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        is_valid, error_msg, model_info = validate_pytorch_model(tmp_path)
        if not is_valid:
            raise ValueError(error_msg)
    except ValueError:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f'Failed to save file: {e}') from e
    return StagedUpload(tmp_path, PYTORCH_MODELS_DIR / f'{triton_name}.pt', model_info)
