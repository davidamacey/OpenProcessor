"""TensorRT build settings: the one place that decides workspace and engine max batch.

``OP_TRT_BUILD_TARGET`` selects ``default`` or ``small``; ``GPU_PROFILE=small``
implies ``small`` when the target is not set. ``OP_TRT_WORKSPACE_GB`` overrides the
workspace of every model. Invalid values raise: a build never silently falls back.

``scripts/lib/model_setup.sh`` keeps the same workspace table for the trtexec
build of the PE image encoder (a test keeps the two equal).
"""

from __future__ import annotations

import os
import re


TARGETS = ('default', 'small')

# Triton max_batch_size of config_templates/profiles/small.json (a test keeps them equal):
# an engine whose profile max is below it would make Triton reject full batches.
SMALL_PROFILE_MAX_BATCH = 16

SMALL_WORKSPACE_GB = 2
DEFAULT_WORKSPACE_GB = 4
_DEFAULT_WORKSPACE_GB_BY_MODEL = {
    'pe_image_encoder': 8,
    'paddleocr_rec_trt': 8,
}

_GB = 1 << 30


def build_target() -> str:
    """Return ``default`` or ``small``; raises ValueError on any other value."""
    raw = os.environ.get('OP_TRT_BUILD_TARGET', '')
    if raw == '':
        return 'small' if os.environ.get('GPU_PROFILE', '') == 'small' else 'default'
    if raw not in TARGETS:
        raise ValueError(f'OP_TRT_BUILD_TARGET must be one of {TARGETS}, got {raw!r}')
    return raw


def _workspace_override_gb() -> int | None:
    raw = os.environ.get('OP_TRT_WORKSPACE_GB', '')
    if raw == '':
        return None
    if not re.fullmatch(r'[0-9]+', raw) or int(raw) < 1:
        raise ValueError(f'OP_TRT_WORKSPACE_GB must be a whole number >= 1, got {raw!r}')
    return int(raw)


def workspace_gb(model: str) -> int:
    """TensorRT workspace pool limit for ``model``, in GB."""
    override = _workspace_override_gb()
    if override is not None:
        return override
    if build_target() == 'small':
        return SMALL_WORKSPACE_GB
    return _DEFAULT_WORKSPACE_GB_BY_MODEL.get(model, DEFAULT_WORKSPACE_GB)


def workspace_bytes(model: str) -> int:
    return workspace_gb(model) * _GB


def max_batch(model: str, default: int) -> int:
    """Engine max batch: ``default``, halved on the small target but never below the floor.

    The floor is the small profile's Triton max_batch_size; a default already below it
    (the OCR detector's 4) is never raised.
    """
    del model  # one rule for all engines; kept so a per-model table can be added without a rename
    if build_target() != 'small':
        return default
    return min(default, max(default // 2, SMALL_PROFILE_MAX_BATCH))
