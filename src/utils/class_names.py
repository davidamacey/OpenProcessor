"""Per-model class-id -> name resolution for detection responses.

F-42 (fresh-start E2E findings 2026-09-25, round 2): ``/detect`` and
``/detect/batch`` accept an arbitrary ``model_name`` query param, but the
class-name lookup used to be a single hardcoded ``COCO_CLASSES.get(...)``
(see :mod:`src.utils.affine`) applied to *every* model. A promoted
detector with its own class registry (e.g. a vehicle detector where id 0
is ``car``) would come back labeled with whatever COCO word happens to
share that id (``bus``, ``train``, ...) -- silently wrong, not an error.

Resolution order, cached per model name:

1. ``<models_dir>/<model_name>/labels.txt`` -- one name per line, in id
   order. This is exactly what
   :func:`src.services.training.triton_promote` writes at promote time
   (:func:`src.services.training.yolo_triton_config.render_labels_file`),
   and what the stock detectors already ship (see
   ``models/yolov11_small_trt_end2end/labels.txt``).
2. A ``label_filename`` referenced from the model's ``config.pbtxt``
   (Triton's own native mechanism for this; unused by any config this
   repo currently generates, but honored here for a hand-authored or
   third-party model that sets it).
3. :data:`src.utils.affine.COCO_CLASSES`, but **only** for the small,
   fixed set of stock COCO detector names -- never as a silent default
   for an unrecognized model. An unresolvable id for any other model
   renders as ``class_{id}``, exactly like the existing "not in
   COCO_CLASSES" fallback, rather than a wrong label from a different
   model's vocabulary.

The cache is invalidated on promote/unload so a freshly promoted or
removed model's labels are picked up on its very first request
afterward, not stale for the lifetime of the process.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from src.utils.affine import COCO_CLASSES


# Mirrors src.services.training.triton_promote.resolve_triton_models_dir --
# duplicated (not imported) so this module has no dependency on the
# training subsystem, which pulls in a much heavier import chain.
_DEFAULT_TRITON_MODELS_DIR = Path('/app/models')

# Stock detectors this repo ships that serve the standard 80-class COCO
# vocabulary. Both already carry their own labels.txt (resolution step 1
# handles them without ever reaching this set); it exists purely as a
# safety net if that file is ever missing or unreadable.
_STOCK_COCO_MODEL_NAMES = frozenset(
    {
        'yolov11_small_trt_end2end',
        'yolov11_small_end2end',
    }
)

_LABEL_FILENAME_RE = re.compile(r'label_filename\s*:\s*"([^"]*)"')

_cache_lock = threading.Lock()
_class_name_cache: dict[str, dict[int, str]] = {}


def _triton_models_dir() -> Path:
    override = os.environ.get('OP_TRITON_MODEL_REPO')
    return Path(override) if override else _DEFAULT_TRITON_MODELS_DIR


def _read_labels_file(path: Path) -> dict[int, str] | None:
    """Parse a ``labels.txt``: one name per line, line index = class id.

    Returns ``None`` (not found / unreadable / empty) rather than ``{}``
    so callers can tell "no labels file" apart from "labels file with no
    usable content" and keep trying the next resolution step either way.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return None
    lines = [line.strip() for line in text.splitlines()]
    names = {i: name for i, name in enumerate(lines) if name}
    return names or None


def _label_filename_from_config(model_dir: Path) -> str | None:
    try:
        text = (model_dir / 'config.pbtxt').read_text(encoding='utf-8')
    except OSError:
        return None
    match = _LABEL_FILENAME_RE.search(text)
    return match.group(1) if match and match.group(1) else None


def _load_class_names(model_name: str) -> dict[int, str]:
    model_dir = _triton_models_dir() / model_name

    names = _read_labels_file(model_dir / 'labels.txt')
    if names is not None:
        return names

    label_filename = _label_filename_from_config(model_dir)
    if label_filename:
        names = _read_labels_file(model_dir / label_filename)
        if names is not None:
            return names

    if model_name in _STOCK_COCO_MODEL_NAMES:
        return dict(COCO_CLASSES)

    return {}


def get_class_names(model_name: str) -> dict[int, str]:
    """Class-id -> name map for ``model_name``, loaded once and cached."""
    with _cache_lock:
        cached = _class_name_cache.get(model_name)
    if cached is not None:
        return cached

    names = _load_class_names(model_name)
    with _cache_lock:
        _class_name_cache[model_name] = names
    return names


def resolve_class_name(model_name: str, class_id: int) -> str:
    """Human-readable name for ``class_id`` under ``model_name``.

    Falls back to ``class_{id}`` -- never another model's vocabulary --
    when the id has no resolvable name.
    """
    return get_class_names(model_name).get(class_id, f'class_{class_id}')


def invalidate_class_names(model_name: str) -> None:
    """Drop the cached mapping for ``model_name``.

    Call after promoting or unloading a model so the very next request
    re-reads its (possibly just-written or just-removed) ``labels.txt``
    instead of serving a stale cached mapping for the rest of the
    process's life.
    """
    with _cache_lock:
        _class_name_cache.pop(model_name, None)


def clear_class_name_cache() -> None:
    """Testing helper: drop every cached mapping."""
    with _cache_lock:
        _class_name_cache.clear()


__all__ = [
    'clear_class_name_cache',
    'get_class_names',
    'invalidate_class_names',
    'resolve_class_name',
]
