"""Which model directories a model name addresses.

One resolver for every route that takes a model name (load, unload, delete,
the project-scoped delete): every name ``/models/status`` shows addresses the
same directory everywhere. A name addresses its own directory when one
exists (a project promote, a core engine) and, as a convenience, the export
family a bare name expands to (``{name}_trt``, ``{name}_trt_end2end`` and
the ONNX intermediate ``{name}_end2end``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

EXPORT_SUFFIXES = ('_trt', '_trt_end2end', '_end2end')
# Preference order when loading by a bare name.
_LOAD_SUFFIXES = ('_trt_end2end', '_trt')


def resolve_model_dirs(models_dir: Path, name: str) -> list[Path]:
    """Every existing directory ``name`` addresses: its own, plus the export
    family of a bare name. Empty when nothing exists."""
    candidates = [name, *(f'{name}{s}' for s in EXPORT_SUFFIXES)]
    return [models_dir / c for c in candidates if (models_dir / c).exists()]


def resolve_load_dir(models_dir: Path, name: str) -> Path | None:
    """The one directory ``name`` loads: its own, else the End2End then the
    plain TRT export."""
    for candidate in (name, *(f'{name}{s}' for s in _LOAD_SUFFIXES)):
        if (models_dir / candidate).exists():
            return models_dir / candidate
    return None


def promoted_owner(model_dir: Path) -> str | None:
    """The project that promoted the model in ``model_dir`` (``promote.json``),
    ``None`` for an engine no project promoted."""
    try:
        raw = json.loads((model_dir / 'promote.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    project = raw.get('project') if isinstance(raw, dict) else None
    return str(project) if project else None
