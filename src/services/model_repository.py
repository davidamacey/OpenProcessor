"""Which model directories a model name addresses.

One resolver for every route that takes a model name (load, unload, delete,
the project-scoped delete): every name ``/models/status`` shows addresses the
same directory everywhere. A name addresses its own directory when one
exists (a project promote, a core engine) and, as a convenience, the export
family a bare name expands to (``{name}_trt``, ``{name}_trt_end2end`` and
the ONNX intermediate ``{name}_end2end``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.model_export import check_triton_name


if TYPE_CHECKING:
    from pathlib import Path

EXPORT_SUFFIXES = ('_trt', '_trt_end2end', '_end2end')
# Preference order when loading by a bare name.
_LOAD_SUFFIXES = ('_trt_end2end', '_trt')


def checked_model_name(name: str) -> str:
    """``name`` if it is a model slug, else ``ValueError``. Allowlist first, so a
    dot segment, separator or empty name never reaches a filesystem call."""
    return check_triton_name(name)


def inside_repository(base: Path, path: Path) -> bool:
    """Whether ``path`` resolves strictly inside ``base`` (symlinks and ``..``
    resolved; ``base`` itself does not count)."""
    resolved = path.resolve()
    return resolved != base.resolve() and resolved.is_relative_to(base.resolve())


def resolve_model_dirs(models_dir: Path, name: str) -> list[Path]:
    """Every existing directory ``name`` addresses: its own, plus the export
    family of a bare name. Empty when nothing exists."""
    candidates = [name, *(f'{name}{s}' for s in EXPORT_SUFFIXES)]
    return [
        models_dir / c
        for c in candidates
        if (models_dir / c).exists() and inside_repository(models_dir, models_dir / c)
    ]


def resolve_load_dir(models_dir: Path, name: str) -> Path | None:
    """The one directory ``name`` loads: its own, else the End2End then the
    plain TRT export."""
    for candidate in (name, *(f'{name}{s}' for s in _LOAD_SUFFIXES)):
        if (models_dir / candidate).exists() and inside_repository(
            models_dir, models_dir / candidate
        ):
            return models_dir / candidate
    return None


async def promoted_owner(name: str) -> str | None:
    """The project that promoted the model ``name`` (its live ``promotions`` row), ``None`` for
    an engine no project promoted."""
    from src.services.training.model_registry import owner_slug

    return await owner_slug(name)
