"""The one path policy for everything a dataset names.

A dataset is untrusted input: ``data.yaml`` entries, list-file lines,
COCO ``file_name`` values, OpenProcessor-export stems and archive members
are all attacker-controlled strings that become filesystem paths. Every
reader resolves them through the two functions here, so there is exactly
one place that decides what a dataset may reach:

* :func:`resolve_ref`: a *relative* reference (a COCO ``file_name``, an
  export stem, an archive member) must stay inside its base directory
  after symlinks are resolved.
* :func:`dataset_path_guard`: an *absolute* path (a source root, a
  ``data.yaml`` ``path:``, a list-file line) must, after symlinks are
  resolved, sit under a configured source root, the project's upload
  root, or the project's export root (read-only).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


PathGuard = Callable[[Path], bool]


class DatasetPathNotAllowedError(Exception):
    """A dataset root (or a path it names) is outside the allowed roots."""


def resolve_ref(base: Path, ref: str) -> Path | None:
    """Resolve ``ref`` under ``base`` and return the resolved path, or
    ``None`` when it escapes ``base`` (``..`` segments, an absolute
    reference, a NUL byte, or a symlink whose target is outside).

    The result is compared against the *resolved* base, so a symlinked
    dataset directory still works while a symlinked member pointing out of
    it does not.
    """
    if not ref or '\x00' in ref:
        return None
    normalized = ref.replace('\\', '/')
    if normalized.startswith(('/', '~')):
        return None
    try:
        base_resolved = base.resolve()
        candidate = (base_resolved / normalized).resolve()
        candidate.relative_to(base_resolved)
    except (OSError, ValueError, RuntimeError):
        return None
    return candidate


def dataset_path_guard(*extra_roots: Path | None) -> PathGuard:
    """A guard accepting a path whose symlink-resolved form is servable
    (:func:`~src.services.curation.image_serving.is_servable_image_path`)
    or lies under one of ``extra_roots`` (the project's export root: an
    OpenProcessor export is re-importable where it was written)."""
    from src.services.curation.image_serving import is_servable_image_path

    roots = [r.resolve() for r in extra_roots if r is not None]

    def guard(path: Path) -> bool:
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            return False
        if is_servable_image_path(str(resolved)):
            return True
        for root in roots:
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            return True
        return False

    return guard


__all__ = ['DatasetPathNotAllowedError', 'PathGuard', 'dataset_path_guard', 'resolve_ref']
