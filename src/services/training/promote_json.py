"""Locked, atomic read-modify-write of a promoted model's ``promote.json``
(projects_plan.md §5.5).

Two writers touch the file: ``PUT .../models/{name}/sharing`` (the owner's
opt-in, with optimistic concurrency on ``sharing_revision``) and a
re-promote (a new version of the same model). ``yolo-api`` runs several
worker processes, so every read-compare-write holds a per-model ``flock``
(:func:`~src.services.curation.job_lock.exclusive_file_lock`) and every
write is a temp file plus rename, so a crash never leaves a torn
file. The functions block; call them through ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from src.services.curation.job_lock import exclusive_file_lock


if TYPE_CHECKING:
    from pathlib import Path


# Fields only the sharing route changes; a re-promote carries them over.
_SHARING_FIELDS = ('shared', 'sharing_revision')


class SharingRevisionConflictError(Exception):
    """``expected_revision`` does not match the stored ``sharing_revision``."""

    def __init__(self, current_revision: int) -> None:
        super().__init__(f'sharing revision is {current_revision}')
        self.current_revision = current_revision


def _lock_file(path: Path) -> Path:
    return path.with_name(f'.{path.name}.lock')


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)


def sharing_revision(raw: dict[str, Any]) -> int:
    """A freshly promoted model (no sharing write yet) is at revision 1."""
    return int(raw.get('sharing_revision') or 1)


def update_sharing(path: Path, *, shared: bool, expected_revision: int) -> int:
    """Set ``shared`` if ``expected_revision`` is current; return the new
    revision. Raises ``OSError`` / ``ValueError`` when the file is missing
    or unreadable, and :class:`SharingRevisionConflictError` on a stale
    revision."""
    with exclusive_file_lock(_lock_file(path)):
        raw = json.loads(path.read_text(encoding='utf-8'))
        current = sharing_revision(raw)
        if expected_revision != current:
            raise SharingRevisionConflictError(current)
        raw['shared'] = shared
        raw['sharing_revision'] = current + 1
        _write_atomic(path, raw)
    return current + 1


def write_promote_backpointer(path: Path, backpointer: dict[str, Any]) -> None:
    """Write a (re-)promote's ``promote.json``, keeping the prior file's
    ``shared`` opt-in and ``sharing_revision`` (the sharing route is the
    only writer of those). A first promote starts unshared."""
    with exclusive_file_lock(_lock_file(path)):
        try:
            prior = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            prior = {}
        carried = {k: prior[k] for k in _SHARING_FIELDS if k in prior}
        if 'shared' in carried:
            carried['shared'] = bool(carried['shared'])
        _write_atomic(path, {**backpointer, 'shared': False, **carried})


__all__ = [
    'SharingRevisionConflictError',
    'sharing_revision',
    'update_sharing',
    'write_promote_backpointer',
]
