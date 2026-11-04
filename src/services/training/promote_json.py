"""Leased, atomic read-modify-write of a promoted model's ``promote.json``
(projects_plan.md §5.5).

Two writers touch the file: ``PUT .../models/{name}/sharing`` (the owner's
opt-in, with optimistic concurrency on ``sharing_revision``) and a
re-promote (a new version of the same model). ``api`` runs several
worker processes, so every read-compare-write holds the short
``promote_json:<model_name>`` lease in the model owner's project (the bound
project) and every write is a temp file plus rename, so a crash never leaves
a torn file. A writer that finds the lease held does not wait: the sharing
route answers its revision conflict (refresh and retry), a re-promote fails.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any

from src.domain.job_lease import LeaseKind
from src.services.curation.lease_heartbeat import leased, new_lease_client
from src.utils.atomic_write import atomic_write_json


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# Fields only the sharing route changes; a re-promote carries them over.
_SHARING_FIELDS = ('shared', 'sharing_revision')


class SharingRevisionConflictError(Exception):
    """``expected_revision`` does not match the stored ``sharing_revision``."""

    def __init__(self, current_revision: int) -> None:
        super().__init__(f'sharing revision is {current_revision}')
        self.current_revision = current_revision


class PromoteJsonBusyError(RuntimeError):
    """Another process is writing this model's ``promote.json`` right now."""


@contextlib.asynccontextmanager
async def _writing(path: Path) -> AsyncIterator[None]:
    """The ``promote_json:<model_name>`` lease for one read-compare-write."""
    model_name = path.parent.name
    client = new_lease_client()
    try:
        async with leased(
            client,
            LeaseKind.PROMOTE_JSON,
            model_name,
            model_name,
            busy=lambda _holder: PromoteJsonBusyError(
                f'{model_name}: promote.json is being written'
            ),
        ):
            yield
    finally:
        await client.close()


def sharing_revision(raw: dict[str, Any]) -> int:
    """A freshly promoted model (no sharing write yet) is at revision 1."""
    return int(raw.get('sharing_revision') or 1)


async def update_sharing(path: Path, *, shared: bool, expected_revision: int) -> int:
    """Set ``shared`` if ``expected_revision`` is current; return the new
    revision. Raises ``OSError`` / ``ValueError`` when the file is missing
    or unreadable, and :class:`SharingRevisionConflictError` on a stale
    revision (also when a concurrent writer holds the lease: the caller refreshes)."""
    try:
        async with _writing(path):
            raw = json.loads(path.read_text(encoding='utf-8'))
            current = sharing_revision(raw)
            if expected_revision != current:
                raise SharingRevisionConflictError(current)
            raw['shared'] = shared
            raw['sharing_revision'] = current + 1
            atomic_write_json(path, raw, indent=2, sort_keys=True)
    except PromoteJsonBusyError:
        raise SharingRevisionConflictError(
            sharing_revision(json.loads(path.read_text('utf-8')))
        ) from None
    return current + 1


async def write_promote_backpointer(path: Path, backpointer: dict[str, Any]) -> None:
    """Write a (re-)promote's ``promote.json``, keeping the prior file's
    ``shared`` opt-in and ``sharing_revision`` (the sharing route is the
    only writer of those). A first promote starts unshared."""
    async with _writing(path):
        try:
            prior = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            prior = {}
        carried = {k: prior[k] for k in _SHARING_FIELDS if k in prior}
        if 'shared' in carried:
            carried['shared'] = bool(carried['shared'])
        atomic_write_json(
            path, {**backpointer, 'shared': False, **carried}, indent=2, sort_keys=True
        )


__all__ = [
    'PromoteJsonBusyError',
    'SharingRevisionConflictError',
    'sharing_revision',
    'update_sharing',
    'write_promote_backpointer',
]
