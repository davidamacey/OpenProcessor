"""Per-job archive of auto-label states, so a job stays addressable by id.

``state.json`` only ever holds the current/last job. When a new job
replaces it, :func:`archive` keeps the outgoing job's final state under
``<state_dir>/history/<job_id>.json`` (newest :data:`KEEP` kept), and
:func:`load` reads it back for ``GET /pipeline/auto_label/status/{job_id}``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path


KEEP = 50
_JOB_ID = re.compile(r'^[0-9a-f]{32}$')


def _history_dir(state_dir: Path) -> Path:
    return state_dir / 'history'


def valid_job_id(job_id: str) -> bool:
    """``job_id`` has the shape :func:`uuid.uuid4().hex` produces (and so
    is safe as a file name)."""
    return bool(_JOB_ID.match(job_id))


def archive(state_dir: Path, state: dict[str, Any]) -> None:
    """Keep ``state`` (a job's final snapshot) and prune to :data:`KEEP`."""
    job_id = str(state.get('job_id') or '')
    if not valid_job_id(job_id):
        return
    hist = _history_dir(state_dir)
    hist.mkdir(parents=True, exist_ok=True)
    tmp = hist / f'{job_id}.tmp'
    tmp.write_text(json.dumps(state, default=str))
    tmp.replace(hist / f'{job_id}.json')
    kept = sorted(hist.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in kept[KEEP:]:
        stale.unlink(missing_ok=True)


def load(state_dir: Path, job_id: str) -> dict[str, Any] | None:
    """The archived state of ``job_id``; ``None`` if unknown."""
    if not valid_job_id(job_id):
        return None
    try:
        data = json.loads((_history_dir(state_dir) / f'{job_id}.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


__all__ = ['KEEP', 'archive', 'load', 'valid_job_id']
