"""Ownership of the global worker pause sentinel (issue #127).

Two writers share ``vlm_worker/pause.sentinel``: the GPU arbiter (single-GPU
training claim) and ``openprocessor vlm use`` (pauses workers across a model
swap). The arbiter's reconcile loop used to ``unlink`` the file whenever no run
was active, which silently un-paused the workers in the middle of a vlm switch.

The file body says who made it. Arbiter-made files (and legacy empty ones, which
only the arbiter ever wrote) are the arbiter's to clear. A file owned by anyone
else is left alone until it is older than a TTL, so a crashed CLI cannot pause the
workers forever.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


ARBITER_OWNER = 'arbiter'
DEFAULT_FOREIGN_TTL_SECONDS = 1800.0


def foreign_ttl_seconds() -> float:
    try:
        return max(
            1.0, float(os.environ.get('OP_PAUSE_SENTINEL_TTL_S', DEFAULT_FOREIGN_TTL_SECONDS))
        )
    except ValueError:
        return DEFAULT_FOREIGN_TTL_SECONDS


def arbiter_body() -> str:
    return json.dumps({'owner': ARBITER_OWNER, 'created_at': time.time()})


def sentinel_owner(path: Path) -> str | None:
    """Owner named in the file body; ``None`` for an empty or unreadable body."""
    try:
        text = path.read_text(encoding='utf-8').strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return text.split()[0]
    owner = data.get('owner') if isinstance(data, dict) else None
    return owner if isinstance(owner, str) and owner else None


def arbiter_may_clear(path: Path, *, now: float | None = None) -> bool:
    """True when the arbiter owns ``path`` (or legacy-empty), or a foreign one is stale."""
    owner = sentinel_owner(path)
    if owner is None or owner == ARBITER_OWNER:
        return True
    try:
        age = (time.time() if now is None else now) - path.stat().st_mtime
    except OSError:
        return True
    return age > foreign_ttl_seconds()
