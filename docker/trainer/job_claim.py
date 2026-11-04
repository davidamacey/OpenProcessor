"""Create-only generation claims on a training job file.

Two trainer replicas must never run one job. The claim is a file next to the job,
``<job_id>.claim.g<N>``, created with ``O_CREAT | O_EXCL`` -- Python's ``os.open`` raises
``FileExistsError`` when the file exists
(https://docs.python.org/3/library/os.html#os.open), so exactly one process wins each
generation. The holder is the highest generation present.

A stale holder (no status heartbeat for ``STALE_AFTER_S``) is replaced by creating generation
``N + 1``: exactly one reclaimer wins, the rest see ``FileExistsError`` and back off, and the
winner removes the older generations. A claim is never unlinked and recreated, and never
renamed over a live one. Before each status write a holder checks that no higher generation
exists (:func:`superseded`); if one does, it has been replaced and stops writing.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


STALE_AFTER_S = 120.0


def _claims(job_path: Path) -> dict[int, Path]:
    job_id = job_path.name.removesuffix('.job.json')
    found: dict[int, Path] = {}
    for path in job_path.parent.glob(f'{job_id}.claim.g*'):
        suffix = path.name.rsplit('.claim.g', 1)[1]
        if suffix.isdigit():
            found[int(suffix)] = path
    return found


def _create(job_path: Path, generation: int) -> Path | None:
    job_id = job_path.name.removesuffix('.job.json')
    path = job_path.with_name(f'{job_id}.claim.g{generation}')
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None
    with os.fdopen(fd, 'w') as handle:
        json.dump(
            {
                'owner': f'{socket.gethostname()}:{os.getpid()}',
                'generation': generation,
                'claimed_at': time.time(),
            },
            handle,
        )
    return path


def _last_alive(job_path: Path, claim: Path) -> float:
    """The newest sign of life: the claim file or the job's status heartbeat."""
    status = job_path.with_name(job_path.name.removesuffix('.job.json') + '.status.json')
    times = [claim.stat().st_mtime]
    with contextlib.suppress(OSError):
        times.append(status.stat().st_mtime)
    return max(times)


def claim(job_path: Path, *, stale_after_s: float = STALE_AFTER_S) -> int | None:
    """Claim ``job_path``: the generation held, or ``None`` when a live holder has it (or
    another reclaimer won the race)."""
    existing = _claims(job_path)
    if not existing:
        return 1 if _create(job_path, 1) is not None else None
    holder = max(existing)
    try:
        alive = _last_alive(job_path, existing[holder])
    except OSError:
        return None  # the holder's claim vanished while we looked: back off, retry next poll
    if time.time() - alive <= stale_after_s:
        return None
    if _create(job_path, holder + 1) is None:
        return None
    for generation, path in existing.items():
        if generation <= holder:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
    return holder + 1


def superseded(job_path: Path, generation: int) -> bool:
    """True once a claim with a higher generation exists: the holder was replaced."""
    return any(g > generation for g in _claims(job_path))


def release(job_path: Path) -> None:
    """Delete this job's claim files (a terminal state needs none)."""
    for path in _claims(job_path).values():
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
