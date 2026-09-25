"""Shared cross-process singleton-start lock for the curation subsystem's
file-backed job runners (sibling of :mod:`src.services.curation.job_reconcile`,
which handles startup repair rather than the start-time race).

Every file-backed job module here (``item_scores.job``, ``probe_job``, ...)
guards "start a new run" with a read-state / write-``'running'``-state
pair. That pair is only atomic *within one process*: ``yolo-api`` runs
under ``--workers=N``, i.e. N separate OS processes, so two concurrent
``start_job()`` calls landing on different worker processes at nearly the
same instant can both read ``status != 'running'`` before either has
written ``'running'`` -- both then schedule a background task against the
same job id, and the API has silently started two runs.

``fcntl.flock(..., LOCK_EX | LOCK_NB)`` on a dedicated lock file closes
that gap: only one process can hold the lock at a time (the OS enforces
this across processes, not just within one), so the
check-is-busy-then-claim sequence becomes atomic across the whole fleet,
not just one worker. The lock is held only for the duration of that
check-and-claim -- never across the job's actual execution -- so it never
blocks a concurrent status poll or cancel request (those only read/touch
plain files, never this lock).

This was a real gap in ``item_scores.job.start_job`` (no lock at all, just
a bare ``_is_busy()`` check before the write) as well as the probe job
this module was built to fix -- both now go through
:func:`exclusive_start_lock`.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextlib.contextmanager
def exclusive_start_lock(lock_file: Path) -> Iterator[bool]:
    """Best-effort cross-process mutual exclusion for a job's
    check-and-claim critical section.

    Yields ``True`` if the lock was acquired -- the caller may proceed
    with its own busy check + state write, holding the lock until the
    ``with`` block exits. Yields ``False`` if another process currently
    holds it, which the caller should treat identically to "a job is
    already running" (some other process is mid-claim right now).
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


__all__ = ['exclusive_start_lock']
