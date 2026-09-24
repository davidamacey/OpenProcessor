"""Parallel directory walk for the curation ingest walker.

A plain ``os.walk`` is single-threaded and, on network-attached or
otherwise high-latency storage, spends most of its wall time blocked on
``readdir`` syscalls rather than CPU. This module fans a directory tree
out across a thread pool: each subdirectory ``os.scandir`` call is an
independent unit of work, submitted recursively as it discovers more
subdirectories, so I/O-bound directory listings overlap.

Not a generic filesystem library — scoped to exactly what
``ingest_walker.py`` needs: yield every regular file under ``root``
whose suffix is in an allowed set.
"""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


def iter_image_paths(
    root: Path,
    extensions: frozenset[str] = frozenset({'.jpg', '.jpeg', '.png'}),
    workers: int = 8,
) -> Iterator[Path]:
    """Yield every file under ``root`` (recursively) matching ``extensions``.

    Case-insensitive on suffix. Directories that raise ``OSError`` while
    being listed (permission denied, a broken symlink, a mount that
    dropped mid-walk) are skipped, not fatal — a single bad directory
    must never abort a multi-hour ingest sweep.

    Args:
        root: Directory to walk.
        extensions: Lowercase suffixes to include, e.g. ``{'.jpg'}``.
        workers: Thread-pool size for concurrent ``os.scandir`` calls.

    Yields:
        ``Path`` objects for matching files, in discovery order (not
        sorted — callers needing a stable order should sort the result).
    """
    exts = frozenset(e.lower() for e in extensions)
    out: queue.Queue[Path] = queue.Queue(maxsize=4096)
    pending = 0
    pending_lock = threading.Lock()
    pending_zero = threading.Event()

    def _scan(dir_path: Path) -> None:
        nonlocal pending
        try:
            with os.scandir(dir_path) as it:
                entries = list(it)
        except OSError:
            entries = []
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    with pending_lock:
                        pending += 1
                    executor.submit(_scan, Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    suffix = Path(entry.name).suffix.lower()
                    if suffix in exts:
                        out.put(Path(entry.path))
            except OSError:
                continue
        with pending_lock:
            pending -= 1
            if pending == 0:
                pending_zero.set()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        with pending_lock:
            pending += 1
        executor.submit(_scan, root)

        # Drain the queue on the calling thread while workers fan out;
        # a sentinel-free design would race the "done" check against
        # in-flight puts, so we poll with a short timeout instead.
        while True:
            try:
                item = out.get(timeout=0.1)
                yield item
            except queue.Empty:
                if pending_zero.is_set() and out.empty():
                    break
        # Drain anything queued between the last empty check and shutdown.
        while not out.empty():
            yield out.get_nowait()


__all__ = ['iter_image_paths']
