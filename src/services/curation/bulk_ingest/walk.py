"""Parallel directory walk for bulk ingest.

A plain ``os.walk`` is single-threaded and, on network-attached or
otherwise high-latency storage, spends most of its wall time blocked on
``readdir`` syscalls rather than CPU. This module fans a directory tree
out across a thread pool: each subdirectory ``os.scandir`` call is an
independent unit of work, submitted recursively as it discovers more
subdirectories, so I/O-bound directory listings overlap.

Not a generic filesystem library — scoped to exactly what
bulk ingest needs: yield every regular file under ``root``
whose suffix is in an allowed set.
"""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Generator


UNREADABLE_SAMPLE_LIMIT = 10


@dataclass
class WalkStats:
    """What a walk skipped, so a caller can report it instead of hiding it."""

    unreadable_dirs: int = 0
    symlinks: int = 0
    # The first few unreadable directories, for the message.
    unreadable_samples: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_unreadable(self, path: Path) -> None:
        with self._lock:
            self.unreadable_dirs += 1
            if len(self.unreadable_samples) < UNREADABLE_SAMPLE_LIMIT:
                self.unreadable_samples.append(str(path))

    def add_symlink(self) -> None:
        with self._lock:
            self.symlinks += 1


class _Walk:
    """One parallel walk: workers scan directories and feed a bounded queue the caller drains."""

    def __init__(
        self, root: Path, extensions: frozenset[str], workers: int, stats: WalkStats | None
    ) -> None:
        self._root = root
        self._exts = frozenset(e.lower() for e in extensions)
        self._stats = stats
        self._out: queue.Queue[Path] = queue.Queue(maxsize=4096)
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._pending_zero = threading.Event()
        # Set when the consumer stops early (a cancelled run): a worker blocked on a
        # full queue must give up, or closing the pool would wait on it forever.
        self._stop = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers))

    def _put(self, path: Path) -> None:
        while not self._stop.is_set():
            try:
                self._out.put(path, timeout=0.1)
                return
            except queue.Full:
                continue

    def _list(self, dir_path: Path) -> list[os.DirEntry[str]]:
        # A closed walk must not keep listing the rest of a huge tree behind it.
        if self._stop.is_set():
            return []
        try:
            with os.scandir(dir_path) as it:
                return list(it)
        except OSError:
            if self._stats is not None:
                self._stats.add_unreadable(dir_path)
            return []

    def _visit(self, entry: os.DirEntry[str]) -> None:
        if entry.is_symlink():
            if self._stats is not None:
                self._stats.add_symlink()
        elif entry.is_dir(follow_symlinks=False):
            with self._pending_lock:
                self._pending += 1
            self._executor.submit(self._scan, Path(entry.path))
        elif entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in self._exts:
            self._put(Path(entry.path))

    def _scan(self, dir_path: Path) -> None:
        for entry in self._list(dir_path):
            try:
                self._visit(entry)
            except OSError:
                continue
        with self._pending_lock:
            self._pending -= 1
            if self._pending == 0:
                self._pending_zero.set()

    def run(self, should_stop: Callable[[], bool]) -> Generator[Path, None, None]:
        with self._pending_lock:
            self._pending += 1
        self._executor.submit(self._scan, self._root)
        try:
            # Drain the queue on the calling thread while workers fan out; a sentinel-free
            # design would race the "done" check against in-flight puts, so we poll with a
            # short timeout instead, which is also where a stop request is noticed even
            # when the tree yields no matching file.
            while True:
                try:
                    yield self._out.get(timeout=0.1)
                except queue.Empty:
                    if self._pending_zero.is_set() and self._out.empty():
                        break
                    if should_stop():
                        return
            # Drain anything queued between the last empty check and shutdown.
            while not self._out.empty():
                yield self._out.get_nowait()
        finally:
            self._stop.set()
            # Never wait on a listing stuck on a dead mount: a stopped walk returns now
            # and the stuck worker thread ends when its syscall does.
            self._executor.shutdown(wait=False, cancel_futures=True)


def iter_image_paths(
    root: Path,
    extensions: frozenset[str] = frozenset({'.jpg', '.jpeg', '.png'}),
    workers: int = 8,
    stats: WalkStats | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> Generator[Path, None, None]:
    """Yield every file under ``root`` (recursively) matching ``extensions``.

    Case-insensitive on suffix. Directories that raise ``OSError`` while
    being listed (permission denied, a broken symlink, a mount that
    dropped mid-walk) are skipped, not fatal — a single bad directory
    must never abort a multi-hour ingest sweep.

    Args:
        root: Directory to walk.
        extensions: Lowercase suffixes to include, e.g. ``{'.jpg'}``.
        workers: Thread-pool size for concurrent ``os.scandir`` calls.
        should_stop: Polled while the walk waits; when true the walk ends (a cancelled
            or time-boxed caller), even if the tree has yielded nothing for a while.
        stats: Filled with the unreadable directories and symlinks the walk skipped.

    Yields:
        ``Path`` objects for matching files, in discovery order (not
        sorted — callers needing a stable order should sort the result).
    """
    yield from _Walk(root, extensions, workers, stats).run(should_stop)


__all__ = ['WalkStats', 'iter_image_paths']
