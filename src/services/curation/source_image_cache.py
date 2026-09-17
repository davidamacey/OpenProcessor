"""NAS/NVM prefetch / release helpers for source image bytes.

Generic, dataset-agnostic wrappers around :func:`os.posix_fadvise` that let
callers hint the kernel page cache:

* :func:`prefetch_paths` issues ``POSIX_FADV_WILLNEED`` for a bounded queue
  of paths. The kernel keeps the willneed hint after the fd is closed, so
  callers don't need to manage fd lifetime themselves.
* :func:`release_after_decode` issues ``POSIX_FADV_DONTNEED`` once the
  caller has finished decoding a file, freeing the bytes from the page
  cache so large ingest sweeps don't evict hotter pages.

Both helpers tolerate missing files (logged at DEBUG, skipped) and are
safe to call from any thread.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)

DEFAULT_QUEUE_DEPTH = 32

PathOrFd = Path | str | int


def _fadvise(fd: int, advice: int) -> None:
    """Call :func:`os.posix_fadvise` over the whole file (offset=0, len=0)."""
    os.posix_fadvise(fd, 0, 0, advice)


def prefetch_paths(paths: list[Path], depth: int = DEFAULT_QUEUE_DEPTH) -> None:
    """Hint the kernel to read-ahead each path into the page cache.

    Opens each path read-only, issues ``POSIX_FADV_WILLNEED`` for the
    entire file, then closes the fd. The kernel retains the willneed hint
    after close, so the read-ahead happens regardless.

    Missing or unreadable paths are logged and skipped — this helper is
    advisory only and never raises for I/O errors on individual entries.

    Args:
        paths: Source-image paths to prefetch. May be empty.
        depth: Bounded queue depth. Currently used only to cap the
            iteration window so we don't issue thousands of fadvise calls
            in one go; callers chunk their own work via repeated calls.
    """
    if depth <= 0:
        raise ValueError(f'depth must be positive, got {depth}')

    for path in paths[:depth]:
        try:
            fd = os.open(os.fspath(path), os.O_RDONLY)
        except FileNotFoundError:
            logger.debug('prefetch_paths: missing file, skipping: %s', path)
            continue
        except OSError as exc:
            logger.debug('prefetch_paths: open failed for %s: %s', path, exc)
            continue
        try:
            _fadvise(fd, os.POSIX_FADV_WILLNEED)
        except OSError as exc:
            logger.debug('prefetch_paths: fadvise failed for %s: %s', path, exc)
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)


def release_after_decode(path_or_fd: PathOrFd) -> None:
    """Hint the kernel to drop a file's pages from the page cache.

    Accepts either a path-like or an already-open file descriptor int.
    When given a path, opens it read-only solely to issue
    ``POSIX_FADV_DONTNEED`` then closes. Missing files are logged and
    skipped.
    """
    if isinstance(path_or_fd, int):
        try:
            _fadvise(path_or_fd, os.POSIX_FADV_DONTNEED)
        except OSError as exc:
            logger.debug('release_after_decode: fadvise failed on fd %d: %s', path_or_fd, exc)
        return

    try:
        fd = os.open(os.fspath(path_or_fd), os.O_RDONLY)
    except FileNotFoundError:
        logger.debug('release_after_decode: missing file, skipping: %s', path_or_fd)
        return
    except OSError as exc:
        logger.debug('release_after_decode: open failed for %s: %s', path_or_fd, exc)
        return
    try:
        _fadvise(fd, os.POSIX_FADV_DONTNEED)
    except OSError as exc:
        logger.debug('release_after_decode: fadvise failed for %s: %s', path_or_fd, exc)
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
