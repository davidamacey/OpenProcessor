"""Source-image page-cache hints, plus the tmpfs item-crop cache.

Two independent things share this module because both sit between the
ingest path and a downstream reader of the same bytes:

* :func:`prefetch_paths` / :func:`release_after_decode` — generic,
  dataset-agnostic wrappers around :func:`os.posix_fadvise` that let
  callers hint the kernel page cache for *source* images on NAS/NVM
  storage.
* :func:`write_crop_cache` — the write side of the tmpfs item-crop
  cache (``CurationConfig.crop_cache_dir``). The curation ingest
  service writes each item's JPEG bytes here at ingest time; downstream
  readers (the detection worker, the VLM router) read
  ``<crop_cache_dir>/<crop_id>.jpg`` before falling back to re-cropping
  from the source image. Before this, the cache had readers and no
  writer (100% miss rate out of the box).

``prefetch_paths`` / ``release_after_decode`` tolerate missing files
(logged at DEBUG, skipped) and are safe to call from any thread.
``write_crop_cache`` is best-effort: a write failure is logged and
swallowed, never raised, since a cache miss just costs a downstream
re-crop rather than losing data.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from PIL import Image


logger = logging.getLogger(__name__)

DEFAULT_QUEUE_DEPTH = 32
DEFAULT_CROP_CACHE_QUALITY = 90

# ST-1: prune the tmpfs crop cache back down to this fraction of the cap
# once it's exceeded, so a big ingest batch doesn't retrigger a prune on
# every single write.
DEFAULT_PRUNE_TARGET_FRACTION = 0.9
# Skip .tmp.* partial writes younger than this — an in-flight write, not
# an orphan from a crashed process.
_TMP_FILE_MIN_AGE_S = 60.0

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


def write_crop_cache(
    crop_id: str,
    crop: Image.Image,
    cache_dir: str | Path,
    *,
    quality: int = DEFAULT_CROP_CACHE_QUALITY,
) -> None:
    """Write one item crop's JPEG bytes to the tmpfs crop cache.

    Best-effort: a write failure is logged and swallowed, not raised —
    a downstream reader falls back to re-cropping from the source image
    on a miss, so a cache-write failure never blocks ingest.

    Args:
        crop_id: The item's stable id; the cache key is ``<crop_id>.jpg``.
        crop: A decoded PIL crop, already in the desired orientation.
        cache_dir: Root cache directory (``CurationConfig.crop_cache_dir``).
            A falsy value disables the cache entirely (no-op).
        quality: JPEG encode quality.
    """
    if not cache_dir:
        return
    try:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        out = cache_root / f'{crop_id}.jpg'
        # Atomic write: tmp + rename so a partial write is never visible
        # to a concurrent reader.
        tmp = cache_root / f'{crop_id}.jpg.tmp.{os.getpid()}'
        crop.save(tmp, format='JPEG', quality=quality)
        tmp.replace(out)
    except Exception as exc:
        logger.warning('write_crop_cache failed for crop_id=%s: %s', crop_id, exc)


def prune_crop_cache(
    cache_dir: str | Path,
    max_bytes: int,
    *,
    target_fraction: float = DEFAULT_PRUNE_TARGET_FRACTION,
) -> tuple[int, int]:
    """Evict the oldest crop-cache entries until the total size is back
    under ``target_fraction * max_bytes``.

    ``max_bytes <= 0`` disables pruning (returns ``(0, 0)`` immediately) --
    ST-1's ``CurationConfig.crop_cache_max_bytes`` uses ``0`` as the
    explicit opt-out.

    Takes a non-blocking ``flock`` on ``<cache_dir>/.prune.lock`` so two
    concurrent callers (e.g. two ingest workers) never race each other's
    deletes; a caller that doesn't get the lock returns ``(0, 0)`` rather
    than blocking. ``*.tmp.*`` partial writes younger than
    ``_TMP_FILE_MIN_AGE_S`` are skipped (an in-flight write, not an
    orphan).

    Returns ``(files_removed, bytes_removed)``. Increments
    :data:`~src.services.curation.metrics.OP_SHM_CROP_CACHE_EVICTIONS` by
    the file count on removal.
    """
    if max_bytes <= 0:
        return (0, 0)

    cache_root = Path(cache_dir)
    if not cache_root.is_dir():
        return (0, 0)

    lock_path = cache_root / '.prune.lock'
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as exc:
        logger.warning('prune_crop_cache lock open failed: %s', exc)
        return (0, 0)

    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another process holds the lock — this run is a no-op, not
            # an error; the holder is already pruning.
            return (0, 0)

        now = time.time()
        entries: list[tuple[float, int, Path]] = []
        total_bytes = 0
        with os.scandir(cache_root) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                name = entry.name
                if name.startswith('.'):
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if '.tmp.' in name and (now - st.st_mtime) < _TMP_FILE_MIN_AGE_S:
                    continue
                total_bytes += st.st_size
                if name.endswith('.jpg'):
                    entries.append((st.st_mtime, st.st_size, Path(entry.path)))

        if total_bytes <= max_bytes:
            return (0, 0)

        target_bytes = int(max_bytes * target_fraction)
        entries.sort(key=lambda e: e[0])  # oldest mtime first

        removed_files = 0
        removed_bytes = 0
        remaining = total_bytes
        for _mtime, size, path in entries:
            if remaining <= target_bytes:
                break
            try:
                path.unlink()
            except OSError as exc:
                logger.debug('prune_crop_cache unlink failed for %s: %s', path, exc)
                continue
            remaining -= size
            removed_files += 1
            removed_bytes += size

        if removed_files:
            from src.services.curation.metrics import OP_SHM_CROP_CACHE_EVICTIONS

            OP_SHM_CROP_CACHE_EVICTIONS.inc(removed_files)
            logger.info(
                'prune_crop_cache removed %d files (%d bytes), cache_dir=%s',
                removed_files,
                removed_bytes,
                cache_root,
            )
        return (removed_files, removed_bytes)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(lock_fd)


class _PruneCallCounter:
    """Mutable holder so ``maybe_prune_crop_cache`` avoids a bare module
    ``global`` statement; still one counter per process, same semantics."""

    count: int = 0


_prune_call_counter = _PruneCallCounter()


def maybe_prune_crop_cache(
    cache_dir: str | Path,
    max_bytes: int,
    *,
    every: int = 256,
) -> None:
    """Call :func:`prune_crop_cache` roughly every ``every`` calls.

    A per-process write counter, not a per-file one — cheap enough to call
    unconditionally from the ingest hot path (``ingest_one``) without
    ``scandir``-ing the cache directory on every single crop write.
    """
    _prune_call_counter.count += 1
    if every <= 0 or _prune_call_counter.count % every != 0:
        return
    try:
        prune_crop_cache(cache_dir, max_bytes)
    except Exception as exc:
        logger.warning('maybe_prune_crop_cache failed: %s', exc)
