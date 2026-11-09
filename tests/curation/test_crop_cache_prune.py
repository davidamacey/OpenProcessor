"""ST-1: the tmpfs crop cache had no size cap. ``write_crop_cache`` never
evicted anything, so a busy ingest workload grows it unbounded until the
tmpfs mount fills. ``prune_crop_cache`` adds the missing cap; the beat tick
``op.periodic.cache_prune`` calls it under an advisory lock.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from src.services.curation.source_image_cache import prune_crop_cache


if TYPE_CHECKING:
    from pathlib import Path


_CROP_BYTES = 100 * 1024  # 100 KiB


def _write_fake_crops(cache_dir: Path, n: int, *, size: int = _CROP_BYTES) -> list[Path]:
    """Write ``n`` fake crop files with strictly increasing mtimes (oldest
    first, i.e. crop-0 is oldest)."""
    paths = []
    base = time.time() - n
    for i in range(n):
        p = cache_dir / f'crop-{i:04d}.jpg'
        p.write_bytes(b'x' * size)
        mtime = base + i
        os.utime(p, (mtime, mtime))
        paths.append(p)
    return paths


class TestPruneCropCache:
    def test_prunes_oldest_first_down_to_target_fraction(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / 'crops'
        cache_dir.mkdir()
        paths = _write_fake_crops(cache_dir, 40)
        max_bytes = 2 * 1024 * 1024  # 2 MiB

        removed_files, removed_bytes = prune_crop_cache(cache_dir, max_bytes)

        assert removed_files > 0
        assert removed_bytes > 0
        remaining = list(cache_dir.glob('*.jpg'))
        total_remaining = sum(p.stat().st_size for p in remaining)
        assert total_remaining <= int(max_bytes * 0.9)

        # The 10 newest crops must all survive.
        newest_names = {p.name for p in paths[-10:]}
        remaining_names = {p.name for p in remaining}
        assert newest_names.issubset(remaining_names)

    def test_max_bytes_zero_is_a_noop(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / 'crops'
        cache_dir.mkdir()
        _write_fake_crops(cache_dir, 10)

        removed_files, removed_bytes = prune_crop_cache(cache_dir, 0)
        assert (removed_files, removed_bytes) == (0, 0)
        assert len(list(cache_dir.glob('*.jpg'))) == 10

    def test_under_cap_is_a_noop(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / 'crops'
        cache_dir.mkdir()
        _write_fake_crops(cache_dir, 5)

        removed_files, removed_bytes = prune_crop_cache(cache_dir, 100 * 1024 * 1024)
        assert (removed_files, removed_bytes) == (0, 0)

    def test_eviction_counter_increments(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / 'crops'
        cache_dir.mkdir()
        _write_fake_crops(cache_dir, 40)

        from src.services.curation.metrics import OP_SHM_CROP_CACHE_EVICTIONS

        before_value = OP_SHM_CROP_CACHE_EVICTIONS._value.get()  # type: ignore[attr-defined]
        removed_files, _ = prune_crop_cache(cache_dir, 2 * 1024 * 1024)
        after_value = OP_SHM_CROP_CACHE_EVICTIONS._value.get()  # type: ignore[attr-defined]
        assert removed_files > 0
        assert after_value == before_value + removed_files

    def test_skips_young_tmp_files(self, tmp_path: Path) -> None:
        """An in-flight write (a fresh .tmp.<pid> file) is never counted
        toward the cap or deleted, even if it would otherwise be the
        oldest thing in the directory by name."""
        cache_dir = tmp_path / 'crops'
        cache_dir.mkdir()
        tmp = cache_dir / 'crop-x.jpg.tmp.123'
        tmp.write_bytes(b'x' * _CROP_BYTES)

        removed_files, removed_bytes = prune_crop_cache(cache_dir, 1)
        assert (removed_files, removed_bytes) == (0, 0)
        assert tmp.exists()
