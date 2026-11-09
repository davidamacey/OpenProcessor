"""``op.periodic.cache_prune``: evict the oldest crops over the cap, once at a time.

Replaces the ``.prune.lock`` file lock and the prune call inside ingest: the deployment-wide
advisory lock ``CACHE_PRUNE`` is the only guard, and a tick that cannot take it deletes nothing.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import psycopg
import pytest

from src.config.advisory_locks import CACHE_PRUNE
from src.services.periodic import cache_prune
from src.services.periodic.runner import run_periodic


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_env')]

CROP = 100 * 1024


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / 'crops'
    (root / 'proj').mkdir(parents=True)
    base = time.time() - 100
    for i in range(40):
        path = root / 'proj' / f'crop-{i:03d}.jpg'
        path.write_bytes(b'x' * CROP)
        os.utime(path, (base + i, base + i))
    config = SimpleNamespace(crop_cache_dir=root, crop_cache_max_bytes=2 * 1024 * 1024)
    monkeypatch.setattr(cache_prune, 'base_curation_config', lambda: config)
    return root


def files(root: Path) -> list[str]:
    return sorted(p.name for p in (root / 'proj').glob('*.jpg'))


def test_the_oldest_crops_go_until_the_cache_is_under_the_cap(pg, cache) -> None:
    removed, freed = run_periodic('cache_prune', cache_prune.tick())

    assert removed > 0
    assert freed == removed * CROP
    left = files(cache)
    assert len(left) == 40 - removed
    assert left[-1] == 'crop-039.jpg'  # the newest survive
    assert 'crop-000.jpg' not in left


def test_a_tick_that_cannot_take_the_lock_deletes_nothing(pg, cache) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as holder:
        holder.execute('SELECT pg_advisory_lock(%s, 0)', (CACHE_PRUNE,))
        assert run_periodic('cache_prune', cache_prune.tick()) == (0, 0)
        holder.execute('SELECT pg_advisory_unlock(%s, 0)', (CACHE_PRUNE,))
    assert len(files(cache)) == 40
