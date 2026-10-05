"""#127: the arbiter reconcile loop must not clear a pause sentinel it did not create.

`openprocessor vlm use` pauses the workers for the 3-4 minute model load; the
arbiter's idle-path ``resume_gpu_worker`` used to unlink the file within seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from src.services.training import gpu_arbiter as ga


REPO_ROOT = Path(__file__).resolve().parents[2]
SWITCH_SH = REPO_ROOT / 'scripts' / 'lib' / 'vlm_switch.sh'


def _cli_sentinel(state: Path) -> Path:
    """Create the sentinel with the real `_vlm_in_api pause-create` script (the
    container exec is replaced by a local sh)."""
    env = {**os.environ, 'OP_STATE_DIR': str(state)}
    prog = f'dc() {{ shift 3; "$@"; }}; source "{SWITCH_SH}"; _vlm_in_api "$1"'
    created = subprocess.run(
        ['bash', '-c', prog, '_', 'pause-create'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    assert created == 'created'
    return state / 'vlm_worker' / 'pause.sentinel'


def _remove_via_cli(state: Path) -> None:
    env = {**os.environ, 'OP_STATE_DIR': str(state)}
    prog = f'dc() {{ shift 3; "$@"; }}; source "{SWITCH_SH}"; _vlm_in_api pause-remove'
    subprocess.run(['bash', '-c', prog], env=env, check=True, timeout=30)


@pytest.mark.asyncio
async def test_reconcile_with_no_active_run_keeps_the_cli_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = _cli_sentinel(tmp_path)
    monkeypatch.setattr(ga, '_docker_client', lambda: None)
    jobs = tmp_path / 'jobs'
    jobs.mkdir()
    await ga.reconcile_on_startup(train_jobs_dir=jobs, sentinel=sentinel)
    assert sentinel.exists()


@pytest.mark.asyncio
async def test_reconcile_still_clears_its_own_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / 'pause.sentinel'
    await ga.pause_gpu_worker(sentinel=sentinel)
    assert json.loads(sentinel.read_text())['owner'] == 'arbiter'
    monkeypatch.setattr(ga, '_docker_client', lambda: None)
    await ga.reconcile_on_startup(train_jobs_dir=tmp_path / 'none', sentinel=sentinel)
    assert not sentinel.exists()


@pytest.mark.asyncio
async def test_a_legacy_empty_sentinel_is_still_the_arbiters_to_clear(tmp_path: Path) -> None:
    sentinel = tmp_path / 'pause.sentinel'
    sentinel.touch()
    await ga.resume_gpu_worker(sentinel=sentinel)
    assert not sentinel.exists()


@pytest.mark.asyncio
async def test_a_stale_foreign_sentinel_is_cleared_so_an_orphan_cannot_pause_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = _cli_sentinel(tmp_path)
    monkeypatch.setenv('OP_PAUSE_SENTINEL_TTL_S', '60')
    old = time.time() - 3600
    os.utime(sentinel, (old, old))
    res = await ga.resume_gpu_worker(sentinel=sentinel)
    assert res.action == 'sentinel_cleared'
    assert not sentinel.exists()


@pytest.mark.asyncio
async def test_a_fresh_foreign_sentinel_survives_resume(tmp_path: Path) -> None:
    sentinel = _cli_sentinel(tmp_path)
    res = await ga.resume_gpu_worker(sentinel=sentinel)
    assert res.action == 'noop'
    assert sentinel.exists()


def test_the_cli_undo_path_still_removes_its_sentinel(tmp_path: Path) -> None:
    sentinel = _cli_sentinel(tmp_path)
    _remove_via_cli(tmp_path)
    assert not sentinel.exists()
