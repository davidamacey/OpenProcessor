"""Proves the FastAPI lifespan actually runs the GPU arbiter.

:func:`src.services.training.gpu_arbiter.reconcile_on_startup` is the only
thing that ever releases a training run's GPU claim (pause sentinel set,
GPU-resident containers stopped) when the trainer dies mid-run instead of
calling ``release_gpus_after_training``. It is not self-starting: unless
``src.main``'s lifespan reconciles once at startup *and* keeps a periodic
task re-reconciling, a crashed trainer leaves the claim in place forever.

These tests cover the *wiring*, which is exactly what was missing — the
arbiter's own decision logic is covered by
``tests/curation/test_gpu_arbiter.py``. The first test stubs the reconcile
call to prove it is invoked at startup and then re-invoked on a timer; the
second lets the real implementation run against a tmp jobs dir and proves
a sentinel that appears *after* startup is cleared by a later tick (i.e.
the loop genuinely ticks, a startup-only call would leave it behind).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.integration


# Fast enough to observe several ticks inside a test, slow enough not to
# spin the event loop into a busy-wait.
_TEST_INTERVAL_S = 0.05


def _wait_until(predicate, timeout: float = 5.0, poll: float = 0.02) -> bool:
    """Poll ``predicate`` from the test thread while the app's loop runs."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


def test_lifespan_reconciles_at_startup_and_keeps_reconciling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup calls the arbiter once, then a background task re-calls it."""
    import src.main as main_module
    from src.services.training import gpu_arbiter

    calls: list[dict] = []

    async def _fake_reconcile(**kwargs):
        calls.append(kwargs)
        return gpu_arbiter.ArbiterAction(action='noop', detail='stubbed')

    monkeypatch.setattr(gpu_arbiter, 'reconcile_on_startup', _fake_reconcile)
    monkeypatch.setattr(main_module, 'ARBITER_RECONCILE_INTERVAL_SECONDS', _TEST_INTERVAL_S)

    with TestClient(main_module.app):
        # Reconciled once during startup, before the app served anything.
        assert len(calls) >= 1

        task = main_module.AppResources.arbiter_task
        assert task is not None, 'lifespan did not schedule the arbiter reconcile loop'
        assert not task.done(), 'arbiter reconcile loop exited immediately'

        # ... and keeps reconciling on its own, which is what makes it a
        # backstop for a trainer that died without releasing its claim.
        assert _wait_until(lambda: len(calls) >= 3), f'loop did not tick (calls={len(calls)})'

    # Shutdown cancels the task and clears the slot, so a second app
    # lifecycle in the same process starts clean.
    assert task.done()
    assert main_module.AppResources.arbiter_task is None


def test_periodic_tick_clears_a_stale_pause_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sentinel left behind mid-run is cleared by the loop, not just startup.

    Simulates the real failure: a trainer claimed a GPU (pause sentinel
    set) and then died without releasing it. The sentinel is created
    *after* the app has started, so only a genuinely periodic reconcile
    can clear it.
    """
    import src.main as main_module
    from src.services.training import gpu_arbiter

    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    sentinel = tmp_path / 'pause.sentinel'
    lock = tmp_path / 'training_gpus.lock'

    # Keep the real reconcile logic, but off any real /jobs mount or state dir.
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(jobs_dir))
    monkeypatch.setattr(gpu_arbiter, '_default_sentinel_path', lambda: sentinel)
    monkeypatch.setattr(gpu_arbiter, '_default_lock_path', lambda: lock)
    monkeypatch.setattr(main_module, 'ARBITER_RECONCILE_INTERVAL_SECONDS', _TEST_INTERVAL_S)

    with TestClient(main_module.app):
        assert main_module.AppResources.arbiter_task is not None

        # Orphaned claim appears after startup: no job.json, no lock, but
        # the worker is still paused.
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
        assert sentinel.exists()

        assert _wait_until(lambda: not sentinel.exists()), (
            'stale pause sentinel survived — the reconcile loop is not running'
        )


def test_lifespan_survives_a_failing_arbiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """An arbiter that blows up must not take the whole service down."""
    import src.main as main_module
    from src.services.training import gpu_arbiter

    async def _boom(**_kwargs):
        raise RuntimeError('simulated arbiter failure')

    monkeypatch.setattr(gpu_arbiter, 'reconcile_on_startup', _boom)

    with TestClient(main_module.app) as client:
        assert client.get('/live').status_code == 200

    assert main_module.AppResources.arbiter_task is None
