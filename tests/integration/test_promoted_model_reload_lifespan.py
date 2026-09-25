"""Proves the FastAPI lifespan reloads promoted Triton models both at
startup and on every periodic reconcile tick.

Item 3 (fresh-start E2E findings 2026-09-25, round 2): Triton in
explicit-control mode only loads its ``--load-model`` list at startup, so
a bare Triton restart (or any ``docker compose restart``/recreate of the
Triton service) silently strands every previously-promoted model at
UNAVAILABLE. ``src.services.training.triton_promote.reload_promoted_models``
already re-loads them by scanning for ``promote.json`` markers; this test
covers the *wiring* -- that ``src.main``'s lifespan actually calls it once
at startup and keeps calling it on the same periodic tick the GPU arbiter
uses, so a Triton bounce between API restarts is still picked up without
an API bounce of its own. The reload function's own decision logic
(skip-if-ready, ignore non-promoted dirs, best-effort on an unreachable
Triton) is covered by ``tests/curation/test_triton_promote.py``.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration

# Fast enough to observe several ticks inside a test, slow enough not to
# spin the event loop into a busy-wait.
_TEST_INTERVAL_S = 0.05


def _wait_until(predicate, timeout: float = 5.0, poll: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


def test_lifespan_reloads_promoted_models_at_startup_and_on_every_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.main as main_module
    from src.services.training import gpu_arbiter, triton_promote

    calls: list[str] = []

    async def _fake_reconcile(**_kwargs):
        return gpu_arbiter.ArbiterAction(action='noop', detail='stubbed')

    async def _fake_reload(*_args, **_kwargs):
        calls.append('tick')
        return {'status': 'ok', 'reloaded': [], 'failed': []}

    monkeypatch.setattr(gpu_arbiter, 'reconcile_on_startup', _fake_reconcile)
    monkeypatch.setattr(triton_promote, 'reload_promoted_models', _fake_reload)
    monkeypatch.setattr(main_module, 'ARBITER_RECONCILE_INTERVAL_SECONDS', _TEST_INTERVAL_S)

    with TestClient(main_module.app):
        # Once at startup, before the app serves anything.
        assert len(calls) >= 1

        # ...and again on the periodic tick, without any explicit trigger.
        assert _wait_until(lambda: len(calls) >= 3), f'loop did not re-reload (calls={calls})'


def test_a_failing_reload_does_not_kill_the_reconcile_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan failure or unreachable Triton must never take the arbiter
    loop down -- the next tick (GPU reconcile and the next reload
    attempt) still has to run."""
    import src.main as main_module
    from src.services.training import gpu_arbiter, triton_promote

    arbiter_calls: list[str] = []

    async def _fake_reconcile(**_kwargs):
        arbiter_calls.append('tick')
        return gpu_arbiter.ArbiterAction(action='noop', detail='stubbed')

    async def _boom(*_args, **_kwargs):
        raise RuntimeError('simulated reload failure')

    monkeypatch.setattr(gpu_arbiter, 'reconcile_on_startup', _fake_reconcile)
    monkeypatch.setattr(triton_promote, 'reload_promoted_models', _boom)
    monkeypatch.setattr(main_module, 'ARBITER_RECONCILE_INTERVAL_SECONDS', _TEST_INTERVAL_S)

    with TestClient(main_module.app) as client:
        assert _wait_until(lambda: len(arbiter_calls) >= 3), (
            f'arbiter reconcile loop died after a failing reload (ticks={arbiter_calls})'
        )
        assert client.get('/live').status_code == 200
