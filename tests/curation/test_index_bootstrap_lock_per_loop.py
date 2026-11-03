"""The cold-start bootstrap lock must work from any event loop.

A process can run the bootstrap under more than one event loop (a script that
calls ``asyncio.run`` per pass, the ingest upload driver's tests). An
``asyncio.Lock`` binds to the first loop that has to wait on it, so a single
module-level lock fails with "bound to a different event loop" once a later
loop contends for it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pytest


async def _contend(index_bootstrap: Any) -> None:
    # Two requests arrive while the project is still unbootstrapped; the first
    # holds the lock across a real suspension, so the second has to wait on it.
    await asyncio.gather(index_bootstrap.ensure_indexes(None), index_bootstrap.ensure_indexes(None))


def test_bootstrap_lock_survives_a_second_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation import index_bootstrap

    async def _slow_failing_bootstrap(_opensearch: Any) -> None:
        await asyncio.sleep(0.01)  # leaves the project unbootstrapped, as a failure does

    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', set())
    monkeypatch.setattr(index_bootstrap, '_ensure_indexes_locked', _slow_failing_bootstrap)

    asyncio.run(_contend(index_bootstrap))
    asyncio.run(_contend(index_bootstrap))
