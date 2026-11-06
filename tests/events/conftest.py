from __future__ import annotations

import pytest
from _events_support import FakeBroker, FakeSyncRedis

from src.tasks.runtime import event_wakeup


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> FakeBroker:
    """A fake Valkey that every ``wake`` in the test publishes to."""
    fake = FakeBroker()
    monkeypatch.setattr(event_wakeup, '_client', lambda: FakeSyncRedis(fake))
    return fake


@pytest.fixture(scope='module')
def pg(pg_events):
    """The cursor waits for the cluster's oldest open transaction: use a server of its own."""
    return pg_events
