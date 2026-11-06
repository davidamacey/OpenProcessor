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
