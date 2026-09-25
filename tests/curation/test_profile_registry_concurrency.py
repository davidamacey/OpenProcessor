"""A configured region profile is never reported missing while it resolves.

Sync routes run in a thread pool, so a worker's first profile lookups can
arrive on several threads at once. A lookup that lands mid-resolution must
wait for the profile, not see "no region profile" and answer 409.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from src.config import DetectionProfile
from src.services.detection import profile_registry


if TYPE_CHECKING:
    import pytest


def test_lookup_during_registration_waits_for_the_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_registry._reset_registry_for_tests()
    entered = threading.Event()
    real_register = profile_registry.register_profile

    def slow_register(profile: DetectionProfile, *, default: bool = False) -> None:
        # Hold the thread between "resolved" and "registered", where a
        # thread switch let another request see no profile.
        entered.set()
        time.sleep(0.2)
        real_register(profile, default=default)

    monkeypatch.setattr(
        profile_registry,
        'region_profile_from_env',
        lambda: DetectionProfile(name='region', display_name='Regions'),
    )
    monkeypatch.setattr(profile_registry, 'register_profile', slow_register)
    try:
        first = threading.Thread(target=profile_registry.get_active_region_profile)
        first.start()
        assert entered.wait(1.0)

        seen = profile_registry.get_active_region_profile()
        first.join()

        assert seen is not None
        assert seen.name == 'region'
    finally:
        profile_registry._reset_registry_for_tests()
