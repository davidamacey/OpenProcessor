"""Fixtures for the bake-off harness tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_profile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith('OP_BAKEOFF_PROFILE'):
            monkeypatch.delenv(key)
