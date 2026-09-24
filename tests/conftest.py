"""Pytest configuration for the test suite.

Some files under ``tests/`` are standalone smoke-test *scripts* meant to be
run against a live deployment (``python tests/test_full_system.py``), not
pytest suites. They define helper functions named ``test_*(name, ...)`` that
pytest would otherwise miscollect as test cases — failing with
``fixture 'name' not found``. Exclude them from collection here; run them
directly per the project README / CLAUDE.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import Iterator


collect_ignore = [
    'test_full_system.py',
    'test_scrfd_pipeline.py',
    'test_validate_models.py',
    'validate_visual_results.py',
]


@pytest.fixture
def reference_region_profile(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Activate the built-in ``license_plate`` reference region profile.

    The neutral default has no region profile, so the detection worker's
    cascade refuses to run. Tests that exercise the cascade opt in to the
    reference profile the same way a deployment does: ``OP_REGION_PROFILE``.
    """
    from src.services.detection import profile_registry

    monkeypatch.setenv('OP_REGION_PROFILE', 'license_plate')
    profile_registry._reset_registry_for_tests()
    yield
    # Lazy re-resolution: the next accessor call (after monkeypatch restores
    # the env) sees the unconfigured default again.
    profile_registry._reset_registry_for_tests()
