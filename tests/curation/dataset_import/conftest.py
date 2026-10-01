from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_region_profile_registry() -> Iterator[None]:
    """``activate_region_profile`` registers into the process-wide profile
    registry; leaving it set makes every later test on the worker see an
    active region profile (the no-profile gating tests then fail)."""
    yield
    from src.services.detection import profile_registry

    profile_registry._reset_registry_for_tests()
