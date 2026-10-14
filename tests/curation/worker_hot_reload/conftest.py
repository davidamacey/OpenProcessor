"""Reset the config-store singletons around every hot-reload test."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.services.config_store.store import reset_config_stores


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    reset_config_stores()
    yield
    reset_config_stores()
