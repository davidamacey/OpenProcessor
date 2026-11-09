from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from .world import World


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    return World(tmp_path, monkeypatch)
