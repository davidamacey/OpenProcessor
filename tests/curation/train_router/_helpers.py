"""Helpers shared by the curation train router tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pytest


def _set_arbiter_config(monkeypatch: pytest.MonkeyPatch, cfg: Any) -> None:
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(gpu_arbiter_config_module, '_default_gpu_arbiter_config', cfg)
