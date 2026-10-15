"""A test double for "this project opted in to the deployment's env VLM".

The activation doc is the real thing, but many tests run on fake storage that
cannot hold it. Every project snapshot without an activation then reads as
having activated the ``env`` endpoint."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pytest


def project_opts_in_to_env_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.labeling import vlm_endpoints

    real = vlm_endpoints._project

    def opted_in() -> Any:
        snapshot = real()
        if snapshot.active_vlm is None:
            return dataclasses.replace(snapshot, active_vlm=('env', None))
        return snapshot

    monkeypatch.setattr(vlm_endpoints, '_project', opted_in)
