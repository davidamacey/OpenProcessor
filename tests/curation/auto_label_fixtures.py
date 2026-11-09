"""Shared fixtures for the auto-label route and job tests.

Registered as a plugin in ``tests/conftest.py`` (pytest ``pytest_plugins``,
https://docs.pytest.org/en/stable/how-to/writing_plugins.html) so test modules
use them without importing fixtures from each other.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_pipeline import install_registry
from src.config.curation import CurationConfig
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def packs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CurationConfig:
    """Default pack ``pallet_v1`` + extra selectable pack ``food_v2``."""
    paths = {}
    for name in ('pallet_v1', 'food_v2'):
        data = GENERIC_ITEM_PACK.to_dict()
        data['name'] = name
        paths[name] = tmp_path / f'{name}.json'
        paths[name].write_text(json.dumps(data))
    cfg = CurationConfig(prompt_pack_path=paths['pallet_v1'], prompt_pack_paths=(paths['food_v2'],))
    monkeypatch.setattr('src.config.curation_accessors.get_curation_config', lambda: cfg)
    return cfg


@pytest.fixture
def job_dir(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The start route's job service replaced by a recorder (these tests check what the route
    resolves and passes, not the database); the recorded list is the params of each start."""
    from src.services.curation.autolabel import service

    started: list[dict[str, Any]] = []

    async def start(_session: Any, params: Any, *, idempotency_key: str | None = None) -> Any:
        started.append(dict(params))
        return {'job_id': 'job-1', 'status': 'queued', 'args': dict(params)}

    monkeypatch.setattr(service, 'start', start)
    return started


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = AsyncMock()
    # No settings doc -> every axis resolves to its process default.
    fake_os.get = AsyncMock(side_effect=RuntimeError('no settings index'))
    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    from src.core.db_session import get_db

    app.dependency_overrides[get_db] = lambda: None
    return TestClient(app)


@pytest.fixture
def labeler_spy(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every labeler the pipeline obtains (real instances)."""
    from src.routers.curation import vlm
    from src.services.curation.autolabel import vlm_stage

    got: list[Any] = []

    def _spy(pack_name: str | None = None, revision: int | None = None, **kw: Any) -> Any:
        inst = vlm.get_vlm_labeler(pack_name, revision, **kw)
        got.append(inst)
        return inst

    monkeypatch.setattr(vlm_stage, 'get_vlm_labeler', _spy)
    install_registry((3, 'wooden_pallet'))
    return got
