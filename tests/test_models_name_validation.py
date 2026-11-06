"""Model lifecycle routes validate the model name and keep every target inside
the model repository."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.routers import models as models_router, models_lifecycle
from src.services.model_repository import inside_repository


if TYPE_CHECKING:
    from pathlib import Path


class _Triton:
    async def unload_model(self, _name: str) -> tuple[bool, str]:
        return True, ''

    async def get_repository_index(self) -> list[dict[str, str]]:
        return []


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    models = tmp_path / 'repo' / 'models'
    models.mkdir(parents=True)
    (tmp_path / 'repo' / 'pytorch_models').mkdir()
    (tmp_path / 'repo' / 'keep.txt').write_text('x')
    (models / 'good_trt').mkdir()
    (models / 'good_trt' / 'config.pbtxt').write_text('x')
    (models / 'other_trt').mkdir()
    (models / 'other_trt' / 'config.pbtxt').write_text('x')
    monkeypatch.setattr(models_lifecycle, 'TRITON_MODELS_DIR', models)
    monkeypatch.setattr(
        models_lifecycle, 'PYTORCH_MODELS_DIR', tmp_path / 'repo' / 'pytorch_models'
    )
    monkeypatch.setattr(models_lifecycle, 'TritonControlService', _Triton)

    async def _no_other_projects() -> dict[str, frozenset[str]]:
        return {}

    monkeypatch.setattr(
        'src.services.model_unload_guard.ingest_models_by_project', _no_other_projects
    )
    return tmp_path / 'repo'


@pytest.fixture
def client(repo: Path) -> TestClient:  # repo patches the module first
    app = FastAPI()
    app.include_router(models_router.router)
    return TestClient(app)


BAD_NAMES = ['..', '.', '', 'a/b', '../models', '../../etc', 'UPPER', 'a b', '-x', 'a.b']


@pytest.mark.parametrize('name', BAD_NAMES)
def test_delete_refuses_bad_names_and_deletes_nothing(repo: Path, name: str) -> None:
    with pytest.raises(HTTPException) as exc:
        asyncio.run(models_lifecycle.delete_model(name, force=True))
    assert exc.value.status_code == 400
    assert (repo / 'keep.txt').exists()
    assert sorted(p.name for p in (repo / 'models').iterdir()) == ['good_trt', 'other_trt']


@pytest.mark.parametrize('name', BAD_NAMES)
def test_other_lifecycle_routes_refuse_bad_names(repo: Path, name: str) -> None:
    calls = [
        models_lifecycle.get_model_status(name),
        models_lifecycle.load_model(name),
        models_lifecycle.unload_model(name, force=True),
    ]
    for call in calls:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(call)
        assert exc.value.status_code == 400
    assert (repo / 'keep.txt').exists()


def test_normal_delete_removes_only_that_models_directories(client: TestClient, repo: Path) -> None:
    (repo / 'pytorch_models' / 'good.pt').write_text('x')
    resp = client.delete(f'{models_router.router.prefix}/good')
    assert resp.status_code == 200, resp.text
    assert sorted(p.name for p in (repo / 'models').iterdir()) == ['other_trt']
    assert not (repo / 'pytorch_models' / 'good.pt').exists()
    assert (repo / 'keep.txt').exists()


def test_inside_repository_rejects_escapes_and_the_root(tmp_path: Path) -> None:
    base = tmp_path / 'models'
    base.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (base / 'link').symlink_to(outside)
    (base / 'real').mkdir()
    assert inside_repository(base, base / 'real')
    assert not inside_repository(base, base)
    assert not inside_repository(base, base / '..')
    assert not inside_repository(base, base / 'link')
