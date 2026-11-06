"""Models are deployment-wide, so unloading or deleting one is refused while ANY
project's active ingest profile runs it, and ``force`` never overrides that."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.app.factory import create_app
from src.core.dependencies import get_curation_opensearch
from src.services import model_unload_guard
from src.services.model_unload_guard import (
    UnloadRefusedError,
    check_unload,
    is_region_protected_model,
)


def _projects_using(monkeypatch: pytest.MonkeyPatch, usage: dict[str, set[str]]) -> None:
    async def _scan() -> dict[str, frozenset[str]]:
        return {slug: frozenset(models) for slug, models in usage.items()}

    monkeypatch.setattr(model_unload_guard, 'ingest_models_by_project', _scan)


@pytest.mark.asyncio
@pytest.mark.parametrize('force', [False, True])
async def test_a_model_another_projects_ingest_profile_runs_is_refused(
    monkeypatch: pytest.MonkeyPatch, force: bool
) -> None:
    _projects_using(monkeypatch, {'alpha': {'birds'}, 'beta': set()})
    with pytest.raises(UnloadRefusedError) as info:
        await check_unload('birds', force=force, own_project='beta')
    assert info.value.status_code == 403
    assert 'alpha' in info.value.detail


@pytest.mark.asyncio
async def test_the_callers_own_ingest_model_is_left_to_the_routes_own_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _projects_using(monkeypatch, {'alpha': {'birds'}})
    assert await check_unload('birds', force=False, own_project='alpha') is False


@pytest.mark.asyncio
async def test_an_unreadable_project_refuses_instead_of_allowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom() -> dict[str, frozenset[str]]:
        msg = 'configs unreadable'
        raise ConnectionError(msg)

    monkeypatch.setattr(model_unload_guard, 'ingest_models_by_project', _boom)
    with pytest.raises(UnloadRefusedError) as info:
        await check_unload('birds', force=True)
    assert info.value.status_code == 503


def test_the_listing_flag_marks_a_model_another_project_runs() -> None:
    assert is_region_protected_model('birds', used_elsewhere=frozenset({'birds'})) is True
    assert is_region_protected_model('birds') is False


@pytest.mark.parametrize(
    ('method', 'path'), [('DELETE', '/models/birds'), ('POST', '/models/birds/unload')]
)
@pytest.mark.parametrize('force', [False, True])
def test_the_public_routes_refuse_it_and_remove_nothing(
    method: str,
    path: str,
    force: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from src.routers import models_lifecycle as public

    _projects_using(monkeypatch, {'alpha': {'birds'}})
    unload = AsyncMock(return_value=(True, ''))
    monkeypatch.setattr(public.TritonControlService, 'unload_model', unload)
    monkeypatch.setattr(public, 'TRITON_MODELS_DIR', tmp_path / 'models')
    (tmp_path / 'models' / 'birds').mkdir(parents=True)
    app = create_app()
    app.dependency_overrides[get_curation_opensearch] = lambda: object()  # noqa: PLW0108  # FastAPI parses a class here as a dependency with parameters
    resp = TestClient(app).request(method, path, params={'force': str(force).lower()})

    assert resp.status_code == 403, resp.text
    unload.assert_not_awaited()
    assert (tmp_path / 'models' / 'birds').exists()
