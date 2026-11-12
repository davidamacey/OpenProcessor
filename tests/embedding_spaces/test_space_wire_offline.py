"""The wire of the space list, with the registry replaced by fakes.

The SQL behind them is proven against a real PostgreSQL in ``test_space_routes.py``; this file pins
what the routes do with what the services return: the run fields on the list, the evaluation
summary, and the 404 of an unknown space.
"""

from __future__ import annotations

import datetime as dt
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from _curation_app import SCOPED, mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.db_session import get_db
from src.domain.embedding_spaces import Role, Status, Target
from src.routers.curation import router as curation_router
from src.routers.curation.embedding_spaces import _index
from src.services.embedding_spaces import spaces, views
from src.services.embedding_spaces.reindex_runs import LatestRun


NOW = dt.datetime(2026, 11, 12, 12, 0, tzinfo=dt.UTC)
RUN = uuid.uuid4()


def _space(key: str, role: Role, status: Status) -> Any:
    return SimpleNamespace(
        id=uuid.uuid4(),
        space_key=key,
        target=Target.ITEM,
        dim=512,
        role=role,
        status=status,
        ready_at=NOW,
    )


MODEL: Any = SimpleNamespace(model_key='mobileclip2_s2', version=1)
PRIMARY = _space('mobileclip_v1', Role.PRIMARY, Status.READY)
CANDIDATE = _space('pe_core_v1', Role.CANDIDATE, Status.BUILDING)


class _Index:
    async def count_with_vector(self, _space: Any) -> int:
        return 7


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    async def list_project_spaces(_s: Any, _p: Any, _t: Any) -> list[Any]:
        return [PRIMARY, CANDIDATE]

    async def with_models(_s: Any, _p: Any, rows: list[Any]) -> list[views.SpaceWithModel]:
        return [views.SpaceWithModel(r, MODEL) for r in rows]

    monkeypatch.setattr('src.routers.curation.embedding_spaces.current_project_id', uuid.uuid4)
    monkeypatch.setattr(spaces, 'list_project_spaces', list_project_spaces)
    monkeypatch.setattr(views, 'with_models', with_models)
    app = FastAPI()
    mount_curation_routers(app, curation_router)

    async def no_session():
        yield object()

    app.dependency_overrides[get_db] = no_session
    app.dependency_overrides[_index] = _Index
    return TestClient(app)


def _latest(monkeypatch: pytest.MonkeyPatch, found: dict[uuid.UUID, LatestRun]) -> None:
    async def latest_runs(_s: Any, _p: Any, _rows: Any) -> dict[uuid.UUID, LatestRun]:
        return found

    monkeypatch.setattr(views, 'latest_runs', latest_runs)


def test_the_list_names_the_run_of_a_building_space_and_its_state(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _latest(monkeypatch, {CANDIDATE.id: LatestRun(RUN, 'build', 'running')})
    listed = {s['space_key']: s for s in client.get(f'{SCOPED}/embedding/spaces').json()['spaces']}
    assert listed['pe_core_v1']['run_id'] == str(RUN)
    assert listed['pe_core_v1']['run_mode'] == 'build'
    assert listed['pe_core_v1']['run_status'] == 'running'


def test_a_space_that_was_never_built_has_no_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _latest(monkeypatch, {})
    primary = client.get(f'{SCOPED}/embedding/spaces').json()['spaces'][0]
    assert (primary['run_id'], primary['run_mode'], primary['run_status']) == (None, None, None)
