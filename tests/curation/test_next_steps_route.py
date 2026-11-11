"""`GET /next_steps` through the app: the route serves what the service builds from the facts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation.class_counts import ClassCounts
from src.services.curation.workflow_steps import WorkflowFacts


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pytest

URL = '/curation/projects/default/next_steps'


def _client(monkeypatch: pytest.MonkeyPatch, facts: WorkflowFacts) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.core.db_session import get_db
    from src.routers.curation import _raw_opensearch_dep, router as curation_router, workflow_steps

    registry = SimpleNamespace(snapshot=AsyncMock(return_value=SimpleNamespace(classes=[])))
    monkeypatch.setattr(workflow_steps, 'get_class_registry', lambda: registry)
    monkeypatch.setattr(
        workflow_steps, 'fetch_class_counts', AsyncMock(return_value=(ClassCounts(), {}))
    )
    monkeypatch.setattr(workflow_steps, 'current_project_id', lambda: 'project-id')
    gather = AsyncMock(return_value=facts)
    monkeypatch.setattr(workflow_steps, 'gather_facts', gather)

    async def no_database() -> AsyncIterator[None]:
        yield None

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: object()
    app.dependency_overrides[get_db] = no_database
    return TestClient(app)


def test_a_new_project_is_told_to_ingest_and_opens_on_the_gallery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _client(monkeypatch, WorkflowFacts()).get(URL).json()
    assert body['next']['id'] == 'ingest'
    assert body['next']['section'] == 'ingest'
    assert body['workspace_default_view'] == 'gallery'
    assert body['steps'][0]['id'] == 'ingest'


def test_a_project_with_a_serving_clustering_opens_on_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = WorkflowFacts(images=5, item_partitions=1, item_partition_serving=True)
    body = _client(monkeypatch, facts).get(URL).json()
    assert body['workspace_default_view'] == 'clusters'
    assert {s['id']: s['done'] for s in body['steps']}['activate'] is True
