"""Fixtures for the curation train router tests: an app over a mocked OpenSearch and fake run seams.

The database-backed run service (start, status, queue) and the broker probe are replaced by the
recording fakes of ``train_fake``; their real behaviour is tested against PostgreSQL in
``tests/training``."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from _train_run_fakes import install_run_fakes
from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from types import SimpleNamespace


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_opensearch() -> AsyncMock:
    """An AsyncMock with the methods the preflight uses."""
    fake = AsyncMock()
    fake.search = AsyncMock(
        return_value={
            'hits': {'hits': [], 'total': {'value': 0}},
            'aggregations': {'by_class': {'buckets': []}},
        }
    )
    # count_pending_ingest (preflight_checks.py) does
    # `int(resp.get('count', 0))` on the awaited result — a *sync* dict
    # method call. Left unstubbed, AsyncMock auto-generates `.count` (and
    # then `.get` on its return value) as further AsyncMocks, so `resp`
    # ends up being an AsyncMock instead of a dict and `.get(...)` itself
    # returns an unawaited coroutine (TypeError: int() argument ... not
    # 'coroutine'). Stub it with a real dict so `.get` behaves like OpenSearch's
    # actual `count` API response.
    fake.count = AsyncMock(return_value={'count': 0})
    return fake


@pytest.fixture
def train_fake(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The run service and the broker probe, replaced; ``started`` records every queued run."""
    return install_run_fakes(monkeypatch)


@pytest.fixture
def app_client(
    fake_opensearch: AsyncMock,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    train_fake: SimpleNamespace,
):
    """Build a minimal FastAPI app with just the train router."""
    from curation._promote_start_fake import no_database

    no_database(monkeypatch)

    async def _no_promote(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr('src.routers.curation.train.runs.promote_job.latest_for_run', _no_promote)
    monkeypatch.setattr('src.routers.curation.train.runs.current_project_id', uuid.uuid4)

    # The preflight asks the database whether gpu-arbiter-worker is alive; these tests have none.
    # A test that cares overrides it.
    monkeypatch.setattr(
        'src.services.training.preflight_run.arbiter_consumer_live', AsyncMock(return_value=True)
    )

    from src.routers.curation._common import _raw_opensearch_dep
    from src.routers.curation.train import router as curation_train_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_train_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch

    from src.core.db_session import get_db

    async def no_session():
        yield None

    app.dependency_overrides[get_db] = no_session

    with TestClient(app) as client:
        yield client
