"""Fixtures for the curation train router tests: an app over a mocked OpenSearch and fake run seams.

The database-backed run service (start, status, queue) and the broker probe are replaced by the
recording fakes of ``train_fake``; their real behaviour is tested against PostgreSQL in
``tests/training``."""

from __future__ import annotations

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

    # claim_gpus_for_training (called unconditionally by /start and
    # /start_campaign before the run is queued) writes a sentinel/lock
    # file under CurationConfig.state_dir. get_curation_config() is a
    # process-wide cached singleton, so setting OP_STATE_DIR here would
    # have no effect once the process has already built it once -- patch
    # the arbiter's own _state_dir() seam instead (same pattern
    # test_gpu_arbiter.py uses) so these tests exercise a real,
    # succeeding claim instead of relying on an exception being silently
    # swallowed by a broad except-Exception (now removed).
    from src.services.training import gpu_arbiter_policy

    monkeypatch.setattr(gpu_arbiter_policy, '_state_dir', lambda: tmp_path)

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
