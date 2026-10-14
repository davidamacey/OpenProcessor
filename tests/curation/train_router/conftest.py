"""Fixtures for the curation_train router tests: an app over a mocked OpenSearch and a temp jobs volume."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


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
def app_client(
    fake_opensearch: AsyncMock,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """Build a minimal FastAPI app with just the train router."""
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))
    # P1R §6.1/D-A: project_jobs_dir() always nests /projects/<slug>,
    # `default` (the project every unit test binds) included.
    (tmp_path / 'projects' / 'default').mkdir(parents=True, exist_ok=True)

    # claim_gpus_for_training (called unconditionally by /start and
    # /start_campaign before job.json is written) writes a sentinel/lock
    # file under CurationConfig.state_dir. get_curation_config() is a
    # process-wide cached singleton, so setting OP_STATE_DIR here would
    # have no effect once the process has already built it once -- patch
    # the arbiter's own _state_dir() seam instead (same pattern
    # test_gpu_arbiter.py uses) so these tests exercise a real,
    # succeeding claim instead of relying on an exception being silently
    # swallowed by a broad except-Exception (now removed).
    from src.services.training import gpu_arbiter as _gpu_arbiter

    monkeypatch.setattr(_gpu_arbiter, '_state_dir', lambda: tmp_path)

    # Stub the registry so preflight class-resolution doesn't blow up.
    class _Reg:
        def load(self) -> Any:
            class _Snap:
                classes: list[Any] = []

            return _Snap()

        def get(self, _cid: int) -> Any:
            return None

    monkeypatch.setattr('src.services.training.preflight_run.get_class_registry', _Reg)
    monkeypatch.setattr('src.services.training.preflight_checks.get_class_registry', _Reg)

    from src.routers.curation._common import _raw_opensearch_dep
    from src.routers.curation_train import router as curation_train_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_train_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch

    with TestClient(app) as client:
        yield client
