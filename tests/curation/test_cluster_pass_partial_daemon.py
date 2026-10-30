"""The refresh daemon's assign-only pass is ``partial`` when it did not cover its pool (#357).

The daemon assigns residuals against the persisted centroids. Another writer (ingest
assigning at write time) can take items out of the pool between the moment the pass counts
it and the moment it reads it; a pass that wrote a fraction of the pool must not report
``success``, and a failed write-back must not either.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import src.services.curation.clustering.orchestrator as orch
from src.services.curation.clustering.pool_coverage import POOL_COVERAGE_MIN
from tests.curation._fake_items_pit import FakeItemsClient


pytestmark = pytest.mark.asyncio

N_DOCS = 3000


class _TakenFromPool(FakeItemsClient):
    """``count`` answers the pool, then another writer takes ``taken`` documents out of it."""

    def __init__(self, ids: list[str], vectors: np.ndarray, taken: int) -> None:
        super().__init__(ids, vectors)
        self._to_take = taken
        self._gone: set[str] = set()

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        answer = await super().count(index=index, body=body)
        self._gone = set(self.ids[: self._to_take])
        return answer

    async def search(self, **kw: Any) -> dict[str, Any]:
        resp = await super().search(**kw)
        hits = resp['hits']['hits']
        resp['hits']['hits'] = [h for h in hits if h['_id'] not in self._gone]
        return resp


@pytest.fixture
def ivf_state(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.clustering.methods import ivf_store as ivf_store_mod

    class _Config:
        project_state_dir = tmp_path

    monkeypatch.setattr(ivf_store_mod, 'get_curation_config', _Config)


@pytest.fixture
def pool() -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(5)
    centers = rng.standard_normal((20, 8))
    vectors = centers[rng.integers(0, 20, N_DOCS)] + 0.2 * rng.standard_normal((N_DOCS, 8))
    return [f'd{i:05d}' for i in range(N_DOCS)], vectors.astype(np.float32)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(orch, 'record_last_run', calls.append)
    return calls


async def _train(pool: tuple[list[str], np.ndarray]) -> None:
    await orch.cluster_residuals(FakeItemsClient(*pool), clustering_method='ivf')


async def test_full_pool_is_success_and_reports_coverage(
    pool: tuple[list[str], np.ndarray], ivf_state: None, recorded: list[dict[str, Any]]
) -> None:
    await _train(pool)
    recorded.clear()
    result = await orch.assign_only_residuals(_TakenFromPool(*pool, taken=0), chunk_size=500)
    assert result['status'] == 'success'
    assert result['n_assigned'] == N_DOCS
    assert result['expected_pool'] == N_DOCS
    assert result['coverage'] == 1.0
    assert len(recorded) == 1


async def test_pool_taken_by_another_writer_is_partial(
    pool: tuple[list[str], np.ndarray], ivf_state: None, recorded: list[dict[str, Any]]
) -> None:
    await _train(pool)
    recorded.clear()
    client = _TakenFromPool(*pool, taken=int(N_DOCS * 0.75))
    result = await orch.assign_only_residuals(client, chunk_size=500)

    assert result['status'] == 'partial'
    assert result['reason'] == 'short_read'
    assert result['expected_pool'] == N_DOCS
    assert result['n_assigned'] == N_DOCS - int(N_DOCS * 0.75)
    assert result['coverage'] == pytest.approx(0.25)
    # A partial pass is not recorded as the project's last run.
    assert recorded == []


async def test_coverage_at_the_threshold_passes_and_below_fails(
    pool: tuple[list[str], np.ndarray], ivf_state: None
) -> None:
    await _train(pool)
    allowed = N_DOCS - int(N_DOCS * (1 - POOL_COVERAGE_MIN))
    ok = await orch.assign_only_residuals(
        _TakenFromPool(*pool, taken=N_DOCS - allowed), chunk_size=500
    )
    assert ok['status'] == 'success'
    bad = await orch.assign_only_residuals(
        _TakenFromPool(*pool, taken=N_DOCS - allowed + 1), chunk_size=500
    )
    assert bad['status'] == 'partial'


async def test_failed_writes_make_the_assign_only_pass_partial(
    pool: tuple[list[str], np.ndarray], ivf_state: None, recorded: list[dict[str, Any]]
) -> None:
    await _train(pool)
    recorded.clear()
    client = _TakenFromPool(*pool, taken=0)

    async def refuse(*, body: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
        n = len(body) // 2
        item = {'update': {'status': 500, 'error': {'type': 'boom'}}}
        return {'errors': True, 'items': [item] * n}

    client.bulk = refuse  # type: ignore[method-assign]
    result = await orch.assign_only_residuals(client, chunk_size=500)

    assert result['status'] == 'partial'
    assert result['reason'] == 'write_failures'
    assert result['n_failed_writes'] == N_DOCS
    assert recorded == []
