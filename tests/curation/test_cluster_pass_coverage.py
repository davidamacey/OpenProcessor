"""A cluster pass that did not cover its pool is ``partial``, never ``success`` (#357).

The fake index below is a real pool of documents read through the production
PIT + slice fetch, so a second writer that assigns items first (the cluster-refresh
daemon assigning a new project's items) changes what the pass sees exactly as it did on
the live stack (5,117 of 25,000 and 62,760 of 91,184 clustered, status ``success``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import src.services.curation.clustering.orchestrator as orch
from src.services.curation.autolabel.job import failed_stages
from src.services.curation.clustering.pool_coverage import POOL_COVERAGE_MIN


pytestmark = pytest.mark.asyncio

OFFSET_ASSIGNED = 10_000_000


class _PoolIndex:
    """Just enough of AsyncOpenSearch: count, PIT, sliced search_after, bulk, refresh.

    ``docs`` maps id to whether another writer already assigned it (assigned items are
    outside the strict residual pool). ``before_pit`` runs once between the pass's count
    and its point-in-time, the window the daemon writes into.
    """

    def __init__(self, n: int, *, assigned: int = 0, bulk_fail: bool = False) -> None:
        self.ids = [f'd{i:06d}' for i in range(n)]
        self.assigned = {i: pos < assigned for pos, i in enumerate(self.ids)}
        self.bulk_fail = bulk_fail
        self.bulk_docs = 0
        self.before_pit: Any = None
        self._pit: list[str] = []
        self.indices = self

    def _pool(self) -> list[str]:
        return [i for i in self.ids if not self.assigned[i]]

    async def refresh(self, *, index: str) -> None:  # noqa: ARG002
        return None

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return {'count': len(self._pool())}

    async def create_pit(self, *, index: str, keep_alive: str) -> dict[str, Any]:  # noqa: ARG002
        if self.before_pit is not None:
            self.before_pit(self)
            self.before_pit = None
        self._pit = self._pool()
        return {'pit_id': 'pit'}

    async def delete_pit(self, *, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def search(self, *, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        sl = body.get('slice') or {'id': 0, 'max': 1}
        after = (body.get('search_after') or [-1])[0]
        mine = [
            (pos, i)
            for pos, i in enumerate(self._pit)
            if pos % sl['max'] == sl['id'] and pos > after
        ][: body['size']]
        hits = [
            {'_id': i, '_source': {'pe_embedding': [1.0, float(pos % 7), 1.0]}, 'sort': [pos]}
            for pos, i in mine
        ]
        return {'hits': {'hits': hits}}

    async def bulk(self, *, body: list[dict[str, Any]], **_kw: Any) -> dict[str, Any]:
        n = len(body) // 2
        self.bulk_docs += n
        if self.bulk_fail:
            item = {'update': {'status': 500, 'error': {'type': 'boom'}}}
            return {'errors': True, 'items': [item] * n}
        return {'errors': False, 'items': []}


@dataclass
class _Backend:
    name: str = 'cpu'
    detail: str = 'test'
    free_vram_mb: int | None = None


@dataclass
class _FitResult:
    labels: np.ndarray
    method: str = 'fake'
    backend: str = 'cpu'
    params: dict[str, Any] | None = None
    distances: np.ndarray | None = None
    extra: dict[str, Any] | None = None


class _Method:
    def __init__(self) -> None:
        self.fitted = 0

    async def fit_predict(self, embeddings: np.ndarray, **_kw: Any) -> _FitResult:
        self.fitted += 1
        return _FitResult(labels=np.arange(len(embeddings)) % 3)


@pytest.fixture
def method(monkeypatch: pytest.MonkeyPatch) -> _Method:
    m = _Method()
    monkeypatch.setattr('src.services.curation.clustering.backend.detect_cluster_backend', _Backend)
    monkeypatch.setattr('src.services.curation.clustering.methods.get_method', lambda *_a, **_k: m)
    return m


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(orch, 'record_last_run', calls.append)
    return calls


async def test_full_pool_is_success(method: _Method, recorded: list[dict[str, Any]]) -> None:
    index = _PoolIndex(2000)
    result = await orch.cluster_residuals(index, clustering_method='ivf')
    assert result['status'] == 'success'
    assert result['n_residuals'] == 2000
    assert index.bulk_docs == 2000
    assert len(recorded) == 1


async def test_writer_assigns_between_count_and_fetch_is_partial(
    method: _Method, recorded: list[dict[str, Any]]
) -> None:
    index = _PoolIndex(2000)

    def daemon(idx: _PoolIndex) -> None:
        for i in idx.ids[:1500]:
            idx.assigned[i] = True

    index.before_pit = daemon
    result = await orch.cluster_residuals(index, clustering_method='ivf')

    assert result['status'] == 'partial'
    assert result['n_residuals'] == 500
    assert result['expected_pool'] == 2000
    assert result['coverage'] == pytest.approx(0.25)
    # Fail closed: nothing was trained, written or recorded for a pool it did not cover.
    assert method.fitted == 0
    assert index.bulk_docs == 0
    assert recorded == []


async def test_pool_the_caller_expects_but_another_writer_took_is_partial(
    method: _Method, recorded: list[dict[str, Any]]
) -> None:
    index = _PoolIndex(5000, assigned=4000)
    result = await orch.cluster_residuals(index, clustering_method='ivf', expected_pool=5000)

    assert result['status'] == 'partial'
    assert result['n_residuals'] == 1000
    assert result['expected_pool'] == 5000
    assert index.bulk_docs == 0
    assert recorded == []


async def test_a_pool_that_shrank_before_the_pass_is_not_partial_without_an_expectation(
    method: _Method,
) -> None:
    index = _PoolIndex(5000, assigned=4000)
    result = await orch.cluster_residuals(index, clustering_method='ivf')
    assert result['status'] == 'success'
    assert result['n_residuals'] == 1000


async def test_coverage_at_the_threshold_passes_and_below_fails(method: _Method) -> None:
    n = 1000
    short = n - int(n * (1 - POOL_COVERAGE_MIN))
    ok = await orch.cluster_residuals(_PoolIndex(short), clustering_method='ivf', expected_pool=n)
    assert ok['status'] == 'success'
    bad = await orch.cluster_residuals(
        _PoolIndex(short - 1), clustering_method='ivf', expected_pool=n
    )
    assert bad['status'] == 'partial'


async def test_failed_writes_make_the_pass_partial(
    method: _Method, recorded: list[dict[str, Any]]
) -> None:
    index = _PoolIndex(2000, bulk_fail=True)
    result = await orch.cluster_residuals(index, clustering_method='ivf')

    assert result['status'] == 'partial'
    assert result['reason'] == 'write_failures'
    assert result['n_failed_writes'] == 2000
    assert recorded == []


async def test_a_partial_stage_fails_the_job() -> None:
    assert failed_stages({'stages': {'cluster_residuals': {'status': 'partial'}}}) == [
        'cluster_residuals'
    ]
    assert failed_stages({'stages': {'cluster_residuals': {'status': 'success'}}}) == []
