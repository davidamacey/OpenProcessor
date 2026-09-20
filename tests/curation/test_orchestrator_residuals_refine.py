"""Targeted tests for two of the worst-covered functions in
``src/services/curation/clustering/orchestrator.py`` (plan Wave 5 W5.c
— 31.92% coverage, 465 statements missed): ``refine_region_cluster``
and ``cluster_residuals``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import src.services.curation.clustering.orchestrator as orch


pytestmark = pytest.mark.asyncio


class _FakeScrollBulkOS:
    """Enough of AsyncOpenSearch for _fetch_cluster_members + the bulk
    subid write: one scroll page of cluster members, then bulk()."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs
        self.bulk_calls: list[list[dict[str, Any]]] = []

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        hits = [{'_id': d['_id'], '_source': d['_source']} for d in self._docs]
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': hits}}

    async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def bulk(self, *, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        self.bulk_calls.append(body)
        return {'errors': False}


def _member(doc_id: str, embedding: list[float], class_name: str | None = None) -> dict[str, Any]:
    from src.config import get_region_fields

    F = get_region_fields()
    return {'_id': doc_id, '_source': {F.embedding: embedding, 'class_name': class_name}}


# ---------------------------------------------------------------------------
# refine_region_cluster — real sklearn AHC over two well-separated groups
# ---------------------------------------------------------------------------


async def test_refine_region_cluster_splits_two_separated_groups() -> None:
    from src.config import get_region_fields

    F = get_region_fields()
    group_a = [[1.0, 0.01 * i, 0.0] for i in range(3)]
    group_b = [[0.0, 1.0, 0.01 * i] for i in range(3)]
    members = [_member(f'a{i}', v) for i, v in enumerate(group_a)] + [
        _member(f'b{i}', v) for i, v in enumerate(group_b)
    ]
    client = _FakeScrollBulkOS(members)

    result = await orch.refine_region_cluster(client, region_cluster_id=42)

    assert result['action'] == 'refined'
    assert result['n_members'] == 6
    assert result['n_subclusters'] == 2

    # Wrote through RegionFields.cluster_subid (not the vehicle-class field).
    written_fields = set()
    subids_written = set()
    for chunk in client.bulk_calls:
        for action, doc in zip(chunk[0::2], chunk[1::2], strict=True):
            assert '_index' in action['update']
            written_fields.update(doc['doc'].keys())
            subids_written.add(doc['doc'][F.cluster_subid])
    assert F.cluster_subid in written_fields
    # Two distinct subcluster labels sharing the parent id prefix.
    assert len(subids_written) == 2
    assert all(sid.startswith('42') for sid in subids_written)


async def test_refine_region_cluster_skips_too_few_members() -> None:
    members = [_member('only-one', [1.0, 0.0, 0.0])]
    client = _FakeScrollBulkOS(members)

    result = await orch.refine_region_cluster(client, region_cluster_id=7)

    assert result['action'] == 'skipped_too_small'
    assert result['n_subclusters'] == 0
    assert client.bulk_calls == []


async def test_refine_region_cluster_computes_purity_per_subcluster() -> None:
    group_a = [_member(f'a{i}', [1.0, 0.01 * i, 0.0], class_name='sedan') for i in range(3)]
    group_b = [
        _member('b0', [0.0, 1.0, 0.0], class_name='pickup'),
        _member('b1', [0.0, 1.0, 0.01], class_name='pickup'),
        _member('b2', [0.0, 1.0, 0.02], class_name='van'),
    ]
    client = _FakeScrollBulkOS(group_a + group_b)

    result = await orch.refine_region_cluster(client, region_cluster_id=9)

    assert result['n_subclusters'] == 2
    # group_a is 100% pure sedan, group_b is 2/3 pickup -> weighted mean
    # purity strictly between the two, and overall (mixed) purity lower
    # than either sub-cluster's.
    assert 0.0 < result['subcluster_weighted_purity'] < 1.0
    assert result['purity'] <= result['subcluster_weighted_purity']


# ---------------------------------------------------------------------------
# cluster_residuals — the broadened residual pool (recluster_unvalidated)
# ---------------------------------------------------------------------------


@dataclass
class _FakeBackendInfo:
    name: str = 'cpu'
    detail: str = 'test'
    free_vram_mb: int | None = None
    build_algo: Any = None


@dataclass
class _FakeMethodResult:
    labels: np.ndarray
    method: str = 'fake'
    backend: str = 'cpu'
    params: dict[str, Any] | None = None
    distances: np.ndarray | None = None
    extra: dict[str, Any] | None = None


class _FakeMethod:
    def __init__(self, labels: np.ndarray) -> None:
        self._labels = labels

    async def fit_predict(self, _embeddings: Any, **_kw: Any) -> _FakeMethodResult:
        return _FakeMethodResult(labels=self._labels, params={})


class _FakeBulkOnlyOS:
    def __init__(self) -> None:
        self.bulk_calls: list[list[dict[str, Any]]] = []

        class _Indices:
            async def refresh(self, *, index: str) -> None:
                pass

        self.indices = _Indices()

    async def bulk(self, *, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        self.bulk_calls.append(body)
        return {'errors': False}


async def _run_cluster_residuals(
    monkeypatch: pytest.MonkeyPatch, *, recluster_unvalidated: bool, pool_ids: list[str]
) -> tuple[dict[str, Any], list[list[dict[str, Any]]], dict[str, Any]]:
    captured_fetch_kwargs: dict[str, Any] = {}

    async def _fake_fetch(_client: Any, **kwargs: Any) -> tuple[list[str], np.ndarray]:
        captured_fetch_kwargs.update(kwargs)
        embeddings = np.array(
            [[1.0, float(i), 0.0] for i in range(len(pool_ids))], dtype=np.float32
        ).reshape(len(pool_ids), 3)
        return pool_ids, embeddings

    monkeypatch.setattr(
        'src.services.curation.clustering.backend.detect_cluster_backend',
        lambda: _FakeBackendInfo(),
    )
    monkeypatch.setattr(
        'src.services.curation.clustering.embedding_reduce.fetch_residual_v6_embeddings_parallel',
        _fake_fetch,
    )
    labels = np.array([i % 2 for i in range(len(pool_ids))])
    monkeypatch.setattr(
        'src.services.curation.clustering.methods.get_method',
        lambda _name, **_kw: _FakeMethod(labels),
    )

    client = _FakeBulkOnlyOS()
    result = await orch.cluster_residuals(
        client, recluster_unvalidated=recluster_unvalidated, clustering_method='ivf'
    )
    return result, client.bulk_calls, captured_fetch_kwargs


_ABOVE_MIN_POOL = [f'r{i}' for i in range(orch.MIN_RESIDUALS_FOR_CLUSTERING + 4)]


async def test_cluster_residuals_broadened_pool_flag_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _result, _bulk, fetch_kwargs = await _run_cluster_residuals(
        monkeypatch, recluster_unvalidated=True, pool_ids=_ABOVE_MIN_POOL
    )
    assert fetch_kwargs['include_candidate_clusters'] is True

    _result2, _bulk2, fetch_kwargs2 = await _run_cluster_residuals(
        monkeypatch, recluster_unvalidated=False, pool_ids=_ABOVE_MIN_POOL
    )
    assert fetch_kwargs2['include_candidate_clusters'] is False


async def test_cluster_residuals_writes_land_in_the_residual_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, bulk_calls, _fetch_kwargs = await _run_cluster_residuals(
        monkeypatch, recluster_unvalidated=False, pool_ids=_ABOVE_MIN_POOL
    )
    assert result['status'] == 'success'
    assert result['cluster_id_offset'] == orch.RESIDUAL_CLUSTER_ID_OFFSET

    written_cluster_ids = []
    for chunk in bulk_calls:
        for _action, doc in zip(chunk[0::2], chunk[1::2], strict=True):
            written_cluster_ids.append(doc['doc']['cluster_id'])
    assert written_cluster_ids  # something was written
    for cid in written_cluster_ids:
        assert cid >= orch.RESIDUAL_CLUSTER_ID_OFFSET


async def test_cluster_residuals_too_few_residuals_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, bulk_calls, _fetch_kwargs = await _run_cluster_residuals(
        monkeypatch, recluster_unvalidated=False, pool_ids=[]
    )
    assert result['status'] == 'no_residuals'
    assert bulk_calls == []
