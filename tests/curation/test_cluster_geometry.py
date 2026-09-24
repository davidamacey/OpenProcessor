"""DQ-M3: every clustered item gets geometry, stale geometry is not served,
and cluster members can be served core-first.

- :func:`write_cluster_geometry` writes ``cluster_distance`` (member-mean
  centroid) for class-cluster members, which labelling never gave one, and
  ``cluster_distance_cluster_id`` for every clustered member; candidate
  members keep their clustering method's distance.
- Writes are guarded on the item still being in the measured cluster.
- The wire serves ``cluster_distance`` / ``cluster_similarity`` /
  ``cluster_is_core`` as null when they were measured against a cluster
  the item has since left.
- ``GET /crops?cluster_id=N&order=core_first`` serves members nearest the
  centroid first, with each item's core flag recomputed against the same
  live centroid, so the first non-core item is the cut line.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch, matches
from src.config import get_curation_config
from src.services.curation.cluster_ids import CORE_SIMILARITY_MIN
from src.services.curation.clustering.cluster_geometry import (
    DISTANCE_REF_FIELD,
    cluster_geometry_stage,
    write_cluster_geometry,
)
from src.services.curation.wire import serialize_item


ITEMS = get_curation_config().items_index
DIM = 8


def _vec(*weights: float) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(weights)] = weights
    return (v / np.linalg.norm(v)).tolist()


class _GeometryFake:
    """Evaluates queries with the shared matcher and applies the pass's
    guarded script updates (``fields`` set only if ``cluster_id == cid``)."""

    def __init__(self, docs: dict[str, dict[str, Any]]) -> None:
        self.docs = copy.deepcopy(docs)
        self.indices = AsyncMock()
        self.move_during_write: dict[str, int] = {}

    async def search(self, *, index: str, body: dict[str, Any], scroll: str | None = None) -> Any:  # noqa: ARG002
        pool = [(i, d) for i, d in self.docs.items() if matches(d, body.get('query'))]
        if body.get('aggs'):
            keys = sorted({d['cluster_id'] for _i, d in pool})
            return {'aggregations': {'ids': {'buckets': [{'key': k} for k in keys]}}}
        hits = [{'_id': i, '_source': copy.deepcopy(d)} for i, d in pool]
        return {'_scroll_id': 's', 'hits': {'hits': hits}}

    async def scroll(self, **_kw: Any) -> Any:
        return {'_scroll_id': 's', 'hits': {'hits': []}}

    async def clear_scroll(self, **_kw: Any) -> None:
        return None

    async def bulk(self, *, body: list[dict[str, Any]], **_kw: Any) -> Any:
        for doc_id, cid in self.move_during_write.items():
            self.docs[doc_id]['cluster_id'] = cid
        for action, payload in zip(body[::2], body[1::2], strict=True):
            doc = self.docs[action['update']['_id']]
            params = payload['script']['params']
            if doc.get('cluster_id') == params['cid']:
                doc.update(params['fields'])
        return {'errors': False, 'items': []}


def _docs() -> dict[str, dict[str, Any]]:
    return {
        # Class cluster 5: no distances yet; c3 carries a stale one from
        # the candidate cluster it came from.
        'c1': {'crop_id': 'c1', 'cluster_id': 5, 'pe_embedding': _vec(1, 0)},
        'c2': {'crop_id': 'c2', 'cluster_id': 5, 'pe_embedding': _vec(1, 0.2)},
        'c3': {
            'crop_id': 'c3',
            'cluster_id': 5,
            'pe_embedding': _vec(0.2, 1),
            'cluster_distance': 0.01,
            DISTANCE_REF_FIELD: 10001,
        },
        # Candidate cluster: keeps its method's distance.
        'k1': {
            'crop_id': 'k1',
            'cluster_id': 10001,
            'pe_embedding': _vec(0, 0, 1),
            'cluster_distance': 0.3,
        },
        'k2': {
            'crop_id': 'k2',
            'cluster_id': 10001,
            'pe_embedding': _vec(0, 0, 1, 0.1),
            'cluster_distance': 0.2,
        },
        'excluded': {
            'crop_id': 'excluded',
            'cluster_id': 5,
            'pe_embedding': _vec(0, 1),
            'class_excluded': True,
        },
    }


def _expected(members: list[list[float]]) -> np.ndarray:
    x = np.asarray(members, dtype=np.float32)
    c = x.mean(axis=0)
    c /= np.linalg.norm(c)
    return np.clip(1.0 - x @ c, 0.0, 2.0)


@pytest.mark.asyncio
async def test_class_cluster_members_get_distances() -> None:
    fake = _GeometryFake(_docs())
    result = await write_cluster_geometry(fake, index=ITEMS)
    assert result['status'] == 'success'
    assert result['n_class_distances'] == 3

    docs = fake.docs
    want = _expected([docs[i]['pe_embedding'] for i in ('c1', 'c2', 'c3')])
    got = [docs[i]['cluster_distance'] for i in ('c1', 'c2', 'c3')]
    np.testing.assert_allclose(got, want, atol=1e-5)
    assert all(docs[i][DISTANCE_REF_FIELD] == 5 for i in ('c1', 'c2', 'c3'))
    # Candidate members keep the method's distance but record the reference.
    assert docs['k1']['cluster_distance'] == 0.3
    assert docs['k1'][DISTANCE_REF_FIELD] == 10001
    # Excluded items are neither measured nor written.
    assert 'cluster_distance' not in docs['excluded']


@pytest.mark.asyncio
async def test_item_moved_during_the_pass_is_left_alone() -> None:
    fake = _GeometryFake(_docs())
    fake.move_during_write = {'c2': 7}
    await write_cluster_geometry(fake, index=ITEMS)
    assert 'cluster_distance' not in fake.docs['c2']
    assert DISTANCE_REF_FIELD not in fake.docs['c2']


@pytest.mark.asyncio
async def test_stage_reports_failure_instead_of_raising() -> None:
    boom = AsyncMock()
    boom.search = AsyncMock(side_effect=RuntimeError('down'))
    out = await cluster_geometry_stage(boom)
    assert out == {'status': 'error', 'error': 'down'}


def test_wire_hides_geometry_measured_against_another_cluster() -> None:
    stale = serialize_item(
        {'cluster_id': 5, 'cluster_distance': 0.01, DISTANCE_REF_FIELD: 10001}, 'x', api_prefix=''
    )
    assert stale['cluster_distance'] is None
    assert stale['cluster_similarity'] is None
    assert stale['cluster_is_core'] is None

    current = serialize_item(
        {'cluster_id': 5, 'cluster_distance': 0.1, DISTANCE_REF_FIELD: 5}, 'x', api_prefix=''
    )
    assert current['cluster_distance'] == 0.1
    assert current['cluster_is_core'] is True
    # Written before the reference existed: served as stored.
    legacy = serialize_item({'cluster_id': 5, 'cluster_distance': 0.1}, 'x', api_prefix='')
    assert legacy['cluster_distance'] == 0.1


def test_residual_and_ingest_writers_record_the_reference() -> None:
    from src.services.curation.clustering.orchestrator import _guarded_class_cluster_write
    from src.services.curation.item_doc import DetectedItem, build_item_doc

    script = _guarded_class_cluster_write(10003, 0.2)['script']['source']
    assert "ctx._source['cluster_distance_cluster_id'] = params.cid" in script
    doc = build_item_doc(
        crop_id='x',
        image_id='i',
        image_path='p.jpg',
        source='s',
        request_id='r',
        bbox_norm=[0.1, 0.1, 0.2, 0.2],
        item=DetectedItem(
            bbox_pixel=(0, 0, 1, 1), score=0.9, cluster_id=10003, cluster_distance=0.2
        ),
        now='2026-09-24T00:00:00Z',
        crop_area_norm=0.01,
        crop_rank_in_image=1,
        blur_full_var=None,
        blur_lap_var=None,
        blur_lap_ratio=None,
    )
    assert doc[DISTANCE_REF_FIELD] == 10003


# ------------------------------------------------------------- core_first


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router
    from src.services.curation.clustering import outliers

    outliers._CACHE.clear()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_core_first_orders_nearest_first_with_a_consistent_cut_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs: dict[str, dict[str, Any]] = {}
    # A tight core around axis 0 plus two far members.
    for i, w in enumerate([0.0, 0.05, 0.1, 0.15]):
        docs[f'core{i}'] = {
            'crop_id': f'core{i}',
            'cluster_id': 5,
            'pe_embedding': _vec(1, w),
            'updated_at': f'2026-09-2{i}',
        }
    docs['far0'] = {'crop_id': 'far0', 'cluster_id': 5, 'pe_embedding': _vec(0.1, 1)}
    docs['far1'] = {'crop_id': 'far1', 'cluster_id': 5, 'pe_embedding': _vec(0, 0.2, 1)}
    docs['other'] = {'crop_id': 'other', 'cluster_id': 6, 'pe_embedding': _vec(1, 0)}
    client = _client(QueryFakeOpenSearch({ITEMS: docs}), monkeypatch)

    r = client.get(
        '/curation/crops', params={'cluster_id': 5, 'order': 'core_first', 'page_size': 50}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['method'] == 'core_first'
    assert body['total'] == 6
    crops = body['crops']
    dists = [c['cluster_distance'] for c in crops]
    assert dists == sorted(dists)
    assert {c['crop_id'] for c in crops[-2:]} == {'far0', 'far1'}
    flags = [c['cluster_is_core'] for c in crops]
    assert all(isinstance(f, bool) for f in flags)
    # Core first, then non-core: exactly one cut.
    cut = flags.index(False)
    assert all(flags[:cut])
    assert not any(flags[cut:])
    for c in crops:
        assert c['cluster_is_core'] == (c['cluster_similarity'] >= CORE_SIMILARITY_MIN)

    page2 = client.get(
        '/curation/crops',
        params={'cluster_id': 5, 'order': 'core_first', 'page_size': 4, 'page': 2},
    ).json()
    assert [c['crop_id'] for c in page2['crops']] == [c['crop_id'] for c in crops[4:]]
