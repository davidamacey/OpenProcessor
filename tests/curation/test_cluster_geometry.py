"""Stale distances are not served, and cluster members can be served core-first.

- The wire serves ``cluster_distance`` / ``cluster_similarity`` /
  ``cluster_is_core`` as null when they were measured against a cluster
  the item has since left.
- ``GET /crops?cluster_id=N&order=core_first`` serves members nearest the
  centroid first, with each item's core flag recomputed against the same
  live centroid, so the first non-core item is the cut line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.services.curation.cluster_ids import CORE_SIMILARITY_MIN
from src.services.curation.wire import serialize_item


if TYPE_CHECKING:
    import pytest


ITEMS = base_curation_config().items_index
DIM = 8
DISTANCE_REF_FIELD = 'cluster_distance_cluster_id'


def _vec(*weights: float) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(weights)] = weights
    return (v / np.linalg.norm(v)).tolist()


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


# ------------------------------------------------------------- core_first


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router
    from src.services.curation.clustering import outliers

    outliers._CACHE.clear()
    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
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
        '/curation/projects/default/crops',
        params={'cluster_id': 5, 'order': 'core_first', 'page_size': 50},
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
        '/curation/projects/default/crops',
        params={'cluster_id': 5, 'order': 'core_first', 'page_size': 4, 'page': 2},
    ).json()
    assert [c['crop_id'] for c in page2['crops']] == [c['crop_id'] for c in crops[4:]]
