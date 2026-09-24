"""DQ-M2: a cluster card's ``purity`` is a real signal, not 1.0 by construction.

Class clusters are filled by labelling (``cluster_id == class_id``), so the
top class's share of their labels is always 1.0 — every class cluster read
"pure", including visibly mixed ones. A card's ``purity`` is now the share
of the members the cluster-geometry pass measured for that cluster whose
nearest cluster centroid is their own (``purity_basis:
'nearest_centroid'``), over ``purity_n`` members. The label purity (the
auto-promote gate's input) is served separately as ``label_purity`` with
``labelled_share``.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from curation.query_fakes import matches
from src.routers.curation.clusters import PURITY_BASIS, list_clusters
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.clustering.cluster_geometry import (
    DISTANCE_REF_FIELD,
    NEAREST_FIELD,
    write_cluster_geometry,
)
from src.services.curation.wire import serialize_item


DIM = 8


def _vec(*weights: float) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(weights)] = weights
    return (v / np.linalg.norm(v)).tolist()


class _Fake:
    def __init__(self, docs: dict[str, dict[str, Any]]) -> None:
        self.docs = copy.deepcopy(docs)
        self.indices = AsyncMock()

    async def search(self, *, index: str, body: dict[str, Any], scroll: str | None = None) -> Any:  # noqa: ARG002
        pool = [(i, d) for i, d in self.docs.items() if matches(d, body.get('query'))]
        if body.get('aggs'):
            keys = sorted({d['cluster_id'] for _i, d in pool})
            return {'aggregations': {'ids': {'buckets': [{'key': k} for k in keys]}}}
        return {'_scroll_id': 's', 'hits': {'hits': [{'_id': i, '_source': d} for i, d in pool]}}

    async def scroll(self, **_kw: Any) -> Any:
        return {'_scroll_id': 's', 'hits': {'hits': []}}

    async def clear_scroll(self, **_kw: Any) -> None:
        return None

    async def bulk(self, *, body: list[dict[str, Any]], **_kw: Any) -> Any:
        for action, payload in zip(body[::2], body[1::2], strict=True):
            doc = self.docs[action['update']['_id']]
            params = payload['script']['params']
            if doc.get('cluster_id') == params['cid']:
                doc.update(params['fields'])
        return {'errors': False, 'items': []}


@pytest.mark.asyncio
async def test_geometry_pass_records_each_members_nearest_centroid() -> None:
    docs = {
        # Class 5 is mostly axis-0 items, plus two that look like class 6.
        **{f'a{i}': {'cluster_id': 5, 'pe_embedding': _vec(1, 0.05 * i)} for i in range(6)},
        'intruder1': {'cluster_id': 5, 'pe_embedding': _vec(0, 1)},
        'intruder2': {'cluster_id': 5, 'pe_embedding': _vec(0.05, 1)},
        **{f'b{i}': {'cluster_id': 6, 'pe_embedding': _vec(0.05 * i, 1)} for i in range(6)},
        'cand': {'cluster_id': RESIDUAL_CLUSTER_ID_OFFSET, 'pe_embedding': _vec(0, 0, 1)},
    }
    fake = _Fake(docs)
    out = await write_cluster_geometry(fake, index='op_items')
    assert out['n_nearest_elsewhere'] == 2
    for i in range(6):
        assert fake.docs[f'a{i}'][NEAREST_FIELD] == 5
        assert fake.docs[f'b{i}'][NEAREST_FIELD] == 6
    assert fake.docs['intruder1'][NEAREST_FIELD] == 6
    assert fake.docs['intruder2'][NEAREST_FIELD] == 6
    assert fake.docs['cand'][NEAREST_FIELD] == RESIDUAL_CLUSTER_ID_OFFSET


def _bucket(
    cid: int, size: int, classes: list[tuple[str, int]], measured: int, fits: int
) -> dict[str, Any]:
    return {
        'key': cid,
        'doc_count': size,
        'top_class': {'buckets': [{'key': k, 'doc_count': n} for k, n in classes]},
        'labelled': {'doc_count': sum(n for _k, n in classes)},
        'validated': {'doc_count': 0},
        'subclusters': {'value': 0},
        'latest_update': {},
        'geometry_measured': {'doc_count': measured},
        'geometry_fits': {'doc_count': fits},
    }


async def _cards(*buckets: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    os_client = AsyncMock()
    os_client.search = AsyncMock(
        return_value={'aggregations': {'clusters': {'buckets': list(buckets)}}}
    )
    resp = await list_clusters(
        os_client,
        per_cluster=0,
        max_clusters=100,
        kind='all',
        class_id=None,
        cluster_id=None,
        max_rank=None,
        min_blur_ratio=None,
        class_source=None,
    )
    assert os_client.search.await_args is not None
    body = os_client.search.await_args.kwargs['body']
    return {c['cluster_id']: c for c in resp['items']}, body


@pytest.mark.asyncio
async def test_mixed_class_cluster_no_longer_reads_pure() -> None:
    # Every member carries the cluster's own class (label purity 1.0), but
    # only 5 of the 87 measured members sit nearest their own centroid.
    cards, _body = await _cards(_bucket(43, 87, [('motard', 87)], measured=87, fits=5))
    card = cards[43]
    assert card['label_purity'] == 1.0
    assert card['purity'] == pytest.approx(5 / 87)
    assert card['purity_n'] == 87
    assert card['purity_basis'] == PURITY_BASIS == 'nearest_centroid'
    assert card['purity_tier'] == 'noisy'


@pytest.mark.asyncio
async def test_candidate_serves_label_purity_and_share_next_to_purity() -> None:
    cid = RESIDUAL_CLUSTER_ID_OFFSET + 13
    # 8 of 79 labelled, all one class: label purity 1.0 over a 10% minority.
    cards, _body = await _cards(_bucket(cid, 79, [('motard', 8)], measured=79, fits=70))
    card = cards[cid]
    assert card['label_purity'] == 1.0
    assert card['labelled_share'] == pytest.approx(8 / 79)
    assert card['purity'] == pytest.approx(70 / 79)
    # The gate still needs half the members labelled.
    assert card['promotable'] is False


@pytest.mark.asyncio
async def test_unmeasured_cluster_has_no_purity() -> None:
    cards, _body = await _cards(_bucket(7, 12, [('van', 12)], measured=0, fits=0))
    assert cards[7]['purity'] is None
    assert cards[7]['purity_n'] == 0
    assert cards[7]['purity_tier'] is None


@pytest.mark.asyncio
async def test_purity_counts_only_members_measured_for_their_current_cluster() -> None:
    _unused, body = await _cards()
    aggs = body['aggs']['clusters']['aggs']
    measured = str(aggs['geometry_measured'])
    fits = str(aggs['geometry_fits'])
    assert "doc['cluster_distance_cluster_id'].value == doc['cluster_id'].value" in measured
    assert "doc['cluster_nearest_id'].value == doc['cluster_id'].value" in fits
    assert "doc['cluster_distance_cluster_id'].value == doc['cluster_id'].value" in fits


def test_wire_serves_nearest_cluster_only_when_measured_for_the_current_one() -> None:
    measured = {'cluster_id': 5, DISTANCE_REF_FIELD: 5, NEAREST_FIELD: 6}
    assert serialize_item(measured, 'x', api_prefix='')['cluster_nearest_id'] == 6
    moved = {'cluster_id': 9, DISTANCE_REF_FIELD: 5, NEAREST_FIELD: 6}
    assert serialize_item(moved, 'x', api_prefix='')['cluster_nearest_id'] is None
    assert serialize_item({'cluster_id': 5}, 'x', api_prefix='')['cluster_nearest_id'] is None
