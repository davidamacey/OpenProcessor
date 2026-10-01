"""Tests for ``src/routers/curation/regions_fp.py`` over production-shaped
items (box lists + per-box vectors) in the query-evaluating fake.

Covers the region-cluster cards (``GET /regions/clusters``: the permanent
false-positive bucket sorts first, ``size`` counts items and ``box_count``
boxes), the "no centroids built yet" short-circuit and the per-box rows of
``GET /regions/suspected_false_positives``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, get_region_fields, index_name
from src.config.curation import base_curation_config
from src.services.curation.cluster_ids import FALSE_POSITIVE_REGION_CLUSTER_ID
from src.services.curation.region_box_embeddings import entry_for
from src.services.curation.region_boxes import RegionBox, boxes_write_fields


# No-profile gating contract: this file exercises region routes, which
# require an active region profile (409 otherwise).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


F = get_region_fields()
ITEMS = index_name(base_curation_config(), IndexRole.ITEMS)
PREFIX = '/curation/projects/default'
FP = FALSE_POSITIVE_REGION_CLUSTER_ID


def _vec(*xyz: float) -> list[float]:
    v = np.asarray(xyz, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


A, B = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)


def _box(box_id: str, state: str = 'accepted', **over: Any) -> RegionBox:
    x = 0.1 * int(box_id[1:])
    return RegionBox(box_id=box_id, bbox_norm=(x, 0.1, x + 0.05, 0.3), state=state, **over)


def _item(
    crop_id: str,
    boxes: list[tuple[RegionBox, tuple[float, float, float] | None]],
    **extra: Any,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'image_path': f'/data/{crop_id}.jpg',
        **boxes_write_fields([b for b, _v in boxes], current_src={}),
        F.status: 'detected',
        **extra,
    }
    doc[F.box_embeddings] = [entry_for(b, _vec(*v)) for b, v in boxes if v is not None]
    return doc


class _CountingOS(QueryFakeOpenSearch):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.search_calls = 0
        self.mget_kwargs: list[dict[str, Any]] = []
        self.raise_on_scroll = False
        self.clear_scroll_calls = 0

    async def search(self, **kw: Any) -> dict[str, Any]:
        self.search_calls += 1
        return await super().search(**kw)

    async def scroll(self, **kw: Any) -> dict[str, Any]:
        if self.raise_on_scroll:
            msg = 'transport boom mid-scroll'
            raise RuntimeError(msg)
        return await super().scroll(**kw)

    async def clear_scroll(self, **kw: Any) -> dict[str, Any]:
        self.clear_scroll_calls += 1
        return await super().clear_scroll(**kw)

    async def mget(self, **kw: Any) -> dict[str, Any]:
        self.mget_kwargs.append(kw)
        return await super().mget(**kw)


def _client(fake: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


@pytest.fixture
def fp_store_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr('src.services.detection.fp_store.fp_store_dir', lambda *_a, **_k: tmp_path)
    return tmp_path


def _save_centroid(vector: tuple[float, float, float], trained_at: str = 'T1') -> None:
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    FalsePositiveCentroidStore().save(
        np.asarray([_vec(*vector)], dtype=np.float32),
        {'trained_at': trained_at, 'k': 1, 'n_boxes': 1, 'subids': [f'{FP}a'], 'dim': 3},
    )


# ---------------------------------------------------------------------------
# cluster cards
# ---------------------------------------------------------------------------


def test_cluster_cards_pin_the_fp_bucket_first() -> None:
    # A larger "good" bucket plus the (smaller) permanent FP bucket: size
    # ordering alone would put the good bucket first; the FP pin overrides it.
    docs = {
        **{f'g{i}': _item(f'g{i}', [(_box('b1', cluster_id=7), A)]) for i in range(5)},
        'fp': _item('fp', [(_box('b1', 'false_positive', cluster_id=FP), B)]),
    }
    client = _client(QueryFakeOpenSearch({ITEMS: docs}))

    resp = client.get(f'{PREFIX}/regions/clusters')

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['count'] == 2
    assert body['clusters'][0]['id'] == FP
    assert body['clusters'][0]['cluster_kind'] == 'false_positive'
    assert body['clusters'][1]['id'] == 7
    assert body['clusters'][1]['cluster_kind'] == 'candidate'


def test_cluster_cards_count_items_and_boxes_separately() -> None:
    docs = {
        # two boxes of ONE item in cluster 9: size 1, box_count 2
        'two': _item('two', [(_box('b1', cluster_id=9), A), (_box('b2', cluster_id=9), A)]),
        'one': _item('one', [(_box('b1', cluster_id=9), A)], **{F.validated: True}),
        'elsewhere': _item('elsewhere', [(_box('b1', cluster_id=3), A)]),
    }
    client = _client(QueryFakeOpenSearch({ITEMS: docs}))

    cards = {c['id']: c for c in client.get(f'{PREFIX}/regions/clusters').json()['clusters']}

    assert cards[9]['size'] == 2
    assert cards[9]['box_count'] == 3
    assert cards[9]['validated_count'] == 1
    assert cards[3]['size'] == cards[3]['box_count'] == 1


def test_cluster_cards_serve_representative_rows_with_their_box_ids() -> None:
    docs = {
        'x': _item(
            'x',
            [
                (_box('b1', cluster_id=9, cluster_subid='9a'), A),
                (_box('b2', cluster_id=4), A),
            ],
        )
    }
    client = _client(QueryFakeOpenSearch({ITEMS: docs}))

    cards = {c['id']: c for c in client.get(f'{PREFIX}/regions/clusters').json()['clusters']}

    nine = cards[9]
    assert nine['representative_crop_ids'] == ['x']
    assert nine['representative_box_ids'] == ['b1']
    assert nine['representative_thumb_urls'][0].endswith('/crops/x/region_thumbnail?box_id=b1')
    (row,) = nine['representatives']
    assert (row['crop_id'], row['region_box_id']) == ('x', 'b1')
    assert [b['box_id'] for b in row['region_boxes']] == ['b1', 'b2']
    assert nine['has_subclusters'] is True
    assert nine['n_subclusters'] == 1
    assert cards[4]['representative_box_ids'] == ['b2']


def test_cluster_representatives_are_the_boxes_nearest_the_centroid() -> None:
    docs = {
        f'c{i}': _item(f'c{i}', [(_box('b1', cluster_id=9, cluster_distance=d), A)])
        for i, d in enumerate([0.9, 0.1, 0.5, 0.3])
    }
    # An unmeasured box (no distance) must sort after every measured one.
    docs['unmeasured'] = _item('unmeasured', [(_box('b1', cluster_id=9), A)])
    client = _client(QueryFakeOpenSearch({ITEMS: docs}))

    card = client.get(f'{PREFIX}/regions/clusters', params={'per_cluster': 3}).json()['clusters'][0]

    assert card['representative_crop_ids'] == ['c1', 'c3', 'c2']


def test_cluster_cards_ignore_rejected_and_unclustered_boxes() -> None:
    docs = {
        'rej': _item('rej', [(_box('b1', 'rejected', cluster_id=5), A)]),
        'plain': _item('plain', [(_box('b1'), A)]),
    }
    client = _client(QueryFakeOpenSearch({ITEMS: docs}))

    assert client.get(f'{PREFIX}/regions/clusters').json() == {'clusters': [], 'count': 0}


def test_cluster_cards_surface_an_opensearch_error_as_503() -> None:
    class _BoomOS:
        async def search(self, **_kw: Any) -> dict[str, Any]:
            raise RuntimeError('cluster down')

    resp = _client(_BoomOS()).get(f'{PREFIX}/regions/clusters')

    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# suspected false positives (one row per box)
# ---------------------------------------------------------------------------


def test_suspected_false_positives_short_circuits_when_no_centroids_built(
    fp_store_dir: Any,
) -> None:
    body = (
        _client(QueryFakeOpenSearch({ITEMS: {}}))
        .get(f'{PREFIX}/regions/suspected_false_positives')
        .json()
    )

    assert body['items'] == []
    assert body['total'] == body['total_rows'] == 0
    assert body['centroids_built'] is False


def test_suspected_false_positives_are_boxes_not_items(fp_store_dir: Any) -> None:
    _save_centroid(B)
    docs = {
        # both boxes resemble the FP centroid: two rows, nearest first
        'both': _item('both', [(_box('b1'), (0.0, 1.0, 0.2)), (_box('b2'), B)]),
        # only the second box does: one row, for that box
        'one': _item('one', [(_box('b1'), A), (_box('b2'), (0.0, 1.0, 0.1))]),
        'far': _item('far', [(_box('b1'), A)]),
    }
    fake = _CountingOS({ITEMS: docs})

    resp = _client(fake).get(f'{PREFIX}/regions/suspected_false_positives')

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [(r['crop_id'], r['region_box_id']) for r in body['items']] == [
        ('both', 'b2'),
        ('one', 'b2'),
        ('both', 'b1'),
    ]
    assert body['total'] == body['total_rows'] == 3
    assert all(r['nearest_fp_subid'] == f'{FP}a' for r in body['items'])
    assert body['items'][0]['suspected_fp_distance'] == pytest.approx(0.0, abs=1e-5)
    assert body['items'][0]['suspected_fp_distance'] < body['items'][2]['suspected_fp_distance']
    assert body['centroids_built'] is True


def test_suspected_fp_hydrates_rows_with_source_excludes_not_a_source_dict(
    fp_store_dir: Any,
) -> None:
    """Regression: the mget must pass ``_source_excludes=`` (a real
    opensearch-py kwarg), not ``_source={'excludes': [...]}`` (silently
    stringified into a useless include pattern -> an empty ``_source``)."""
    from src.routers.curation.regions import _REGION_SOURCE_EXCLUDES

    _save_centroid(B)
    fake = _CountingOS({ITEMS: {'x': _item('x', [(_box('b1'), B)])}})

    body = _client(fake).get(f'{PREFIX}/regions/suspected_false_positives').json()

    (call,) = fake.mget_kwargs
    assert call['_source_excludes'] == _REGION_SOURCE_EXCLUDES
    assert '_source' not in call
    assert body['items'][0]['image_path'] == '/data/x.jpg'


def test_suspected_fp_excludes_holdout_human_and_locked_candidates(fp_store_dir: Any) -> None:
    _save_centroid(B)
    docs = {
        'ok': _item('ok', [(_box('b1'), B)]),
        'holdout': _item('holdout', [(_box('b1'), B)], test_holdout=True),
        'human': _item('human', [(_box('b1'), B)], **{F.verifier: 'human'}),
        'own': _item('own', [(_box('b1', source='human'), B)]),
        'fp': _item('fp', [(_box('b1', 'false_positive'), B)]),
    }

    body = (
        _client(QueryFakeOpenSearch({ITEMS: docs}))
        .get(f'{PREFIX}/regions/suspected_false_positives')
        .json()
    )

    assert [r['crop_id'] for r in body['items']] == ['ok']


def test_suspected_fp_second_page_within_ttl_does_not_rescroll(fp_store_dir: Any) -> None:
    """The scored-list cache means a page-2 request shortly after page-1
    doesn't re-scroll the whole vector pool."""
    _save_centroid(B)
    docs = {f'r{i}': _item(f'r{i}', [(_box('b1'), B)]) for i in range(3)}
    fake = _CountingOS({ITEMS: docs})
    client = _client(fake)

    r1 = client.get(f'{PREFIX}/regions/suspected_false_positives?page=1&page_size=2')
    scans_after_first = fake.search_calls
    r2 = client.get(f'{PREFIX}/regions/suspected_false_positives?page=2&page_size=2')

    assert r1.status_code == r2.status_code == 200
    assert len(r1.json()['items']) == 2
    assert len(r2.json()['items']) == 1
    assert fake.search_calls == scans_after_first


def test_suspected_fp_scan_error_still_clears_the_scroll(fp_store_dir: Any) -> None:
    """An exception mid-scroll must still hit clear_scroll (finally), not
    leak an open scroll context."""
    _save_centroid(B)
    fake = _CountingOS({ITEMS: {'r0': _item('r0', [(_box('b1'), B)])}})
    fake.raise_on_scroll = True

    resp = _client(fake).get(f'{PREFIX}/regions/suspected_false_positives')

    assert resp.status_code == 503
    assert fake.clear_scroll_calls == 1


def test_cluster_status_and_fp_centroid_status_are_reachable(fp_store_dir: Any) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: {}}))

    assert client.get(f'{PREFIX}/regions/cluster/status').status_code == 200
    assert client.get(f'{PREFIX}/regions/fp_centroids/status').status_code == 200
