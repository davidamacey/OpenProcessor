"""Row shapes of the region list routes (W8.10).

A row is the full wire item plus ``region_box_id``. A request that selects
boxes returns one row per matching box (all box filters matching the SAME
box); one that selects only items returns item rows with a null box id.
``total`` counts items and ``total_rows`` rows.
"""

from __future__ import annotations

from typing import Any

import pytest
from _region_profile_fixture import REFERENCE_REGION_DETECTOR_MODEL
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch.ensure_overlay_fields import (
    ensure_items_inner_result_window,
    inner_result_window,
)
from src.config import IndexRole, get_region_fields, index_name
from src.config.curation import base_curation_config
from src.services.curation.cluster_ids import FALSE_POSITIVE_REGION_CLUSTER_ID
from src.services.curation.region_boxes import RegionBox, boxes_write_fields


F = get_region_fields()
CFG = base_curation_config()
ITEMS = index_name(CFG, IndexRole.ITEMS)
PREFIX = '/curation/projects/default'

pytestmark = pytest.mark.usefixtures('reference_region_profile')


def _client(fake: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _box(box_id: str, state: str = 'accepted', **over: Any) -> RegionBox:
    n = int(box_id[1:])
    x = 0.01 * (n % 90)
    kwargs: dict[str, Any] = {
        'box_id': box_id,
        'bbox_norm': (x, 0.1, x + 0.05, 0.3),
        'state': state,
    }
    kwargs.update(over)
    return RegionBox(**kwargs)


def _item(crop_id: str, boxes: list[RegionBox], **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        **boxes_write_fields(boxes, current_src={}),
        F.status: 'detected',
        **extra,
    }


def _rows(body: dict[str, Any]) -> list[tuple[str, str | None]]:
    return [(r['crop_id'], r['region_box_id']) for r in body['items']]


def _get(docs: dict[str, dict[str, Any]] | TestClient, path: str, **params: Any) -> dict[str, Any]:
    """GET ``path``; ``docs`` is a doc set (one fake per call) or a client
    to reuse -- a test making several calls reuses one, since the first
    request's index bootstrap raises that fake's inner-hits window and the
    process remembers it."""
    client = docs if isinstance(docs, TestClient) else _client(QueryFakeOpenSearch({ITEMS: docs}))
    resp = client.get(f'{PREFIX}{path}', params=params)
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


def test_each_matching_box_is_its_own_row_and_total_counts_items() -> None:
    docs = {
        'two': _item('two', [_box('b1'), _box('b2'), _box('b3', 'rejected')]),
        'one': _item('one', [_box('b1')]),
        'none': _item('none', [_box('b1', 'rejected')]),
    }

    body = _get(docs, '/regions', page_size=50)

    assert sorted(_rows(body)) == [('one', 'b1'), ('two', 'b1'), ('two', 'b2')]
    assert body['total'] == 2
    assert body['total_rows'] == 3
    assert body['page_size'] == 50


def test_box_filters_select_the_same_box_not_any_box() -> None:
    # `mixed` has a det_a box scored 0.5 and a det_b box scored 0.9: no
    # single box is det_a AND >= 0.8. `match` has one.
    docs = {
        'mixed': _item(
            'mixed',
            [_box('b1', detector='det_a', score=0.5), _box('b2', detector='det_b', score=0.9)],
        ),
        'match': _item(
            'match',
            [_box('b1', detector='det_b', score=0.2), _box('b2', detector='det_a', score=0.9)],
        ),
    }

    body = _get(docs, '/regions', detector='det_a', min_score=0.8)

    assert _rows(body) == [('match', 'b2')]
    assert body['total'] == 1
    assert body['total_rows'] == 1


def test_cluster_filters_and_state_match_the_same_box() -> None:
    docs = {
        'x': _item(
            'x',
            [
                _box('b1', cluster_id=7),
                _box('b2', 'false_positive', cluster_id=FALSE_POSITIVE_REGION_CLUSTER_ID),
            ],
        ),
    }

    client = _client(QueryFakeOpenSearch({ITEMS: docs}))
    in_bucket = _get(client, '/regions', region_cluster_id=7)
    assert _rows(in_bucket) == [('x', 'b1')]
    # b1 is accepted but in bucket 7; the FP box is not in bucket 7.
    fp_in_seven = _get(client, '/regions', region_cluster_id=7, box_state='false_positive')
    assert _rows(fp_in_seven) == []
    fp = _get(
        client,
        '/regions',
        region_cluster_id=FALSE_POSITIVE_REGION_CLUSTER_ID,
        box_state='false_positive',
    )
    assert _rows(fp) == [('x', 'b2')]


def test_box_state_filter_selects_rejected_boxes() -> None:
    docs = {
        'x': _item('x', [_box('b1'), _box('b2', 'rejected', rejection_reason='verifier_rejected')])
    }

    body = _get(docs, '/regions', box_state='rejected')

    assert _rows(body) == [('x', 'b2')]


def test_status_filter_alone_returns_item_rows_with_a_null_box_id() -> None:
    docs = {
        'rej': _item(
            'rej', [_box('b1', 'rejected'), _box('b2', 'rejected')], **{F.status: 'verify_rejected'}
        ),
        'det': _item('det', [_box('b1')]),
    }

    body = _get(docs, '/regions', status='verify_rejected')

    assert _rows(body) == [('rej', None)]
    assert body['total'] == body['total_rows'] == 1


def test_status_plus_box_filter_is_one_row_per_matching_box_of_those_items() -> None:
    docs = {
        'rej': _item(
            'rej',
            [_box('b1', 'rejected', detector='det_a'), _box('b2', 'rejected', detector='det_b')],
            **{F.status: 'verify_rejected'},
        ),
    }

    body = _get(docs, '/regions', status='verify_rejected', detector='det_b')

    assert _rows(body) == [('rej', 'b2')]


def test_unknown_box_state_is_400() -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: {}}))
    assert client.get(f'{PREFIX}/regions', params={'box_state': 'bogus'}).status_code == 400


def test_a_row_carries_the_full_item_with_its_box_list() -> None:
    docs = {'x': _item('x', [_box('b1'), _box('b2')])}

    body = _get(docs, '/regions')

    row = body['items'][0]
    assert [b['box_id'] for b in row['region_boxes']] == ['b1', 'b2']
    assert row['region_count'] == 2


def test_training_cohort_per_box_modes_return_a_row_per_box() -> None:
    from src.services.detection.profile_registry import region_profile_or_neutral

    seg = region_profile_or_neutral().segmenter_name
    docs = {
        'blind': _item(
            'blind',
            [_box('b1', detector=seg), _box('b2', detector=seg), _box('b3', detector='other')],
            **{F.verified: True, F.detector_chain: [f'{REFERENCE_REGION_DETECTOR_MODEL}:miss']},
        ),
    }

    body = _get(docs, '/regions/training_candidates', mode='detector_blind_spots')

    assert sorted(_rows(body)) == [('blind', 'b1'), ('blind', 'b2')]
    assert body['total'] == 1
    assert body['total_rows'] == 2
    assert {r['selection_reason'] for r in body['items']} == {body['selection_reason']}


def test_training_cohort_item_modes_return_one_item_row() -> None:
    docs = {
        'both': _item(
            'both',
            [_box('b1'), _box('b2')],
            **{F.detector_chain: ['det:hit', 'seg:hit'], F.label_source: 'human'},
        )
    }

    body = _get(docs, '/regions/training_candidates', mode='human_corrected')

    assert _rows(body) == [('both', None)]
    assert body['total'] == body['total_rows'] == 1


def test_fp_cohort_lists_the_fp_box_of_a_detected_item() -> None:
    docs = {'mixed': _item('mixed', [_box('b1'), _box('b2', 'false_positive')])}

    body = _get(docs, '/regions/training_candidates', mode='false_positives')

    assert _rows(body) == [('mixed', 'b2')]


# ---------------------------------------------------------------------------
# inner_hits window
# ---------------------------------------------------------------------------


def _many_boxes_doc(n: int) -> dict[str, dict[str, Any]]:
    return {'big': _item('big', [_box(f'b{i + 1}') for i in range(n)])}


@pytest.mark.asyncio
async def test_inner_window_ensure_issues_one_put_when_low_and_none_when_enough() -> None:
    fake = QueryFakeOpenSearch({ITEMS: {}})
    limit = CFG.region_max_boxes_per_write

    first = await ensure_items_inner_result_window(fake)
    second = await ensure_items_inner_result_window(fake)

    assert first['max_inner_result_window'] == limit
    assert second['max_inner_result_window'] == limit
    assert len(fake.settings_puts) == 1
    assert fake.settings_puts[0]['body'] == {'index': {'max_inner_result_window': limit}}
    assert inner_result_window(ITEMS) == limit


@pytest.mark.asyncio
async def test_inner_window_ensure_never_lowers_a_larger_window() -> None:
    fake = QueryFakeOpenSearch({ITEMS: {}})
    fake.max_inner_result_window = 10_000

    result = await ensure_items_inner_result_window(fake)

    assert result['max_inner_result_window'] == 10_000
    assert fake.settings_puts == []


@pytest.mark.asyncio
async def test_an_item_with_150_matching_boxes_returns_150_rows_once_the_window_is_raised() -> None:
    fake = QueryFakeOpenSearch({ITEMS: _many_boxes_doc(150)})
    await ensure_items_inner_result_window(fake)
    client = _client(fake)

    resp = client.get(f'{PREFIX}/regions', params={'page_size': 10})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body['items']) == 150
    assert body['total'] == 1
    assert body['total_rows'] == 150
    assert body['rows_truncated'] is False


def test_an_item_with_more_matching_boxes_than_the_window_reports_truncation() -> None:
    # No ensure step ran: the window is OpenSearch's default 100.
    client = _client(QueryFakeOpenSearch({ITEMS: _many_boxes_doc(150)}))

    resp = client.get(f'{PREFIX}/regions', params={'page_size': 10})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body['items']) == 100
    assert body['total_rows'] == 150
    assert body['rows_truncated'] is True


def test_inner_hits_size_clamps_to_the_window_the_ensure_step_read_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.clients.curation_opensearch import ensure_overlay_fields
    from src.services.curation.region_rows import inner_hits_size

    limit = CFG.region_max_boxes_per_write
    # Never read back (the ensure step failed): OpenSearch's default, so the
    # request can't 400.
    assert inner_hits_size(ITEMS) == 100
    monkeypatch.setitem(ensure_overlay_fields._INNER_RESULT_WINDOWS, ITEMS, 250)
    assert inner_hits_size(ITEMS) == min(limit, 250)
    monkeypatch.setitem(ensure_overlay_fields._INNER_RESULT_WINDOWS, ITEMS, limit + 1000)
    assert inner_hits_size(ITEMS) == limit
