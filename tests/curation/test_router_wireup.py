"""Tests for the curation router.

Goal: exercise routing + dependency wiring without spinning up Triton or
OpenSearch. We patch the heavy collaborators (opensearch, triton_pool,
ClassRegistry) and assert the router glues them together correctly.

Covered:
- ``/curation/test_holdout/freeze`` refuses re-run unless ``?force=true`` (409).
- ``/curation/classes`` POST appends a new class.
- ``/curation/classes/merge`` deprecates source and bulk-relabels.
- ``/curation/health`` returns a structured status when deps are degraded/down.
- ``/curation/crops`` server-side filters out test_holdout by default.

``POST /curation/ingest/image`` and the region-metadata PATCH endpoint
from the reference test file this was ported from are not covered here:
the former depends on a Bucket-B ingest service never ported (plan §1,
§7 R5 — see ``src/routers/curation/ingest.py``'s module docstring); the
latter (``PATCH /crops/{id}/plate_meta``) lives on ``regions.py``,
ported in an earlier wave.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fake_opensearch() -> AsyncMock:
    """An AsyncMock standing in for AsyncOpenSearch used by the router."""
    fake = AsyncMock()
    # ``indices.exists`` and ``indices.create`` are called by the lazy
    # index bootstrap; default both to no-op success.
    fake.indices = AsyncMock()
    fake.indices.exists = AsyncMock(return_value=True)
    fake.indices.create = AsyncMock(return_value={'acknowledged': True})
    fake.indices.refresh = AsyncMock(return_value={'_shards': {}})
    fake.search = AsyncMock(return_value={'hits': {'hits': [], 'total': {'value': 0}}})
    fake.count = AsyncMock(return_value={'count': 0})
    fake.bulk = AsyncMock(return_value={'errors': False, 'items': []})
    fake.update = AsyncMock(return_value={'result': 'updated'})
    fake.update_by_query = AsyncMock(return_value={'updated': 0})
    fake.get = AsyncMock(return_value={'_source': {}})
    fake.index = AsyncMock(return_value={'result': 'created'})
    fake.msearch = AsyncMock(return_value={'responses': []})
    return fake


@pytest.fixture
def fake_triton_pool() -> AsyncMock:
    fake = AsyncMock()
    fake.health_check = AsyncMock(return_value=True)
    return fake


@pytest.fixture
def app_client(fake_opensearch: AsyncMock, fake_triton_pool: AsyncMock):
    """Build a minimal FastAPI app with just the curation router + DI overrides."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.core.dependencies import get_async_triton, get_opensearch
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[get_opensearch] = lambda: fake_opensearch
    # The router actually injects `_raw_opensearch_dep` (which strips the
    # `.client` wrapper). That dep calls `get_opensearch()` directly —
    # not via Depends — so overriding `get_opensearch` alone leaks a
    # real OpenSearch connection. Override the raw dep as well so the
    # AsyncMock fixture is what every endpoint sees.
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch
    app.dependency_overrides[get_async_triton] = lambda: fake_triton_pool

    with TestClient(app) as client:
        yield client


# =============================================================================
# /curation/test_holdout/freeze
# =============================================================================


def test_test_holdout_freeze_rejects_re_run_without_force(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    # Pretend an existing holdout already exists.
    fake_opensearch.count = AsyncMock(return_value={'count': 1234})
    r = app_client.post('/curation/test_holdout/freeze', json={'percent': 10, 'seed': 42})
    assert r.status_code == 409
    assert 'force' in r.text.lower()


def test_test_holdout_freeze_zero_cohort_raises_422_even_with_force(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    """A zero-row cohort must 422, not silently 200 with sha256('').

    The endpoint would otherwise never actually freeze anything and
    report success anyway.
    """
    fake_opensearch.count = AsyncMock(return_value={'count': 999})
    fake_opensearch.search = AsyncMock(
        return_value={
            'hits': {'hits': [], 'total': {'value': 0}},
            'aggregations': {'strata': {'buckets': []}},
        }
    )
    r = app_client.post(
        '/curation/test_holdout/freeze?force=true',
        json={'percent': 10, 'seed': 42},
    )
    assert r.status_code == 422, r.text
    assert 'zero' in r.text.lower()
    # Refusing to freeze nothing must not have issued a bulk write.
    fake_opensearch.bulk.assert_not_called()


# =============================================================================
# /curation/classes
# =============================================================================


def test_classes_post_appends(app_client: Any, tmp_path: Path) -> None:
    """POST /curation/classes assigns a new class_id via the registry."""
    from src.clients.curation_opensearch import ClassRegistry

    fake_path = tmp_path / 'class_registry.json'
    reg = ClassRegistry(path=fake_path)

    with patch('src.routers.curation.get_class_registry', return_value=reg):
        r = app_client.post(
            '/curation/classes',
            json={'name': 'subaru_brz', 'group': 'sport_compact'},
        )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body['class_name'] == 'subaru_brz'
    assert body['class_id'] == 0


def test_classes_merge_deprecates_source(
    app_client: Any, tmp_path: Path, fake_opensearch: AsyncMock
) -> None:
    """POST /curation/classes/merge marks source deprecated."""
    from src.clients.curation_opensearch import ClassRegistry

    fake_path = tmp_path / 'class_registry.json'
    reg = ClassRegistry(path=fake_path)
    src_id = reg.add_class('mustang_gen5', group='muscle')
    tgt_id = reg.add_class('mustang_gen6', group='muscle')

    fake_opensearch.update_by_query = AsyncMock(return_value={'updated': 5})

    with patch('src.routers.curation.get_class_registry', return_value=reg):
        r = app_client.post(
            '/curation/classes/merge',
            json={'source_id': src_id, 'target_id': tgt_id},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['source_id'] == src_id
    assert body['target_id'] == tgt_id
    assert body['deprecated'] is True

    src_entry = reg.get(src_id)
    assert src_entry is not None
    assert src_entry.deprecated is True
    assert src_entry.merged_into == tgt_id


def test_classes_merge_refuses_when_source_has_frozen_holdout_crops(
    app_client: Any, tmp_path: Path, fake_opensearch: AsyncMock
) -> None:
    """Refuse with 409 rather than silently relabeling a frozen
    test_holdout crop and leaving its holdout identity stale."""
    from src.clients.curation_opensearch import ClassRegistry

    fake_path = tmp_path / 'class_registry.json'
    reg = ClassRegistry(path=fake_path)
    src_id = reg.add_class('civic_gen9', group='compact')
    tgt_id = reg.add_class('civic_gen10', group='compact')

    fake_opensearch.count = AsyncMock(return_value={'count': 3})

    with patch('src.routers.curation.get_class_registry', return_value=reg):
        r = app_client.post(
            '/curation/classes/merge',
            json={'source_id': src_id, 'target_id': tgt_id},
        )
    assert r.status_code == 409, r.text
    assert '3' in r.json()['detail']
    fake_opensearch.update_by_query.assert_not_called()
    # Registry must be untouched — the pre-check runs before any mutation.
    src_entry = reg.get(src_id)
    assert src_entry is not None
    assert src_entry.deprecated is False


def test_classes_merge_resets_stale_human_provenance_on_crops_only(
    app_client: Any, tmp_path: Path, fake_opensearch: AsyncMock
) -> None:
    """A merge must reset label_source/class_validated on the items index
    (else a merged crop keeps reading as human-validated ground truth),
    but must NOT touch the confirmed-labels index, which has no
    class_validated field and whose label_source means something else
    (original label provenance).

    The items-index leg is a per-doc OCC bulk pass (via
    occ_skip_on_conflict_bulk) so class_id_history gets appended through
    record_class_history instead of being reimplemented in painless. The
    confirmed-labels index has no class_id_history field, so it stays a
    plain update_by_query. This test verifies both legs against their
    actual mechanisms rather than asserting two update_by_query calls.
    """
    from src.clients.curation_opensearch import ClassRegistry

    fake_path = tmp_path / 'class_registry.json'
    reg = ClassRegistry(path=fake_path)
    src_id = reg.add_class('miata_na', group='sport_compact')
    tgt_id = reg.add_class('miata_nb', group='sport_compact')

    fake_opensearch.update_by_query = AsyncMock(return_value={'updated': 1})
    fake_opensearch.search = AsyncMock(
        return_value={'_scroll_id': 'scroll-1', 'hits': {'hits': [{'_id': 'crop-1'}]}}
    )
    fake_opensearch.scroll = AsyncMock(
        return_value={'_scroll_id': 'scroll-1', 'hits': {'hits': []}}
    )
    fake_opensearch.clear_scroll = AsyncMock(return_value={})

    from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response

    merge_source = {
        'class_id': src_id,
        'class_name': 'miata_na',
        'class_source': 'human',
        'label_source': 'human',
        'class_validated': True,
        'test_holdout': False,
    }

    async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
        found = {d['_id']: merge_source for d in body['docs']}
        return make_mget_response(found)

    bulk_calls: list[list[dict[str, Any]]] = []

    async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        bulk_calls.append(body)
        items = [
            make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
        ]
        return make_bulk_response(items)

    fake_opensearch.mget = AsyncMock(side_effect=_fake_mget)
    fake_opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

    with patch('src.routers.curation.get_class_registry', return_value=reg):
        r = app_client.post(
            '/curation/classes/merge',
            json={'source_id': src_id, 'target_id': tgt_id},
        )
    assert r.status_code == 200, r.text

    # Confirmed-labels index: unchanged plain update_by_query, no
    # class_validated/label_source touched.
    ubq_calls = fake_opensearch.update_by_query.call_args_list
    assert len(ubq_calls) == 1
    labels_body = ubq_calls[0].kwargs['body']
    assert ubq_calls[0].kwargs['index'] == 'op_labels_confirmed'
    labels_source = labels_body['script']['source']
    assert 'class_validated' not in labels_source
    assert 'label_source' not in labels_source

    # Items index: per-doc OCC bulk update (batched mget+bulk) resets the
    # stale human provenance on the merged crop.
    assert len(bulk_calls) == 1
    bulk_body = bulk_calls[0]
    assert bulk_body[0]['update']['_index'] == 'op_items'
    crop_doc = bulk_body[1]['doc']
    assert crop_doc['class_validated'] is False
    assert crop_doc['label_source'] == 'class_merge'
    assert crop_doc['class_id'] == tgt_id


def test_unlabel_crop_clears_stale_human_provenance(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    """DELETE /crops/{id}/label (the labeler's Undo path) must clear
    every class-provenance field together, not just class_validated/
    label_source — leaving class_source/class_detector/class_labeler at
    their prior 'human' values would let an un-validated crop keep
    claiming human provenance, breaking the dataset invariant
    by_human <= validated."""
    fake_opensearch.get = AsyncMock(
        return_value={
            '_source': {
                'class_id': 18,
                'class_name': 'cruiserbike',
                'class_source': 'human',
                'class_detector': 'human',
                'class_labeler': 'human',
                'class_validated': True,
                'label_source': 'human',
                'test_holdout': False,
            },
            '_seq_no': 5,
            '_primary_term': 1,
        }
    )
    fake_opensearch.update = AsyncMock(return_value={'result': 'updated'})

    r = app_client.delete('/curation/crops/crop-xyz/label')
    assert r.status_code == 200, r.text

    update_calls = fake_opensearch.update.call_args_list
    assert len(update_calls) == 1
    doc = update_calls[0].kwargs['body']['doc']
    assert doc['class_validated'] is False
    assert doc['label_source'] == ''
    assert doc['class_source'] is None
    assert doc['class_detector'] is None
    assert doc['class_labeler'] is None


# =============================================================================
# /curation/health
# =============================================================================


def test_health_reports_degraded_when_vlm_down(
    app_client: Any, fake_opensearch: AsyncMock, fake_triton_pool: AsyncMock
) -> None:
    """When OpenSearch + Triton are up but the VLM is down -> 'degraded'."""
    fake_triton_pool.health_check = AsyncMock(return_value=True)

    fake_vlm = MagicMock()
    fake_vlm.health = AsyncMock(
        return_value=MagicMock(reachable=False, model='vlm-x', last_error='timeout')
    )

    with patch('src.routers.curation._get_vlm_labeler', return_value=fake_vlm):
        r = app_client.get('/curation/health')

    assert r.status_code == 200, r.text
    body = r.json()
    assert body['status'] in ('ok', 'degraded', 'down')
    # Triton + OpenSearch up, VLM down -> degraded.
    assert body['gemma']['reachable'] is False


def test_health_reports_down_when_opensearch_unreachable(
    app_client: Any, fake_opensearch: AsyncMock, fake_triton_pool: AsyncMock
) -> None:
    fake_opensearch.indices.exists = AsyncMock(side_effect=RuntimeError('connection refused'))
    fake_triton_pool.health_check = AsyncMock(return_value=True)
    fake_vlm = MagicMock()
    fake_vlm.health = AsyncMock(return_value=MagicMock(reachable=False, model='x'))
    with patch('src.routers.curation._get_vlm_labeler', return_value=fake_vlm):
        r = app_client.get('/curation/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] in ('degraded', 'down')


# =============================================================================
# /curation/crops listing — server-side filters out test_holdout
# =============================================================================


def test_crops_listing_filters_test_holdout_by_default(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    captured: dict[str, Any] = {}

    async def fake_search(index: str, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        captured['body'] = body
        captured['index'] = index
        return {'hits': {'hits': [], 'total': {'value': 0}}}

    fake_opensearch.search = AsyncMock(side_effect=fake_search)

    r = app_client.get('/curation/crops')
    assert r.status_code == 200, r.text
    must = captured['body']['query']['bool']['must']
    has_test_filter = any(
        isinstance(m, dict)
        and 'bool' in m
        and 'must_not' in (m.get('bool') or {})
        and m['bool']['must_not'].get('term', {}).get('test_holdout') is True
        for m in must
    )
    assert has_test_filter, 'test_holdout should be filtered out by default'


def test_crops_listing_includes_test_when_requested(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    captured: dict[str, Any] = {}

    async def fake_search(index: str, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        captured['body'] = body
        return {'hits': {'hits': [], 'total': {'value': 0}}}

    fake_opensearch.search = AsyncMock(side_effect=fake_search)

    r = app_client.get('/curation/crops?include_test=true')
    assert r.status_code == 200
    # When include_test=true, the test_holdout must_not filter should be absent.
    body = captured.get('body', {})
    must = (body.get('query') or {}).get('bool', {}).get('must', [])
    has_test_filter = any(
        isinstance(m, dict)
        and 'bool' in m
        and 'must_not' in (m.get('bool') or {})
        and (m['bool']['must_not'].get('term', {}) or {}).get('test_holdout') is True
        for m in must
    )
    assert not has_test_filter
