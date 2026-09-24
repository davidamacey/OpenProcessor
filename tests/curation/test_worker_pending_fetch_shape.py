"""F-20: both the VLM worker and the region/cascade worker's pending-batch
fetch push in-flight exclusion server-side (``must_not: {ids: ...}``)
instead of over-fetching ``batch_size + len(in_flight)`` docs and
filtering in Python, set ``track_total_hits: False``, sort with a
``crop_id`` tiebreaker, and (VLM worker only) merge what used to be 5
separate ``term`` clauses on ``class_source`` into one ``terms`` clause.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    'vlm_worker_module_f20', _REPO_ROOT / 'scripts' / 'curation' / 'vlm_worker.py'
)
assert _SPEC is not None
assert _SPEC.loader is not None
vlm_worker = importlib.util.module_from_spec(_SPEC)
sys.modules['vlm_worker_module_f20'] = vlm_worker
_SPEC.loader.exec_module(vlm_worker)


# =============================================================================
# VLM worker — _build_pending_query / fetch_pending_ids
# =============================================================================


def _count_term_clauses_on_field(query: Any, field: str) -> int:
    n = 0
    if isinstance(query, dict):
        if 'term' in query and field in query['term']:
            n += 1
        for v in query.values():
            n += _count_term_clauses_on_field(v, field)
    elif isinstance(query, list):
        for v in query:
            n += _count_term_clauses_on_field(v, field)
    return n


def test_vlm_worker_class_source_exclusions_are_one_terms_clause() -> None:
    query = vlm_worker._build_pending_query(0.8)
    must_not = query['bool']['must_not']
    # No separate `term` clauses on class_source left.
    assert _count_term_clauses_on_field(must_not, 'class_source') == 0
    terms_clause = next(
        c['terms']['class_source']
        for c in must_not
        if 'terms' in c and 'class_source' in c.get('terms', {})
    )
    assert set(terms_clause) == {
        'vlm',
        'classifier_vlm_agreement',
        'cluster_majority_agreement',
        'vlm_unmatched',
        'vlm_new_class_pending',
    }


def test_vlm_worker_build_pending_query_pushes_exclude_ids_server_side() -> None:
    query = vlm_worker._build_pending_query(0.8, exclude_ids=['a', 'b'])
    must_not = query['bool']['must_not']
    assert {'ids': {'values': ['a', 'b']}} in must_not


def test_vlm_worker_build_pending_query_omits_ids_clause_when_no_exclusions() -> None:
    query = vlm_worker._build_pending_query(0.8)
    must_not = query['bool']['must_not']
    assert not any('ids' in c for c in must_not)


@pytest.mark.asyncio
async def test_vlm_worker_fetch_pending_ids_query_shape() -> None:
    captured: dict[str, Any] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {'hits': {'hits': [{'_id': 'c1'}, {'_id': 'c2'}]}}

    class _FakeClient:
        async def post(
            self,
            _url: str,
            *,
            json: dict[str, Any],
            timeout: float,  # noqa: ARG002
        ) -> _FakeResponse:
            captured['body'] = json
            return _FakeResponse()

    ids = await vlm_worker.fetch_pending_ids(
        _FakeClient(),
        opensearch_url='http://os:9200',
        batch_size=64,
        v6_skip_conf=0.8,
        exclude_ids=['x1', 'x2'],
    )
    assert ids == ['c1', 'c2']
    body = captured['body']
    assert body['_source'] is False
    assert 'stored_fields' not in body
    assert body['track_total_hits'] is False
    assert body['sort'] == [
        {'created_at': {'order': 'asc', 'unmapped_type': 'date'}},
        {'crop_id': 'asc'},
    ]
    assert {'ids': {'values': ['x1', 'x2']}} in body['query']['bool']['must_not']


# =============================================================================
# Region/cascade worker — _build_pending_query / _fetch_pending
# =============================================================================


def test_cascade_worker_pending_query_uses_filter_context() -> None:
    from scripts.curation.worker.cascade import _build_pending_query

    query = _build_pending_query()
    b = query['bool']
    assert 'must' not in b  # F-20: filter context, not must — nothing here scores.
    assert len(b['filter']) == 3


def test_cascade_worker_pending_query_pushes_exclude_ids_server_side() -> None:
    from scripts.curation.worker.cascade import _build_pending_query

    query = _build_pending_query(exclude_ids=['a', 'b'])
    assert {'ids': {'values': ['a', 'b']}} in query['bool']['must_not']


def test_cascade_worker_pending_query_omits_must_not_when_no_exclusions() -> None:
    from scripts.curation.worker.cascade import _build_pending_query

    query = _build_pending_query()
    assert 'must_not' not in query['bool']


@pytest.mark.asyncio
async def test_cascade_worker_fetch_pending_query_shape() -> None:
    from scripts.curation.worker.cascade import _fetch_pending

    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'hits': {'hits': []}})
    await _fetch_pending(os_client, batch_size=64, exclude_ids=['x1'])

    assert os_client.search.await_args is not None
    body = os_client.search.await_args.kwargs['body']
    assert body['track_total_hits'] is False
    assert body['sort'] == [
        {'created_at': {'order': 'asc', 'unmapped_type': 'date'}},
        {'crop_id': 'asc'},
    ]
    assert {'ids': {'values': ['x1']}} in body['query']['bool']['must_not']
    # _source stays an explicit includes list (this worker needs bbox etc).
    assert isinstance(body['_source'], list)
    assert 'bbox_norm' in body['_source']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
