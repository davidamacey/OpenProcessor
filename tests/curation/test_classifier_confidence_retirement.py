"""F-6 / D-1: retire ``classifier_raw_confidence`` entirely.

It was never written anywhere in the live code path (only the seed/test
harness wrote it), so the ``primary_low_conf`` review tab, the
``/crops?classifier_conf_lt`` filter, and any sort on it were permanent
no-ops. This module asserts the field is gone from every surface it used
to touch, and that the replacement logic (the stored ``confidence``
field, gated by classifier class_source) is what's actually built.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import matches
from src.config.curation import PROBE_ENTROPY_REVIEW_MIN
from src.services.curation.crop_browse import CROP_SORT_FIELDS
from src.services.curation.ingest_class_sources import (
    classifier_class_sources,
    unlabeled_proposal_class_sources,
)
from src.services.curation.review_queries import build_tab_query
from src.services.curation.wire import ITEM_WIRE_KEYS


def _has_field(query: Any, field: str) -> bool:
    if isinstance(query, dict):
        if any(field in (v if isinstance(v, dict) else {}) for v in query.values()):
            return True
        return any(_has_field(v, field) for v in query.values())
    if isinstance(query, list):
        return any(_has_field(v, field) for v in query)
    return False


# =============================================================================
# Fully gone
# =============================================================================


def test_field_removed_from_wire_keys() -> None:
    assert 'classifier_raw_confidence' not in ITEM_WIRE_KEYS


def test_field_removed_from_crop_sort_fields() -> None:
    assert 'classifier_raw_confidence' not in CROP_SORT_FIELDS


@pytest.mark.parametrize(
    'tab',
    [
        'all',
        'mismatches',
        'vlm_low_conf',
        'outliers',
        'uncertainty',
        'regions',
        'model_disagreements',
        'primary_low_conf',
        'coco_blind_spots',
        'new_class_proposals',
    ],
)
def test_field_never_appears_in_any_tab_query(tab: str) -> None:
    must, must_not, _reason = build_tab_query(tab, include_test=False, text=None, max_rank=None)
    assert not _has_field({'must': must, 'must_not': must_not}, 'classifier_raw_confidence')


def test_outlier_flagged_removed_from_every_tab() -> None:
    for tab in (
        'all',
        'mismatches',
        'vlm_low_conf',
        'outliers',
        'uncertainty',
        'regions',
        'model_disagreements',
        'primary_low_conf',
        'coco_blind_spots',
        'new_class_proposals',
    ):
        must, must_not, _reason = build_tab_query(tab, include_test=False, text=None, max_rank=None)
        assert not _has_field({'must': must, 'must_not': must_not}, 'outlier_flagged')


# =============================================================================
# primary_low_conf: confidence + classifier_class_sources, OR
# unlabeled_proposal_class_sources
# =============================================================================


def test_primary_low_conf_gates_confidence_on_classifier_sources() -> None:
    must, _must_not, _reason = build_tab_query(
        'primary_low_conf', include_test=False, text=None, max_rank=None
    )
    should_clauses = None
    for clause in must:
        if 'bool' in clause and 'should' in clause['bool']:
            should_clauses = clause['bool']['should']
    assert should_clauses is not None
    assert {'terms': {'class_source': sorted(unlabeled_proposal_class_sources())}} in should_clauses
    classifier_gate = next(
        c
        for c in should_clauses
        if 'bool' in c and any('class_source' in m.get('terms', {}) for m in c['bool']['must'])
    )
    assert {'terms': {'class_source': sorted(classifier_class_sources())}} in classifier_gate[
        'bool'
    ]['must']
    assert any('confidence' in m.get('range', {}) for m in classifier_gate['bool']['must'])


def test_primary_low_conf_matches_a_low_confidence_classifier_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'src.services.curation.review_queries.classifier_class_sources',
        lambda: {'classifier_model'},
    )
    must, must_not, _reason = build_tab_query(
        'primary_low_conf', include_test=False, text=None, max_rank=None
    )
    query = {'bool': {'must': must, 'must_not': must_not}}
    doc = {
        'crop_rank_in_image': 1,
        'class_source': 'classifier_model',
        'confidence': 0.01,
        'class_validated': False,
        'class_excluded': False,
    }
    assert matches(doc, query)

    # A confident classifier item with the same source must NOT match.
    confident_doc = {**doc, 'confidence': 0.99}
    assert not matches(confident_doc, query)


def test_primary_low_conf_matches_a_blind_spot_with_no_classifier_source() -> None:
    must, must_not, _reason = build_tab_query(
        'primary_low_conf', include_test=False, text=None, max_rank=None
    )
    query = {'bool': {'must': must, 'must_not': must_not}}
    [proposal_source] = sorted(unlabeled_proposal_class_sources())[:1] or ['x']
    doc = {
        'crop_rank_in_image': 1,
        'class_source': proposal_source,
        'class_validated': False,
        'class_excluded': False,
    }
    assert matches(doc, query)


# =============================================================================
# 'all' tab: range probe_pred_entropy >= PROBE_ENTROPY_REVIEW_MIN, not
# `exists` (which matches almost everything post-probe-run)
# =============================================================================


def test_all_tab_gates_probe_entropy_on_a_threshold_not_mere_existence() -> None:
    must, _must_not, _reason = build_tab_query('all', include_test=False, text=None, max_rank=None)
    should_clauses = next(
        c['bool']['should'] for c in must if 'bool' in c and 'should' in c['bool']
    )
    assert {'range': {'probe_pred_entropy': {'gte': PROBE_ENTROPY_REVIEW_MIN}}} in should_clauses
    assert not any(
        'exists' in c and c['exists'].get('field') == 'probe_pred_entropy' for c in should_clauses
    )


def test_all_tab_excludes_a_low_entropy_item_that_only_has_the_field() -> None:
    """The load-bearing regression: before the fix, any item that merely
    *had* a probe_pred_entropy value (regardless of how low) matched the
    'all' tab via `exists`. A confidently-predicted item (low entropy)
    must not match anymore."""
    must, must_not, _reason = build_tab_query('all', include_test=False, text=None, max_rank=None)
    query = {'bool': {'must': must, 'must_not': must_not}}
    confident_item = {
        'probe_pred_entropy': 0.01,
        'class_id': 5,
        'pe_embedding': [0.1],
        'class_source': 'human',
        'vlm_confidence': 'high',
        'cluster_distance': 0.0,
    }
    assert not matches(confident_item, query)

    uncertain_item = {**confident_item, 'probe_pred_entropy': PROBE_ENTROPY_REVIEW_MIN + 0.1}
    assert matches(uncertain_item, query)


# =============================================================================
# /crops?classifier_conf_lt router-level query shape
# =============================================================================


def _crops_client(fake_os: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


class _FakeOS:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        self.bodies.append(body)
        return {'hits': {'total': {'value': 0}, 'hits': []}}


def test_classifier_conf_lt_filter_uses_confidence_field() -> None:
    fake = _FakeOS()
    client = _crops_client(fake)
    resp = client.get('/curation/crops', params={'classifier_conf_lt': 0.5})
    assert resp.status_code == 200, resp.text

    body = fake.bodies[-1]
    assert not _has_field(body['query'], 'classifier_raw_confidence')
    filt = body['query']['bool']['filter']
    gate = next(c for c in filt if 'bool' in c and 'should' in c['bool'])
    should = gate['bool']['should']
    assert {'terms': {'class_source': sorted(unlabeled_proposal_class_sources())}} in should
    classifier_branch = next(
        c
        for c in should
        if 'bool' in c and any('class_source' in m.get('terms', {}) for m in c['bool']['must'])
    )
    assert {'terms': {'class_source': sorted(classifier_class_sources())}} in classifier_branch[
        'bool'
    ]['must']
    assert {'range': {'confidence': {'lt': 0.5}}} in classifier_branch['bool']['must']
