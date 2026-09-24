"""GET /curation/stats/dataset — dashboard contract test.

The reference this was ported from hit a live yolo-api instance over
HTTP (skipped when unreachable). Per plan §6.0's house rule ("do not
add the repo's first live-stack dependency" — write a unit test with
the I/O boundary faked instead, matching
``tests/integration/test_health_endpoints.py``'s style), this port
mounts the curation router directly with a faked OpenSearch client
instead of making a real network call, and asserts the same response
shape + internal-consistency invariants against synthetic aggregation
buckets.

Schema produced by ``stats._rollup_class_sources`` + ``stats_dataset``:

- ``labeled.{by_human, by_vlm, by_classifier, by_proposal, other}``
  — *class-label* provenance. ``by_human`` counts crops whose
  ``class_source`` starts with ``human``; auto-validation
  (a majority-agreement rule) lives in ``by_classifier``, deliberately
  separate from ``by_human`` so the dashboard can distinguish
  "human applied this label" from "any validator approved it".
- ``regions.{total_detected, by_detector, by_segmenter, by_human}`` —
  *region-detector* provenance. ``by_detector`` is the primary region
  detector's count and intentionally lives here (not under ``labeled``)
  because it's a region detector, not a class labeler.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def _fake_dataset_search_response() -> dict[str, Any]:
    """A synthetic aggregation response covering every bucket the
    endpoint reads, sized so the invariants below are checkable by hand.
    """
    return {
        'hits': {'total': {'value': 10}},
        'aggregations': {
            'by_source': {'buckets': [{'key': 'hdd1', 'doc_count': 10}]},
            'validated': {'doc_count': 4},
            'test_holdout': {'doc_count': 1},
            'class_sources': {
                'buckets': [
                    {'key': 'human', 'doc_count': 3},
                    {'key': 'human_move', 'doc_count': 1},
                    {'key': 'item_model', 'doc_count': 2},
                    {'key': 'coco_yolo11_proposal', 'doc_count': 2},
                    {'key': '__none__', 'doc_count': 2},
                ],
            },
            'region_detectors': {'buckets': [{'key': 'lpr_detector_v1', 'doc_count': 3}]},
            'region_verifiers': {'buckets': [{'key': 'human', 'doc_count': 2}]},
            'regions_validated_by_human': {'doc_count': 2},
            'region_status': {'buckets': [{'key': 'detected', 'doc_count': 3}]},
            'region_boxed': {'doc_count': 3},
            'no_label_source': {'doc_count': 0},
            'distinct_clusters': {'value': 5},
            'noise_clusters': {'doc_count': 0},
        },
    }


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = AsyncMock()
    fake_os.search = AsyncMock(return_value=_fake_dataset_search_response())
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def test_stats_dataset_endpoint_responds_with_full_schema(app_client: TestClient) -> None:
    """Every documented top-level + nested key is present and well-typed."""
    resp = app_client.get('/curation/stats/dataset')
    assert resp.status_code == 200, f'unexpected {resp.status_code}: {resp.text[:500]}'
    body = resp.json()

    assert isinstance(body.get('as_of'), str)
    assert body['as_of']
    assert isinstance(body.get('total_crops'), int)

    labeled = body.get('labeled')
    assert isinstance(labeled, dict)
    for k in ('by_human', 'by_vlm', 'by_classifier', 'by_proposal', 'other'):
        assert k in labeled, f'labeled.{k} missing'
        assert isinstance(labeled[k], int)
        assert labeled[k] >= 0

    assert 'plates' not in body
    regions = body.get('regions')
    assert isinstance(regions, dict), 'regions section missing'
    for k in (
        'total_detected',
        'by_detector',
        'by_segmenter',
        'by_human',
        'verified_by_human',
        'verified_by_vlm',
    ):
        assert k in regions, f'regions.{k} missing'
        assert isinstance(regions[k], int)
        assert regions[k] >= 0
    for old in ('by_lpr', 'by_sam3', 'verified_by_' + 'gemma'):
        assert old not in regions

    unlabeled = body.get('unlabeled')
    assert isinstance(unlabeled, dict)
    for k in ('pending_detection', 'pending_verification', 'no_label_source'):
        assert k in unlabeled, f'unlabeled.{k} missing'
        assert isinstance(unlabeled[k], int)
        assert unlabeled[k] >= 0

    in_progress = body.get('in_progress')
    assert isinstance(in_progress, dict)
    assert isinstance(in_progress.get('sam_drain_total_unfinished'), int)
    assert in_progress['sam_drain_total_unfinished'] >= 0

    clusters = body.get('clusters')
    assert isinstance(clusters, dict)
    # last_run_at may be None if no auto_label has ever run.
    assert 'last_run_at' in clusters
    assert isinstance(clusters.get('cluster_count'), int)
    assert isinstance(clusters.get('residual_count'), int)
    assert isinstance(clusters.get('noise_count'), int)
    # method is None | str.
    assert clusters.get('method') is None or isinstance(clusters['method'], str)


def test_stats_dataset_labeled_counts_consistent(app_client: TestClient) -> None:
    """Roll-up math is internally consistent.

    ``labeled.by_human`` counts only crops where the *labeler* was a
    human (``class_source`` startswith ``human``) — auto-validation
    (e.g. a cluster-majority rule -> ``class_validated=True`` with
    ``class_source='cluster_majority_agreement'``) is counted under ``by_classifier``, NOT
    ``by_human``. So ``by_human`` is bounded above by ``validated`` but
    is typically much smaller.

    What we can assert:
    - every bucket is non-negative,
    - the buckets sum to ``total_crops`` (every crop has exactly one
      ``class_source`` bucket assigned),
    - ``by_human <= validated`` (humans always validate when they
      label).
    """
    body = app_client.get('/curation/stats/dataset').json()
    labeled = body['labeled']
    total = int(body.get('total_crops', 0))
    validated = int(body.get('validated', 0))

    bucket_sum = sum(int(labeled[k]) for k in labeled)
    assert bucket_sum == total, f'labeled.* buckets sum to {bucket_sum} but total_crops={total}'

    by_human = int(labeled['by_human'])
    assert by_human >= 0
    assert by_human <= validated, (
        f'by_human ({by_human}) exceeds validated ({validated}); '
        'humans always validate when they label, so this is impossible'
    )

    # sam_drain_total_unfinished equals pending_detection + pending_verification
    # by construction.
    pending_d = int(body['unlabeled']['pending_detection'])
    pending_v = int(body['unlabeled']['pending_verification'])
    sam_total = int(body['in_progress']['sam_drain_total_unfinished'])
    assert sam_total == pending_d + pending_v, (
        f'sam_drain_total_unfinished={sam_total} '
        f'!= pending_detection({pending_d}) + pending_verification({pending_v})'
    )


def test_stats_dataset_legacy_keys_preserved(app_client: TestClient) -> None:
    """Legacy fields the labeler ``getStats`` adapter reads still exist."""
    body = app_client.get('/curation/stats/dataset').json()
    assert 'total_crops' in body
    assert 'validated' in body
    assert 'test_holdout' in body
    by_source = body.get('by_source')
    assert isinstance(by_source, list)
    # each bucket should be {'key': str, 'doc_count': int}
    for b in by_source:
        assert isinstance(b, dict)
        assert 'key' in b
        assert 'doc_count' in b
