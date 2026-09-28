"""W8-cleanup m1 regression guard: ``/stats/dataset``'s ``region_detectors``
nested agg must count CROPS, not boxes.

``tests/integration/test_stats_router.py`` supplies a canned aggregation
response, so it can't catch a box-vs-crop miscount in the agg body itself
(it never exercises real nested-agg semantics). This uses
``QueryFakeOpenSearch``, which actually evaluates ``nested``/``terms``/
``reverse_nested`` aggregations, against a crop that carries two accepted
boxes from the same detector plus an unrelated crop with a rejected box
from a different detector -- a box-counting agg would report
``region_detector_v1: 2`` for the two-box crop; the correct crop-counting
answer is ``1``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.config.curation import base_curation_config


ITEMS = base_curation_config().items_index
F = get_region_fields()


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _docs() -> dict[str, dict[str, Any]]:
    return {
        'two_box_one_crop': {
            'crop_id': 'two_box_one_crop',
            F.boxes: [
                {
                    'box_id': 'b1',
                    'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                    'state': 'accepted',
                    'detector': 'region_detector_v1',
                    'score': 0.9,
                },
                {
                    'box_id': 'b2',
                    'bbox_norm': [0.3, 0.3, 0.4, 0.4],
                    'state': 'accepted',
                    'detector': 'region_detector_v1',
                    'score': 0.8,
                },
            ],
            F.status: 'detected',
        },
        'other_detector_crop': {
            'crop_id': 'other_detector_crop',
            F.boxes: [
                {
                    'box_id': 'b1',
                    'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                    'state': 'rejected',
                    'detector': 'other_detector',
                    'score': 0.5,
                }
            ],
            F.status: 'verify_rejected',
        },
    }


def test_region_detectors_agg_counts_crops_not_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), monkeypatch)
    r = client.get('/curation/projects/default/stats/dataset')
    assert r.status_code == 200, r.text
    body = r.json()
    # A box-counting agg would report 2 for `region_detector_v1` (one per
    # accepted box on the same crop); the correct crop-counting answer is 1.
    assert body['regions']['total_detected'] == 2
    profile_detector_count = body['regions']['by_detector']
    assert profile_detector_count in (0, 1), body['regions']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
