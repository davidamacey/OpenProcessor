"""``/stats/dataset`` keeps imports out of the human and VLM counts and
counts them on their own (``validated_by_import`` and friends).

Runs the real aggregations through ``QueryFakeOpenSearch``, which evaluates
filter / nested / terms aggregations, over two human-reviewed items, one
imported item and one VLM-verified item.
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


def _box(detector: str) -> dict[str, Any]:
    return {
        'box_id': 'b1',
        'bbox_norm': [0.1, 0.1, 0.2, 0.2],
        'state': 'accepted',
        'detector': detector,
        'score': 1.0,
    }


def _item(crop_id: str, class_source: str, detector: str, verifier: str, **over: Any) -> Any:
    doc = {
        'crop_id': crop_id,
        'class_id': 3,
        'class_source': class_source,
        'class_validated': True,
        F.boxes: [_box(detector)],
        F.status: 'detected',
        F.validated: True,
        F.verifier: verifier,
    }
    doc.update(over)
    return doc


DOCS = {
    'human': _item('human', 'human', 'human', 'human'),
    'human2': _item('human2', 'human', 'human', 'human'),
    'imported': _item('imported', 'external_label', 'import', 'import'),
    'vlm': _item(
        'vlm', 'vlm', 'sam3', 'vision-model-1', class_validated=False, **{F.validated: False}
    ),
}


@pytest.fixture
def stats(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: QueryFakeOpenSearch({ITEMS: dict(DOCS)})
    r = TestClient(app).get('/curation/projects/default/stats/dataset')
    assert r.status_code == 200, r.text
    return r.json()


def test_class_labels_an_import_validated_have_their_own_count(stats: dict[str, Any]) -> None:
    assert stats['validated'] == 3  # the legacy total still counts all validated
    assert stats['validated_by_import'] == 1
    assert stats['labeled']['by_import'] == 1
    assert stats['labeled']['by_human'] == 2
    assert stats['labeled']['other'] == 0  # an import is not "unknown provenance"


def test_human_counts_exclude_imports(stats: dict[str, Any]) -> None:
    regions = stats['regions']
    assert regions['validated_by_human'] == 2
    assert regions['verified_by_human'] == 2
    assert regions['by_human_drew'] == 2


def test_import_region_counts(stats: dict[str, Any]) -> None:
    regions = stats['regions']
    assert regions['validated_by_import'] == 1
    assert regions['verified_by_import'] == 1
    assert regions['by_import'] == 1


def test_an_import_is_not_counted_as_the_vlm(stats: dict[str, Any]) -> None:
    # two human + one import + one VLM verifier: only the last is the VLM's.
    assert stats['regions']['verified_by_vlm'] == 1
