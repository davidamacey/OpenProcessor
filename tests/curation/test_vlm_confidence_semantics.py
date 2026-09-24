"""DQ-M8: the "VLM low confidence" queue and the confidence served next to
a label use the confidence of the writer that set the label.

Before the fix the tab also required ``confidence < 0.80`` — the
detector/classifier score, not the VLM's — so VLM-low items on a
confidently-detected crop never reached it, and the only number a client
could print beside a VLM label was that detector score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_curation_config
from src.services.curation.class_sources import (
    VLM_CATEGORY_SCORE,
    class_confidence,
    class_source_catalog,
)
from src.services.curation.ingest_class_sources import (
    classifier_class_sources,
    is_classifier_class_source,
)
from src.services.curation.review_queries import TAB_LABELS
from src.services.curation.wire import serialize_item


if TYPE_CHECKING:
    from collections.abc import Iterator


ITEMS = get_curation_config().items_index
MODEL_SOURCE = 'secondary_model'


@pytest.fixture(autouse=True)
def _fresh_sort_coverage() -> Iterator[None]:
    from src.services.curation.strategy_registry import _reset_field_coverage_cache

    _reset_field_coverage_cache()
    yield
    _reset_field_coverage_cache()


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setattr(
        'src.services.curation.strategy_registry.resolve_effective_default',
        AsyncMock(return_value=None),
    )
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _docs() -> dict[str, dict[str, Any]]:
    return {
        # VLM unsure on a crop the detector scored highly: must be queued.
        'vlm_low_detector_high': {
            'class_source': 'vlm',
            'vlm_confidence': 'low',
            'confidence': 0.95,
        },
        'vlm_medium': {'class_source': 'vlm', 'vlm_confidence': 'medium', 'confidence': 0.3},
        'vlm_new_pending_low': {
            'class_source': 'vlm_new_class_pending',
            'vlm_confidence': 'low',
            'confidence': 0.9,
        },
        'vlm_high': {'class_source': 'vlm', 'vlm_confidence': 'high', 'confidence': 0.3},
        # Relabelled by a classifier after the VLM ran: the stale VLM
        # category no longer describes the current label.
        'stale_vlm_on_model_label': {
            'class_source': MODEL_SOURCE,
            'vlm_confidence': 'low',
            'confidence': 0.3,
        },
    }


def test_vlm_low_conf_selects_on_the_vlm_confidence_only(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {k: {'crop_id': k, **v} for k, v in _docs().items()}
    client = _client(QueryFakeOpenSearch({ITEMS: docs}), monkeypatch)
    r = client.get('/curation/review/vlm_low_conf', params={'page_size': 50})
    assert r.status_code == 200, r.text
    assert {i['crop_id'] for i in r.json()['items']} == {
        'vlm_low_detector_high',
        'vlm_medium',
        'vlm_new_pending_low',
    }


def test_vlm_label_serves_the_vlm_confidence() -> None:
    item = serialize_item(
        {'class_source': 'vlm', 'vlm_confidence': 'low', 'confidence': 0.946}, 'x', api_prefix=''
    )
    assert item['class_confidence'] == VLM_CATEGORY_SCORE['low']
    assert item['class_confidence_source'] == 'vlm'
    # The detector/classifier score stays available, under its own meaning.
    assert item['confidence'] == pytest.approx(0.946)


def test_classifier_label_serves_the_classifier_score() -> None:
    item = serialize_item({'class_source': MODEL_SOURCE, 'confidence': 0.61}, 'x', api_prefix='')
    assert item['class_confidence'] == pytest.approx(0.61)
    assert item['class_confidence_source'] == 'model'


@pytest.mark.parametrize('source', ['human', 'human_move', 'class_merge', 'unlabeled_proposal'])
def test_labels_without_a_machine_confidence_serve_null(source: str) -> None:
    assert class_confidence({'class_source': source, 'confidence': 0.9}) == (None, None)


def test_unknown_vlm_category_is_null_not_a_guess() -> None:
    assert class_confidence({'class_source': 'vlm', 'confidence': 0.9}) == (None, None)
    assert class_confidence({'class_source': 'vlm', 'vlm_confidence': 'certain'}) == (None, None)


def test_tab_description_names_the_vlm_confidence() -> None:
    assert 'VLM' in TAB_LABELS['vlm_low_conf'][1]


def test_classifier_source_recognition_matches_the_profile_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'some_classifier')
    configured = classifier_class_sources()
    assert configured
    assert all(is_classifier_class_source(s) for s in configured)
    # No other writer's value may be mistaken for a classifier.
    others = {e['id'] for e in class_source_catalog()} - configured
    assert not [s for s in others if is_classifier_class_source(s)]
