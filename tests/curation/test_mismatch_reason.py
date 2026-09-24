"""DQ-m4: the Mismatches queue says why each item is there.

A ``vlm_unmatched`` item whose VLM answer *is* a registry class name (the
label-batch path routes low-confidence answers to review instead of
applying them) was served the reason "VLM's reply did not match any
registry class", which is false for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import get_curation_config
from src.services.curation.review_queries import mismatch_reason


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


ITEMS = get_curation_config().items_index


@pytest.fixture(autouse=True)
def _fresh_sort_coverage() -> Iterator[None]:
    from src.services.curation.strategy_registry import _reset_field_coverage_cache

    _reset_field_coverage_cache()
    yield
    _reset_field_coverage_cache()


def _client(fake: Any, registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)
    monkeypatch.setattr(
        'src.services.curation.strategy_registry.resolve_effective_default',
        AsyncMock(return_value=None),
    )
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_mismatch_reasons_per_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('suv')
    docs = {
        'named': {
            'crop_id': 'named',
            'class_source': 'vlm_unmatched',
            'vlm_raw_class': 'SUV',
            'vlm_confidence': 'low',
        },
        'outside': {
            'crop_id': 'outside',
            'class_source': 'vlm_unmatched',
            'vlm_raw_class': 'zeppelin',
        },
        'blank': {'crop_id': 'blank', 'class_source': 'vlm_unmatched', 'vlm_raw_class': ''},
    }
    client = _client(QueryFakeOpenSearch({ITEMS: docs}), reg, monkeypatch)
    items = client.get('/curation/review/mismatches', params={'page_size': 50}).json()['items']
    reasons = {i['crop_id']: i['reason'] for i in items}
    assert 'did not match' not in reasons['named']
    assert "'SUV'" in reasons['named']
    assert 'low' in reasons['named']
    assert reasons['outside'] == "VLM's reply did not match any registry class"
    assert 'no class answer' in reasons['blank']


def test_mismatch_reason_rules() -> None:
    names = frozenset({'suv'})
    default = 'default'
    assert mismatch_reason({'vlm_raw_class': 'zeppelin'}, names, default) == default
    assert 'registry class' in mismatch_reason({'vlm_raw_class': 'suv'}, names, default)
    # The raw label is the fallback when no raw class was stored.
    assert 'registry class' in mismatch_reason({'vlm_raw_label': 'suv'}, names, default)
    assert 'no class answer' in mismatch_reason({}, names, default)
