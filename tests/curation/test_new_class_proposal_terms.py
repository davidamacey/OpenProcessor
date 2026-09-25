"""DQ-M11: new-class proposals are one selection, and junk terms are flagged.

- ``GET /review/new_class_proposals/summary`` counts exactly the items
  ``GET /review/new_class_proposals`` serves (it used to count only
  ``vlm_new_class_pending`` rows, ignoring human / VLM ``needs_new_class``
  flags and dismissed / excluded items, so 164 != 191 live).
- A resolve for a term matches exactly the summary's count for that term.
- Terms that name a generic parent of registry classes, a non-object
  (blur, empty scene, ...), or an existing registry class are flagged by a
  served, configurable rule instead of being offered as a new class.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import get_curation_config
from src.services.curation.new_class_terms import ProposalTermRules, classify_term


ITEMS = get_curation_config().items_index


@pytest.fixture
def registry(tmp_path: Any) -> ClassRegistry:
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('sedan', group='cars')
    reg.add_class('class_b', group='group_b')
    return reg


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


def _pending(crop_id: str, label: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_source': 'vlm_new_class_pending',
        'vlm_proposed_class': label,
        'needs_new_class': True,
        **extra,
    }


def _docs() -> dict[str, dict[str, Any]]:
    return {
        'p1': _pending('p1', 'sidecar'),
        'p2': _pending('p2', 'sidecar'),
        # A VLM-labelled row that also asked for a new class.
        'u1': {
            'crop_id': 'u1',
            'class_source': 'vlm_unmatched',
            'vlm_proposed_class': 'sidecar',
            'needs_new_class': True,
        },
        # Human flag without a proposed name.
        'h1': {'crop_id': 'h1', 'class_source': 'vlm', 'needs_new_class': True},
        # Out of every review queue: must be out of the summary too.
        'dismissed': _pending('dismissed', 'sidecar', review_dismissed_at='2026-09-24T00:00:00Z'),
        'excluded': _pending('excluded', 'sidecar', class_excluded=True),
        'm1': _pending('m1', 'motorcycle'),
        'b1': _pending('b1', 'abstract_blur'),
        'e1': _pending('e1', 'empty_road'),
        'c1': _pending('c1', 'Cars'),
        's1': _pending('s1', 'sedan'),
    }


def test_summary_total_equals_the_queue_total(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), registry, monkeypatch)
    queue = client.get('/curation/review/new_class_proposals', params={'page_size': 100}).json()
    summary = client.get('/curation/review/new_class_proposals/summary').json()
    assert summary['total_pending'] == queue['total'] == 9
    assert summary['without_term'] == 1
    counted = sum(t['count'] for t in summary['top_terms'] + summary['flagged_terms'])
    assert counted + summary['without_term'] == summary['total_pending']


def test_term_count_equals_resolve_match(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), registry, monkeypatch)
    summary = client.get('/curation/review/new_class_proposals/summary').json()
    sidecar = {t['label']: t for t in summary['top_terms']}['sidecar']
    assert sidecar['count'] == 3
    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        params={'dry_run': True},
        json={'label': 'sidecar', 'create': {'class_name': 'sidecar', 'group': 'bikes'}},
    )
    assert r.status_code == 200, r.text
    assert r.json()['matched'] == sidecar['count']
    assert sorted(r.json()['matched_ids']) == ['p1', 'p2', 'u1']


def test_resolve_writes_the_vlm_flagged_row_too(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: _docs()})
    client = _client(fake, registry, monkeypatch)
    class_id = registry.load().classes[0].class_id
    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        json={'label': 'sidecar', 'class_id': class_id},
    )
    assert r.status_code == 200, r.text
    assert sorted(r.json()['updated_ids']) == ['p1', 'p2', 'u1']
    assert fake.docs(ITEMS)['u1']['needs_new_class'] is False
    # Out-of-queue items are never swept up by a resolve.
    assert fake.docs(ITEMS)['dismissed']['class_source'] == 'vlm_new_class_pending'
    assert fake.docs(ITEMS)['excluded']['class_source'] == 'vlm_new_class_pending'


def test_configured_and_registry_rules_flag_terms(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_NEW_CLASS_GENERIC_TERMS', 'motorcycle, vehicle')
    monkeypatch.setenv('OP_NEW_CLASS_NON_OBJECT_TERMS', 'blur,empty_road')
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), registry, monkeypatch)
    body = client.get('/curation/review/new_class_proposals/summary').json()

    assert [t['label'] for t in body['top_terms']] == ['sidecar']
    flagged = {t['label']: t for t in body['flagged_terms']}
    assert flagged['motorcycle']['flag'] == 'generic_parent'
    assert flagged['Cars']['flag'] == 'generic_parent'  # a registry group name
    assert flagged['abstract_blur']['flag'] == 'non_object'  # token match
    assert flagged['empty_road']['flag'] == 'non_object'  # whole-term match
    assert flagged['sedan']['flag'] == 'existing_class'
    assert flagged['sedan']['class_id'] == registry.load().classes[0].class_id
    assert all(t['flag'] is None for t in body['top_terms'])

    rules = body['term_rules']
    assert rules['generic_terms'] == ['motorcycle', 'vehicle']
    assert rules['non_object_terms'] == ['blur', 'empty_road']
    assert rules['registry_groups_are_generic'] is True
    assert rules['existing_classes_flagged'] is True


def test_no_configured_rule_flags_only_registry_matches(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_NEW_CLASS_GENERIC_TERMS', raising=False)
    monkeypatch.delenv('OP_NEW_CLASS_NON_OBJECT_TERMS', raising=False)
    client = _client(QueryFakeOpenSearch({ITEMS: _docs()}), registry, monkeypatch)
    body = client.get('/curation/review/new_class_proposals/summary').json()
    assert {t['label'] for t in body['flagged_terms']} == {'Cars', 'sedan'}
    assert body['term_rules']['generic_terms'] == []


def test_generic_terms_match_whole_terms_only() -> None:
    rules = ProposalTermRules(
        generic_terms=frozenset({'car'}),
        non_object_terms=frozenset(),
        registry_groups=frozenset(),
        existing_classes={},
    )
    assert classify_term('car', rules) == ('generic_parent', None)
    # A specific kind of car is a legitimate proposal, not its parent.
    assert classify_term('sports_car', rules) == (None, None)
