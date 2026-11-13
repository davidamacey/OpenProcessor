"""The five plain review queues: served filter presets, in order, with counts under the filter.

``GET /review/tabs`` serves the queues (To label, Check machine labels, Disagreements, New class
proposals, Audit sample, plus the region queue while a region profile is active); a queue that
cannot fill is absent. The other tab ids stay valid only as ``review_tab`` filter values.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _msearch_fake import install_msearch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation import query_fakes
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.services.curation import review_sorts
from src.services.curation.review_tab_clauses import (
    DETECTOR_DISAGREES_SCRIPT,
    MODEL_DISAGREES_SCRIPT,
)
from src.services.curation.workflow_steps import WorkflowFacts, build_steps


ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default'


def _disagrees(left: str, right: str):
    def predicate(doc: dict[str, Any]) -> bool:
        return bool(doc.get(left)) and bool(doc.get(right)) and doc[left] != doc[right]

    return predicate


@pytest.fixture(autouse=True)
def _script_twins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        query_fakes.SCRIPT_EVALUATORS,
        DETECTOR_DISAGREES_SCRIPT,
        _disagrees('detector_class_name', 'class_name'),
    )
    monkeypatch.setitem(
        query_fakes.SCRIPT_EVALUATORS,
        MODEL_DISAGREES_SCRIPT,
        _disagrees('probe_pred_class', 'class_name'),
    )


def _client(docs: dict[str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake = QueryFakeOpenSearch({ITEMS: docs})
    install_msearch(fake)
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        'src.services.curation.strategy_registry.resolve_effective_default',
        AsyncMock(return_value=None),
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _item(crop_id: str, **kw: Any) -> dict[str, Any]:
    return {'crop_id': crop_id, **kw}


def _machine(crop_id: str, **kw: Any) -> dict[str, Any]:
    defaults = {
        'class_id': 1,
        'class_name': 'car',
        'class_source': 'vlm',
        'class_validated': False,
        'confidence': 0.5,
    }
    return _item(crop_id, **{**defaults, **kw})


def _ids(response: Any, key: str = 'items') -> list[str]:
    assert response.status_code == 200, response.text
    return [i['crop_id'] for i in response.json()[key]]


def _tabs(client: TestClient, **params: Any) -> list[dict[str, Any]]:
    response = client.get(f'{BASE}/review/tabs', params=params)
    assert response.status_code == 200, response.text
    return response.json()['tabs']


@pytest.mark.usefixtures('reference_region_profile')
def test_served_queues_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {'a': _machine('a', audit_sample=True)}
    tabs = _tabs(_client(docs, monkeypatch))
    assert [t['id'] for t in tabs] == [
        'to_label',
        'check_machine',
        'disagreements',
        'new_class_proposals',
        'audit',
        'regions',
    ]
    assert [t['label'] for t in tabs[:5]] == [
        'To label',
        'Check machine labels',
        'Disagreements',
        'New class proposals',
        'Audit sample',
    ]


def test_to_label_is_items_without_a_class(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {
        'bare_low': _item('bare_low', detector_confidence=0.2),
        'bare_high': _item('bare_high', detector_confidence=0.9),
        'classed': _machine('classed', detector_confidence=0.99),
        'excluded': _item('excluded', class_excluded=True, detector_confidence=0.95),
    }
    response = _client(docs, monkeypatch).get(f'{BASE}/review/to_label')
    assert sorted(_ids(response)) == ['bare_high', 'bare_low']
    assert response.json()['sort_applied'] == 'detector_confidence:desc'
    # The in-memory fake only orders ascending: pin the descending clause itself.
    clause, applied, _fallback = asyncio.run(review_sorts.build_sort(None, tab='to_label'))
    assert applied == 'detector_confidence:desc'
    assert clause[0]['detector_confidence']['order'] == 'desc'


def test_check_machine_is_unvalidated_machine_classes_lowest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = {
        'sure': _machine('sure', confidence=0.9),
        'unsure': _machine('unsure', confidence=0.2),
        'middling': _machine('middling', confidence=0.5),
        'validated': _machine('validated', confidence=0.1, class_validated=True),
        'human': _machine('human', confidence=0.1, class_source='human'),
        'unclassed': _item('unclassed'),
    }
    response = _client(docs, monkeypatch).get(f'{BASE}/review/check_machine')
    assert _ids(response) == ['unsure', 'middling', 'sure']
    assert response.json()['sort_applied'] == 'confidence:asc'


def test_disagreements_unions_the_three_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {
        'unmatched': _item('unmatched', class_source='vlm_unmatched', class_validated=False),
        'detector': _machine('detector', detector_class_name='truck'),
        'model': _machine(
            'model', class_source='human', class_validated=True, probe_pred_class='truck'
        ),
        'agrees': _machine('agrees', detector_class_name='car'),
        'validated_agrees': _machine(
            'validated_agrees', class_validated=True, probe_pred_class='car'
        ),
        'validated_no_probe': _machine('validated_no_probe', class_validated=True),
    }
    response = _client(docs, monkeypatch).get(f'{BASE}/review/disagreements')
    assert sorted(_ids(response)) == ['detector', 'model', 'unmatched']
    reasons = {i['crop_id']: i['reason'] for i in response.json()['items']}
    assert reasons['model'] == 'new model disagrees with the validated label'
    assert reasons['detector'] == "VLM's class differs from the detector's class"
    assert reasons['unmatched'] == 'VLM gave no class answer'


def test_audit_queue_only_while_a_sample_is_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    pending = {
        'p1': _machine('p1', audit_sample=True),
        'done': _machine('done', audit_sample=True, class_validated=True),
    }
    client = _client(pending, monkeypatch)
    audit = {t['id']: t for t in _tabs(client)}['audit']
    assert audit['count'] == 1
    assert _ids(client.get(f'{BASE}/review/audit')) == ['p1']

    drawn_and_labelled = {'done': pending['done'], 'other': _machine('other')}
    assert 'audit' not in {t['id'] for t in _tabs(_client(drawn_and_labelled, monkeypatch))}


def test_unfillable_queue_is_absent_and_named_in_next_steps() -> None:
    drawn = WorkflowFacts(images=10, active_classes=2, audit_pending=True)
    idle = dataclasses.replace(drawn, audit_pending=False)
    assert 'audit' not in [s.id for s in build_steps(drawn).steps]
    step = next(s for s in build_steps(idle).steps if s.id == 'audit')
    assert step.done is False
    assert step.section == 'audit'
    assert 'audit sample' in step.detail.lower()


def test_old_tab_id_still_filters_crops(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {
        'far': _machine('far', cluster_score=0.2),
        'near': _machine('near', cluster_score=0.95),
    }
    response = _client(docs, monkeypatch).get(f'{BASE}/crops', params={'review_tab': 'outliers'})
    assert _ids(response, 'crops') == ['far']


@pytest.mark.parametrize('tab', ['outliers', 'all', 'mismatches', 'imported', 'not_a_tab'])
def test_old_tab_path_is_404_not_a_queue(tab: str, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client({'a': _machine('a')}, monkeypatch)
    for suffix in ('', '/locate?crop_id=a'):
        response = client.get(f'{BASE}/review/{tab}{suffix}')
        assert response.status_code == 404, response.text
        assert response.json()['detail']['error'] == 'not_a_queue'


def test_counts_follow_the_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {
        'car1': _item('car1', detector_class_name='car'),
        'car2': _item('car2', detector_class_name='car'),
        'bus': _item('bus', detector_class_name='bus'),
        'classed': _machine('classed'),
    }
    client = _client(docs, monkeypatch)
    assert {t['id']: t['count'] for t in _tabs(client)}['to_label'] == 3
    narrowed = {t['id']: t['count'] for t in _tabs(client, class_name='car')}
    assert narrowed['to_label'] == 0
    assert narrowed['check_machine'] == 1


def test_import_id_narrows_a_queue_and_the_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import_id`` is a shared item-filter parameter: every list route takes it, the queues too."""
    docs = {
        'a': _machine('a', import_ids=['imp1']),
        'b': _machine('b', import_ids=['imp2']),
        'c': _machine('c'),
    }
    client = _client(docs, monkeypatch)
    assert sorted(_ids(client.get(f'{BASE}/review/check_machine'))) == ['a', 'b', 'c']
    assert _ids(client.get(f'{BASE}/review/check_machine', params={'import_id': 'imp1'})) == ['a']
    counts = {t['id']: t['count'] for t in _tabs(client, import_id='imp1')}
    assert counts['check_machine'] == 1


def _options(client: TestClient, queue: str, **params: Any) -> list[str]:
    response = client.get(f'{BASE}/review/{queue}', params=params)
    assert response.status_code == 200, response.text
    return [o['id'] for o in response.json()['sort_options']]


def test_a_queue_serves_the_sorts_it_can_be_read_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """The served options are the selectable registry sorts (outlier-first included), never the
    sentinel or a sort that only belongs to another queue, and the queue's own default is one."""

    async def covered(
        _opensearch: Any, fields: frozenset[str], *, live: bool = False
    ) -> dict[str, int | None]:
        return dict.fromkeys(fields, 5)

    monkeypatch.setattr('src.services.curation.field_coverage.field_coverage', covered)
    client = _client({'a': _machine('a')}, monkeypatch)
    to_label = _options(client, 'to_label')
    assert 'default' not in to_label
    assert {'recent', 'atypicality', 'representativeness'} <= set(to_label)
    assert review_sorts.default_sort_for_tab('to_label') in to_label
    assert 'region_score' not in to_label
    assert review_sorts.AUDIT_SAMPLE_ORDER not in to_label
    assert review_sorts.AUDIT_SAMPLE_ORDER in _options(client, 'audit')
    assert 'disagreement_entropy_asc' in _options(client, 'disagreements')
    shown = client.get(f'{BASE}/review/to_label').json()['sort_options']
    assert all(set(o) == {'id', 'label'} and o['label'] for o in shown)


def test_a_sort_whose_field_no_item_carries_is_not_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def coverage(
        _opensearch: Any, fields: frozenset[str], *, live: bool = False
    ) -> dict[str, int | None]:
        return {f: (0 if f == 'cluster_score' else 5) for f in fields}

    monkeypatch.setattr('src.services.curation.field_coverage.field_coverage', coverage)
    client = _client({'a': _machine('a')}, monkeypatch)
    offered = _options(client, 'to_label')
    assert 'atypicality' not in offered
    assert 'representativeness' not in offered
    assert 'recent' in offered


def test_default_tab_is_the_first_queue_with_work(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {'m': _machine('m'), 'v': _machine('v', class_validated=True)}
    response = _client(docs, monkeypatch).get(f'{BASE}/review/tabs')
    assert response.status_code == 200, response.text
    body = response.json()
    # nothing is unclassed, so To label is empty and the machine labels are where the work is
    assert [t['count'] for t in body['tabs'][:2]] == [0, 1]
    assert body['default_tab'] == 'check_machine'


def test_default_tab_is_null_when_every_queue_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {'v': _machine('v', class_validated=True)}
    body = _client(docs, monkeypatch).get(f'{BASE}/review/tabs').json()
    assert all(t['count'] == 0 for t in body['tabs'])
    assert body['default_tab'] is None
