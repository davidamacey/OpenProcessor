"""Accept machine labels: the per-item merge and its lock rule, the undo, the dry-run counts and
the route. The merge is written once and re-checks every item on the freshest source."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.domain.history import restore_class_state
from src.domain.ingest_class_sources import HUMAN_BULK_ACCEPT_LABEL_SOURCE
from src.domain.label_ops import UndoOutcome, accept_fingerprint, current_query, undo_outcome
from src.services.curation import accept_machine as accept_machine_module
from src.services.curation.accept_machine import plan_accept, start_accept
from src.services.curation.label_ops.accept_merge import ACCEPT_WRITER, accept_merger


OP = '22222222-2222-4222-8222-222222222222'
CLASSES = {'car': (4, 'car'), 'person': (7, 'person')}
BOTH = ('detector', 'vlm')


def _item(**kw: Any) -> dict[str, Any]:
    return {
        'crop_id': 'c1',
        'class_id': 4,
        'class_name': 'car',
        'class_source': 'detector',
        'label_source': 'detector',
        'class_validated': False,
        'confidence': 0.97,
        **kw,
    }


def _merge(item: dict[str, Any], *, min_confidence: float = 0.95, sources: Any = BOTH) -> Any:
    return accept_merger(classes=CLASSES, min_confidence=min_confidence, sources=sources, op_id=OP)(
        item
    )


def test_the_merge_validates_the_machine_class_as_a_bulk_acceptance() -> None:
    merged = _merge(_item())
    assert merged['class_name'] == 'car'
    assert merged['class_id'] == 4  # by name, from the registry
    assert merged['class_validated'] is True
    assert merged['class_source'] == 'human'
    assert merged['label_source'] == HUMAN_BULK_ACCEPT_LABEL_SOURCE
    assert merged['label_op_id'] == OP
    snapshot = merged['class_id_history'][-1]
    assert (snapshot['op_id'], snapshot['writer']) == (OP, ACCEPT_WRITER)
    assert snapshot['class_source'] == 'detector'  # the state an undo restores


def test_the_vlm_label_is_judged_by_its_category_not_the_detector_score() -> None:
    vlm = _item(class_source='vlm', vlm_confidence='high', confidence=0.2)
    assert _merge(vlm, min_confidence=0.9)['class_validated'] is True  # high is 0.92
    assert _merge(vlm, min_confidence=0.95) == {}
    assert _merge(_item(class_source='vlm', vlm_confidence=None), min_confidence=0.5) == {}


def test_the_detector_label_is_judged_by_its_score() -> None:
    assert _merge(_item(confidence=0.949)) == {}
    assert _merge(_item(confidence=0.95)) != {}


def test_a_source_the_person_did_not_choose_is_left_alone() -> None:
    assert _merge(_item(), sources=('vlm',)) == {}
    vlm = _item(class_source='vlm', vlm_confidence='high')
    assert _merge(vlm, min_confidence=0.9, sources=('detector',)) == {}
    # a machine label no source names (a propagated or cluster-voted class) is never accepted
    assert _merge(_item(class_source='assist_propagated')) == {}


@pytest.mark.parametrize(
    'locked',
    [
        {'class_source': 'human', 'class_validated': True},
        {'class_source': 'human_move'},
        {'class_source': 'external_label', 'class_validated': True},
        {'class_validated': True},
        {'test_holdout': True},
        {'class_excluded': True},
        {'review_dismissed_at': '2026-11-12T00:00:00Z'},
        {'audit_sample': True},
        {'class_id': None, 'class_name': None},
        {'class_name': 'bus'},  # deprecated, region or unknown: not in the active classes
    ],
    ids=[
        'human',
        'human_move',
        'import_validated',
        'validated',
        'holdout',
        'excluded',
        'dismissed',
        'audit_sample',
        'no_class',
        'inactive_class',
    ],
)
def test_the_merge_never_writes_a_protected_item(locked: dict[str, Any]) -> None:
    assert _merge(_item(**locked)) == {}


def test_an_undo_restores_the_machine_class_an_accept_replaced() -> None:
    fp = accept_fingerprint(op_id=OP, writer=ACCEPT_WRITER, at='2026-11-12T00:00:00Z')
    before = _item()
    accepted = {**before, **_merge(before)}
    prior = accepted['class_id_history'][-1]
    outcome = undo_outcome(accepted, fp, prior=prior, registry_name=lambda _i: 'car')
    assert outcome is UndoOutcome.RESTORE
    restored = restore_class_state(prior)
    assert (restored['class_source'], restored['class_validated']) == ('detector', False)
    assert (restored['class_id'], restored['class_name']) == (4, 'car')
    # a person relabelled it since: the operation no longer owns the item
    relabelled = {**accepted, 'label_op_id': 'other', 'class_name': 'person'}
    assert (
        undo_outcome(relabelled, fp, prior=prior, registry_name=lambda _i: 'car')
        is UndoOutcome.KEPT_CHANGED
    )


def test_the_current_count_looks_for_human_validated_items_of_the_operation() -> None:
    fp = accept_fingerprint(op_id=OP, writer=ACCEPT_WRITER, at='2026-11-12T00:00:00Z')
    filters = current_query(fp)['bool']['filter']
    assert {'term': {'label_op_id': OP}} in filters
    assert {'term': {'class_source': 'human'}} in filters
    assert {'term': {'class_validated': True}} in filters
    assert not [f for f in filters if 'class_name' in str(f)]  # a different class per item


# --- the count and the queued operation, over a fake that evaluates the queries ----------------


def _population() -> dict[str, dict[str, Any]]:
    docs = [
        _item(crop_id='car-a'),
        _item(crop_id='car-b', confidence=0.99),
        _item(crop_id='car-low', confidence=0.6),
        _item(crop_id='person-a', class_id=7, class_name='person'),
        _item(crop_id='vlm-high', class_source='vlm', vlm_confidence='high', confidence=0.1),
        # protected, each one for a different reason
        _item(crop_id='excluded', class_excluded=True),
        _item(crop_id='dismissed', review_dismissed_at='2026-11-01T00:00:00Z'),
        _item(crop_id='holdout', test_holdout=True),
        _item(crop_id='audit', audit_sample=True),
        _item(crop_id='validated', class_validated=True),
        _item(crop_id='bus', class_id=2, class_name='bus'),  # deprecated
        # not machine labels at all
        _item(crop_id='human', class_source='human', class_validated=True),
        _item(crop_id='imported', class_source='external_label', class_validated=True),
        _item(
            crop_id='unclassed', class_id=None, class_name=None, class_source='detector_proposal'
        ),
    ]
    return {d['crop_id']: d for d in docs}


@dataclass
class _Entry:
    class_id: int
    class_name: str
    deprecated: bool = False
    group: str | None = None


REGISTRY = [_Entry(4, 'car'), _Entry(7, 'person'), _Entry(2, 'bus', deprecated=True)]


@pytest.fixture
def fake() -> QueryFakeOpenSearch:
    from src.config.curation_accessors import items_index

    return QueryFakeOpenSearch({items_index(): _population()})


@pytest.mark.asyncio
async def test_the_dry_run_counts_by_class_and_says_what_is_held_back(
    fake: QueryFakeOpenSearch,
) -> None:
    plan = await plan_accept(
        fake, cast('Any', REGISTRY), min_confidence=0.95, sources=BOTH, item_filter=None
    )
    # car-a and car-b (0.97, 0.99), person-a, vlm-high (high is 0.92: below the bar) -> 3
    assert dict(plan.by_class) == {'car': 2, 'person': 1}
    assert plan.items == 3
    held = plan.held_back
    assert (held.excluded, held.holdout, held.audit_sample) == (2, 1, 1)
    assert (held.locked, held.deprecated_class) == (1, 1)
    low = await plan_accept(
        fake, cast('Any', REGISTRY), min_confidence=0.9, sources=BOTH, item_filter=None
    )
    assert dict(low.by_class) == {'car': 3, 'person': 1}  # the VLM's high category now clears it
    only_vlm = await plan_accept(
        fake, cast('Any', REGISTRY), min_confidence=0.9, sources=('vlm',), item_filter=None
    )
    assert dict(only_vlm.by_class) == {'car': 1}


@pytest.mark.asyncio
async def test_a_dry_run_writes_nothing(fake: QueryFakeOpenSearch) -> None:
    before = fake.write_calls
    await plan_accept(
        fake, cast('Any', REGISTRY), min_confidence=0.95, sources=BOTH, item_filter=None
    )
    assert fake.write_calls == before


@pytest.mark.asyncio
async def test_the_operation_gets_exactly_the_counted_ids_and_its_recipe(
    fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import accept_machine

    started = AsyncMock(return_value=(uuid.uuid4(), SimpleNamespace(id=uuid.uuid4())))
    monkeypatch.setattr(accept_machine.label_op_record, 'start', started)
    result = await start_accept(
        cast('Any', None),
        uuid.uuid4(),
        fake,
        cast('Any', REGISTRY),
        min_confidence=0.95,
        sources=BOTH,
        item_filter=None,
    )
    assert started.await_args is not None
    kwargs = started.await_args.kwargs
    assert sorted(kwargs['ids']) == ['car-a', 'car-b', 'person-a']
    assert kwargs['op'].kind == 'accept_machine'
    assert kwargs['spec']['classes'] == {'car': [4, 'car'], 'person': [7, 'person']}
    assert (kwargs['spec']['min_confidence'], kwargs['spec']['sources']) == (
        0.95,
        ['detector', 'vlm'],
    )
    assert result.remaining == 0


@pytest.mark.asyncio
async def test_one_operation_writes_at_most_the_cap_and_reports_the_rest(
    fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import accept_machine

    monkeypatch.setattr(accept_machine, 'MAX_BULK_SELECTION', 2)
    started = AsyncMock(return_value=(uuid.uuid4(), SimpleNamespace(id=uuid.uuid4())))
    monkeypatch.setattr(accept_machine.label_op_record, 'start', started)
    result = await start_accept(
        cast('Any', None),
        uuid.uuid4(),
        fake,
        cast('Any', REGISTRY),
        min_confidence=0.95,
        sources=BOTH,
        item_filter=None,
    )
    assert started.await_args is not None
    assert len(started.await_args.kwargs['ids']) == 2
    assert result.remaining == 1


@pytest.mark.asyncio
async def test_nothing_qualifying_queues_no_operation(
    fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import accept_machine

    started = AsyncMock()
    monkeypatch.setattr(accept_machine.label_op_record, 'start', started)
    result = await start_accept(
        cast('Any', None),
        uuid.uuid4(),
        fake,
        [],
        min_confidence=0.95,
        sources=BOTH,
        item_filter=None,
    )
    assert (result.op_id, result.job_id, result.remaining) == (None, None, 0)
    started.assert_not_awaited()


def test_the_job_builds_the_same_merge_from_the_stored_recipe() -> None:
    from src.services.curation.label_ops.record import merger_for

    spec = {
        'kind': 'accept_machine',
        'classes': {'car': [4, 'car']},
        'min_confidence': 0.95,
        'sources': ['detector'],
    }
    merge = merger_for(spec, OP)
    assert merge(_item())['label_source'] == HUMAN_BULK_ACCEPT_LABEL_SOURCE
    assert merge(_item(test_holdout=True)) == {}


# --- the route ---------------------------------------------------------------------------------

URL = '/curation/projects/default/label_ops/accept_machine'


def _client(monkeypatch: pytest.MonkeyPatch, fake: QueryFakeOpenSearch) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.core.db_session import get_db
    from src.routers.curation import (
        _common,
        _raw_opensearch_dep,
        label_ops as router_module,
        router as curation_router,
    )

    started = AsyncMock(return_value=(uuid.uuid4(), SimpleNamespace(id=uuid.uuid4())))
    monkeypatch.setattr(accept_machine_module.label_op_record, 'start', started)
    monkeypatch.setattr(router_module, 'current_project_id', lambda: uuid.uuid4())

    class _Snapshot:
        async def snapshot(self) -> Any:
            return SimpleNamespace(classes=REGISTRY)

    async def no_database() -> Any:
        yield None

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    app.dependency_overrides[_common._registry_dep] = lambda: _Snapshot()
    app.dependency_overrides[get_db] = no_database
    return TestClient(app)


def test_the_route_dry_run_is_the_default_and_serves_the_counts(
    monkeypatch: pytest.MonkeyPatch, fake: QueryFakeOpenSearch
) -> None:
    resp = _client(monkeypatch, fake).post(URL, json={'min_confidence': 0.95})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body['dry_run'], body['items'], body['op_id']) == (True, 3, None)
    assert body['by_class'] == [
        {'class_name': 'car', 'count': 2},
        {'class_name': 'person', 'count': 1},
    ]
    assert body['held_back'] == {
        'excluded': 2,
        'holdout': 1,
        'audit_sample': 1,
        'locked': 1,
        'deprecated_class': 1,
    }
    assert accept_machine_module.label_op_record.start.await_count == 0  # type: ignore[attr-defined]


def test_the_route_real_run_queues_the_operation(
    monkeypatch: pytest.MonkeyPatch, fake: QueryFakeOpenSearch
) -> None:
    resp = _client(monkeypatch, fake).post(URL, json={'min_confidence': 0.95, 'dry_run': False})
    assert resp.status_code == 202, resp.text
    assert resp.json()['op_id'] is not None
    assert resp.json()['job_id'] is not None


@pytest.mark.parametrize(
    'body',
    [
        {},  # the bar is required
        {'min_confidence': 0.49},
        {'min_confidence': 1.01},
        {'min_confidence': 0.9, 'sources': []},
        {'min_confidence': 0.9, 'sources': ['human']},
        {'min_confidence': 0.9, 'surprise': 1},
    ],
)
def test_the_route_refuses_a_malformed_bar(
    monkeypatch: pytest.MonkeyPatch, fake: QueryFakeOpenSearch, body: dict[str, Any]
) -> None:
    assert _client(monkeypatch, fake).post(URL, json=body).status_code == 422
