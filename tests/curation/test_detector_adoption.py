"""Adopt detector classes: the plan, the per-item merge and the undo of the recorded operation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.domain.ingest_class_sources import DETECTOR_CLASS_SOURCE
from src.domain.label_ops import UndoOutcome, current_query, detector_fingerprint, undo_outcome
from src.services.curation.detector_adoption import (
    AdoptionStart,
    DetectorCensus,
    adoptable_query,
    plan_adoption,
)
from src.services.curation.label_ops.adopt_merge import ADOPT_WRITER, adoption_merger


if TYPE_CHECKING:
    import pytest

OP = '11111111-1111-4111-8111-111111111111'
CLASSES = {'car': (4, 'car'), 'traffic_light': (9, 'traffic_light')}


@dataclass
class _Entry:
    class_id: int
    class_name: str
    deprecated: bool = False
    group: str | None = None


class _Registry:
    def __init__(self, *entries: _Entry) -> None:
        self.classes = list(entries)


def _item(**kw: Any) -> dict[str, Any]:
    return {
        'crop_id': 'c1',
        'detector_class_name': 'car',
        'class_source': 'item_proposal',
        'class_validated': False,
        **kw,
    }


def test_the_plan_creates_the_unknown_names_and_skips_deprecated_and_region_classes() -> None:
    registry = _Registry(
        _Entry(1, 'car'),
        _Entry(2, 'bus', deprecated=True),
        _Entry(3, 'wheel', group='region'),
    )
    adoptable = {'car': 5, 'bus': 3, 'wheel': 2, 'zebra': 4}
    # 'okapi' is on an item a person already classed: it is not adoptable, but the registry
    # still lacks it, and every label the detector produced should be in the registry.
    seen = {**adoptable, 'okapi': 1}
    plan = plan_adoption(DetectorCensus(seen, adoptable), cast('Any', registry.classes))
    assert plan.create == ('okapi', 'zebra')
    assert plan.existing == {'car': (1, 'car')}
    # A deprecated class is never brought back and the region class never labels a whole item.
    assert set(plan.blocked) == {'bus', 'wheel'}
    assert plan.items == 9
    assert plan.blocked_items == 5


def test_the_merge_writes_the_machine_class_and_tags_the_operation() -> None:
    merged = adoption_merger(CLASSES, OP)(_item())
    assert merged['class_id'] == 4
    assert merged['class_name'] == 'car'
    assert merged['class_source'] == DETECTOR_CLASS_SOURCE
    assert merged['class_validated'] is False  # a guess, never a human label
    assert merged['label_op_id'] == OP
    snapshot = merged['class_id_history'][-1]
    assert snapshot['op_id'] == OP
    assert snapshot['writer'] == ADOPT_WRITER
    assert snapshot['class_source'] == 'item_proposal'  # the state an undo restores


def test_the_merge_leaves_a_classed_locked_or_unknown_item_alone() -> None:
    merge = adoption_merger(CLASSES, OP)
    assert merge(_item(class_id=7, class_name='truck')) == {}
    assert merge(_item(class_source='human', class_validated=True)) == {}
    assert merge(_item(class_source='external_label', class_validated=True)) == {}
    assert merge(_item(test_holdout=True)) == {}
    assert merge(_item(detector_class_name='zebra')) == {}
    assert merge(_item(detector_class_name=None)) == {}


def test_the_query_leaves_out_classed_validated_excluded_and_holdout_items() -> None:
    query = adoptable_query()['bool']
    assert {'exists': {'field': 'detector_class_name'}} in query['filter']
    assert {'exists': {'field': 'class_id'}} in query['must_not']
    for flag in ('class_validated', 'class_excluded', 'test_holdout'):
        assert {'term': {flag: True}} in query['must_not']


def test_an_undo_restores_an_item_that_still_carries_the_adopted_class() -> None:
    fp = detector_fingerprint(op_id=OP, writer=ADOPT_WRITER, at='2026-11-12T00:00:00Z')
    before = _item()
    adopted = {**before, **adoption_merger(CLASSES, OP)(before)}
    prior = adopted['class_id_history'][-1]
    assert (
        undo_outcome(adopted, fp, prior=prior, registry_name=lambda _i: None) is UndoOutcome.RESTORE
    )
    # A person labeled it since: the operation no longer owns the item.
    relabeled = {**adopted, 'class_source': 'human', 'class_validated': True, 'label_op_id': 'x'}
    assert (
        undo_outcome(relabeled, fp, prior=prior, registry_name=lambda _i: None)
        is UndoOutcome.KEPT_CHANGED
    )


def test_the_current_count_looks_for_the_detector_source_unvalidated() -> None:
    fp = detector_fingerprint(op_id=OP, writer=ADOPT_WRITER, at='2026-11-12T00:00:00Z')
    filters = current_query(fp)['bool']['filter']
    assert {'term': {'label_op_id': OP}} in filters
    assert {'term': {'class_source': DETECTOR_CLASS_SOURCE}} in filters
    assert {'term': {'class_validated': False}} in filters


URL = '/curation/projects/default/label_ops/adopt_detector'


def _client(monkeypatch: pytest.MonkeyPatch, started: AdoptionStart | None = None) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.core.db_session import get_db
    from src.routers.curation import (
        _common,
        _raw_opensearch_dep,
        label_ops as router_module,
        router as curation_router,
    )

    registry = _Registry(_Entry(1, 'car'))
    census = DetectorCensus({'car': 2, 'zebra': 1}, {'car': 2, 'zebra': 1})
    monkeypatch.setattr(router_module, 'detector_census', AsyncMock(return_value=census))
    monkeypatch.setattr(router_module, 'start_adoption', AsyncMock(return_value=started))
    monkeypatch.setattr(router_module, 'current_project_id', lambda: uuid.uuid4())

    class _Snapshot:
        async def snapshot(self) -> Any:
            return registry

    async def no_database() -> Any:
        yield None

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: object()
    app.dependency_overrides[_common._registry_dep] = lambda: _Snapshot()
    app.dependency_overrides[get_db] = no_database
    return TestClient(app)


def test_a_dry_run_reports_the_plan_and_starts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.routers.curation import label_ops as router_module

    resp = _client(monkeypatch).post(URL, json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['dry_run'] is True
    assert (body['items'], body['classes_to_create'], body['op_id']) == (3, ['zebra'], None)
    router_module.start_adoption.assert_not_awaited()  # type: ignore[attr-defined]


def test_a_real_run_queues_the_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    op_id, job_id = uuid.uuid4(), uuid.uuid4()
    plan = plan_adoption(
        DetectorCensus({'car': 2, 'zebra': 1}, {'car': 2, 'zebra': 1}),
        cast('Any', _Registry(_Entry(1, 'car')).classes),
    )
    started = AdoptionStart(plan, ('zebra',), op_id, job_id, 0)
    resp = _client(monkeypatch, started).post(URL, json={'dry_run': False})
    assert resp.status_code == 202, resp.text
    assert resp.json()['op_id'] == str(op_id)
    assert resp.json()['job_id'] == str(job_id)
    assert resp.json()['classes_to_create'] == ['zebra']


def test_a_real_run_with_nothing_to_label_queues_no_job(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = plan_adoption(DetectorCensus({}, {}), cast('Any', _Registry().classes))
    resp = _client(monkeypatch, AdoptionStart(plan, (), None, None, 0)).post(
        URL, json={'dry_run': False}
    )
    assert resp.status_code == 200, resp.text
    assert (resp.json()['op_id'], resp.json()['job_id']) == (None, None)
