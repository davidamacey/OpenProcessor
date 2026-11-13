"""``GET /quality/readiness``: each check on seeded facts, the coverage count over a fake that
evaluates the queries, the links, and that Train's preflight and the checklist share their rows."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import parse_qs

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import items_index
from src.schemas.curation.quality import ClassCoverage
from src.services.curation import dataset_readiness as dr
from src.services.curation.dataset_readiness import ReadinessFacts, build_readiness
from src.services.curation.dataset_thresholds import (
    HARD_MIN_CROPS_PER_CLASS,
    MIN_TEST_CROPS_PER_CLASS,
    WARN_MIN_CROPS_PER_CLASS,
)
from src.services.curation.review_tabs import QUEUES


GOOD_AUDIT = {
    'audited': 300,
    'pending': 0,
    'detector': [
        {
            'name': 'car',
            'n': 200,
            'correct': 190,
            'precision': 0.95,
            'ci_low': 0.91,
            'ci_high': 0.97,
            'insufficient_sample': False,
        }
    ],
}


def _row(
    name: str, validated: int, test: int, machine: int = 0, detector: int = 0
) -> ClassCoverage:
    return ClassCoverage(
        class_name=name,
        validated=validated,
        machine_only=machine,
        detector_only=detector,
        test=test,
    )


def _facts(**over: Any) -> ReadinessFacts:
    coverage = [
        _row('car', WARN_MIN_CROPS_PER_CLASS, MIN_TEST_CROPS_PER_CLASS),
        _row('person', WARN_MIN_CROPS_PER_CLASS + 7, MIN_TEST_CROPS_PER_CLASS + 3),
    ]
    base: dict[str, Any] = {
        'coverage': coverage,
        'class_ids': {'car': 1, 'person': 2},
        'holdout_items': 20,
        'audit': GOOD_AUDIT,
        'near_dup_stale': [],
        'unlabeled_on_exportable': 0,
        'export': ('ok', 'export was built from the current items index'),
    }
    return ReadinessFacts(**{**base, **over})


def _check(readiness: Any, check_id: str) -> Any:
    return next(c for c in readiness.checks if c.id == check_id)


def test_a_dataset_with_nothing_to_do_is_ready() -> None:
    readiness = build_readiness(_facts())
    assert readiness.verdict == 'ready'
    assert [c.severity for c in readiness.checks] == ['ok'] * 7
    assert [c.id for c in readiness.checks] == [
        'class_balance',
        'test_holdout',
        'machine_labels',
        'accuracy_audit',
        'near_duplicates',
        'unlabeled_objects',
        'export_current',
    ]
    assert all(c.link is None for c in readiness.checks)
    assert all(c.label for c in readiness.checks)


def test_a_class_under_the_hard_floor_blocks_and_is_named() -> None:
    coverage = [_row('car', HARD_MIN_CROPS_PER_CLASS - 1, 9), _row('bus', 2, 9)]
    readiness = build_readiness(_facts(coverage=coverage, class_ids={'car': 1, 'bus': 3}))
    check = _check(readiness, 'class_balance')
    assert (check.severity, check.label) == ('block', 'Class balance')
    assert {r.class_name for r in check.classes} == {'car', 'bus'}
    assert readiness.verdict == 'blocked'
    assert check.link is not None


def test_a_class_between_the_floors_only_warns() -> None:
    coverage = [_row('car', HARD_MIN_CROPS_PER_CLASS, 9)]
    readiness = build_readiness(_facts(coverage=coverage, class_ids={'car': 1}))
    assert _check(readiness, 'class_balance').severity == 'warn'
    assert readiness.verdict == 'warn'


def test_the_checklist_and_train_preflight_judge_a_class_the_same_way() -> None:
    from src.services.training.preflight_checks import class_balance_check

    counts = {1: HARD_MIN_CROPS_PER_CLASS - 1, 2: WARN_MIN_CROPS_PER_CLASS}
    names = dr._Names({1: 'car', 2: 'person'})
    preflight = class_balance_check([1, 2], counts, names)
    coverage = [_row('car', counts[1], 9), _row('person', counts[2], 9)]
    ours = _check(build_readiness(_facts(coverage=coverage)), 'class_balance')
    assert (ours.severity, ours.message) == (preflight.severity, preflight.message)


def test_an_unfrozen_test_set_blocks() -> None:
    check = _check(build_readiness(_facts(holdout_items=0)), 'test_holdout')
    assert (check.severity, check.count) == ('block', 0)
    assert check.link is not None
    assert check.link.section == 'export'


def test_a_class_with_no_test_examples_blocks_and_a_thin_one_warns() -> None:
    zero = _facts(coverage=[_row('car', 999, 0), _row('person', 999, 9)])
    assert _check(build_readiness(zero), 'test_holdout').severity == 'block'
    thin = _facts(coverage=[_row('car', 999, MIN_TEST_CROPS_PER_CLASS - 1), _row('person', 999, 9)])
    assert _check(build_readiness(thin), 'test_holdout').severity == 'warn'


def test_machine_labels_are_counted_per_class_worst_first() -> None:
    coverage = [
        _row('car', 999, 9, machine=10, detector=402),
        _row('person', 999, 9, detector=380),
        _row('bus', 999, 9),
    ]
    readiness = build_readiness(
        _facts(coverage=coverage, class_ids={'car': 1, 'person': 2, 'bus': 3})
    )
    check = _check(readiness, 'machine_labels')
    assert (check.severity, check.count) == ('warn', 792)
    assert [(r.class_name, r.count) for r in check.classes] == [('car', 412), ('person', 380)]
    assert check.message.startswith('792 machine labels are not checked (car 412, person 380)')
    assert 'Check machine labels' in check.message


@pytest.mark.parametrize(
    ('audit', 'severity', 'count'),
    [
        ({'audited': 0, 'pending': 0, 'detector': []}, 'warn', 0),
        ({**GOOD_AUDIT, 'audited': 112, 'pending': 188}, 'warn', 188),
        (GOOD_AUDIT, 'ok', 300),
    ],
    ids=['none_yet', 'pending', 'done'],
)
def test_the_audit_check_says_none_pending_or_done(
    audit: dict[str, Any], severity: str, count: int
) -> None:
    check = _check(build_readiness(_facts(audit=audit)), 'accuracy_audit')
    assert (check.severity, check.count) == (severity, count)


def test_the_audit_check_serves_each_classs_precision_with_its_interval() -> None:
    (row,) = _check(build_readiness(_facts()), 'accuracy_audit').classes
    assert (row.class_name, row.count, row.precision, row.ci_low, row.ci_high) == (
        'car',
        200,
        0.95,
        0.91,
        0.97,
    )


def test_stale_near_duplicate_groups_warn_and_current_ones_do_not() -> None:
    stale = _check(build_readiness(_facts(near_dup_stale=['image: pending'])), 'near_duplicates')
    assert (stale.severity, stale.count) == ('warn', 1)
    assert 'image: pending' in stale.message
    assert _check(build_readiness(_facts()), 'near_duplicates').severity == 'ok'


@pytest.mark.parametrize(('n', 'severity'), [(0, 'ok'), (31, 'warn'), (None, 'warn')])
def test_unlabelled_objects_on_exportable_photos(n: int | None, severity: str) -> None:
    check = _check(build_readiness(_facts(unlabeled_on_exportable=n)), 'unlabeled_objects')
    assert (check.severity, check.count) == (severity, n)


def test_the_export_check_says_none_stale_or_current() -> None:
    assert _check(build_readiness(_facts(export=None)), 'export_current').severity == 'warn'
    stale = _check(
        build_readiness(_facts(export=('block', 'built from a previous index'))), 'export_current'
    )
    assert (stale.severity, stale.message) == ('block', 'built from a previous index')
    unknown = _check(
        build_readiness(_facts(export=('unknown', 'could not read'))), 'export_current'
    )
    assert unknown.severity == 'warn'
    assert _check(build_readiness(_facts()), 'export_current').label == 'Export is current'


def test_every_link_is_a_page_and_a_query_the_workspace_reads() -> None:
    worst = _facts(
        coverage=[_row('car', 1, 0, machine=5, detector=5)],
        class_ids={'car': 1},
        holdout_items=0,
        audit={'audited': 0, 'pending': 0, 'detector': []},
        near_dup_stale=['image: no_run'],
        unlabeled_on_exportable=4,
        export=None,
    )
    links = [c.link for c in build_readiness(worst).checks if c.link is not None]
    assert len(links) == 7
    views = {'gallery', 'clusters', 'detections', 'review', 'table', 'map', 'embedding'}
    for link in links:
        assert link.section in {'workspace', 'export', 'quality/audit', 'classes'}
        query = parse_qs(link.query or '', strict_parsing=bool(link.query))
        if link.section == 'workspace':
            assert set(query) <= {'view', 'queue', 'near_dup'}
            assert set(query.get('view', ['gallery'])) <= views
            assert set(query.get('queue', ['to_label'])) <= set(QUEUES)


# --- the coverage count and the route ----------------------------------------------------------


@dataclass
class _Entry:
    class_id: int
    class_name: str
    deprecated: bool = False
    group: str | None = None


def _population() -> dict[str, dict[str, Any]]:
    def doc(crop_id: str, name: str, **kw: Any) -> dict[str, Any]:
        return {
            'crop_id': crop_id,
            'class_id': 1,
            'class_name': name,
            'class_validated': False,
            'class_source': 'vlm',
            **kw,
        }

    docs = [
        doc('a', 'car', class_validated=True, class_source='human'),
        doc('b', 'car', class_validated=True, class_source='human', test_holdout=True),
        doc('c', 'car', class_source='detector'),
        doc('d', 'car', class_source='vlm'),
        doc('e', 'person', class_source='detector'),
        doc('x', 'car', class_excluded=True),
        doc('y', 'bus'),  # a class the registry no longer serves
    ]
    return {d['crop_id']: d for d in docs}


def test_coverage_counts_each_active_class_and_leaves_out_excluded_items() -> None:
    fake = QueryFakeOpenSearch({items_index(): _population()})
    import asyncio

    classes = cast('Any', [_Entry(1, 'car'), _Entry(2, 'person'), _Entry(3, 'truck')])
    response = asyncio.run(fake.search(index=items_index(), body=dr.coverage_body()))
    rows, ids = dr.coverage_rows(response, classes)
    assert ids == {'car': 1, 'person': 2, 'truck': 3}
    by_name = {r.class_name: r for r in rows}
    assert (
        by_name['car'].validated,
        by_name['car'].detector_only,
        by_name['car'].machine_only,
        by_name['car'].test,
    ) == (2, 1, 1, 1)
    assert (by_name['person'].validated, by_name['person'].detector_only) == (0, 1)
    assert by_name['truck'] == _row('truck', 0, 0)
    assert 'bus' not in by_name


def test_the_route_serves_the_checklist(monkeypatch: pytest.MonkeyPatch) -> None:
    from _curation_app import mount_curation_routers

    from src.core.db_session import get_db
    from src.routers.curation import (
        _common,
        _raw_opensearch_dep,
        quality,
        router as curation_router,
    )

    async def facts(*_a: Any, **_kw: Any) -> ReadinessFacts:
        return _facts(near_dup_stale=['image: no_run'])

    async def near(*_a: Any, **_kw: Any) -> Any:
        level = SimpleNamespace(level='image', stale_reasons=['no_run'])
        return SimpleNamespace(levels=[level])

    class _Snapshot:
        async def snapshot(self) -> Any:
            return SimpleNamespace(classes=[])

    async def no_database() -> Any:
        yield None

    monkeypatch.setattr(quality, 'gather_facts', facts)
    monkeypatch.setattr(quality, 'near_dup_status', near)
    monkeypatch.setattr(quality, 'current_project_id', lambda: 'p')
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: object()
    app.dependency_overrides[_common._registry_dep] = lambda: _Snapshot()
    app.dependency_overrides[get_db] = no_database
    resp = TestClient(app).get('/curation/projects/default/quality/readiness')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['verdict'] == 'warn'
    assert body['checks'][0]['id'] == 'class_balance'
    assert body['coverage'][0] == {
        'class_name': 'car',
        'validated': 500,
        'machine_only': 0,
        'detector_only': 0,
        'test': 5,
    }
    assert body['computed_at'].endswith('+00:00')
