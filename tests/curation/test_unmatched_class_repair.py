"""Operator repair: clear a stale class_id/class_name a legacy
``vlm_unmatched`` write left in place (IT-2)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.services.curation import unmatched_class_repair as repair


INDEX = 'test_items'


def _doc(crop_id: str, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_source': 'vlm_unmatched',
        'class_id': 5,
        'class_name': 'foo',
        'class_detector': 'coco_yolo11',
        'class_detector_version': '1',
        'class_labeler': 'coco_yolo11',
        'class_labeled_at': '2026-01-01T00:00:00Z',
        'cluster_id': 5,
        'class_validated': False,
        **kw,
    }


def _docs() -> dict[str, dict[str, Any]]:
    return {
        'a_class_range': _doc('a_class_range', cluster_id=5),
        'b_candidate_cluster': _doc('b_candidate_cluster', cluster_id=10042),
        # class_write_locked() is validated-OR-human-owned; a doc can only
        # be BOTH class_source='vlm_unmatched' (the query filter) AND
        # locked via the validated flag -- class_source being itself a
        # human marker would already fail the query filter.
        'c_validated': _doc('c_validated', class_validated=True),
        # Not a candidate at all -- no class_id left to clear.
        'd_already_clear': _doc('d_already_clear', class_id=None, class_name=None),
    }


@pytest.mark.asyncio
async def test_plans_pick_exactly_the_unlocked_rows() -> None:
    fake = QueryFakeOpenSearch({INDEX: _docs()})
    plans = {p.crop_id: p for p in await repair.plan_repairs(fake, index=INDEX, page_size=2)}

    assert set(plans) == {'a_class_range', 'b_candidate_cluster', 'c_validated'}
    assert plans['a_class_range'].applicable is True
    assert plans['a_class_range'].restore['cluster_id'] == -1
    assert plans['b_candidate_cluster'].applicable is True
    assert 'cluster_id' not in plans['b_candidate_cluster'].restore
    assert plans['c_validated'].applicable is False


@pytest.mark.asyncio
async def test_apply_clears_class_and_resets_class_range_cluster() -> None:
    fake = QueryFakeOpenSearch({INDEX: _docs()})
    plans = await repair.plan_repairs(fake, index=INDEX)
    counts = await repair.apply_repairs(fake, plans, index=INDEX)
    assert counts == {'repaired': 2, 'skipped_changed': 0, 'not_applicable': 1, 'errors': 0}

    docs = fake.docs(INDEX)
    a = docs['a_class_range']
    assert a['class_id'] is None
    assert a['class_name'] is None
    assert a['class_detector'] is None
    assert a['cluster_id'] == -1
    assert a['cluster_subid'] is None
    snap = a['class_id_history'][-1]
    assert (snap['class_source'], snap['restorable'], snap['writer']) == (
        'vlm_unmatched',
        True,
        repair.REPAIR_WRITER,
    )

    b = docs['b_candidate_cluster']
    assert b['class_id'] is None
    assert b['cluster_id'] == 10042  # candidate cluster untouched

    # A locked (validated) row is never written.
    assert 'class_id_history' not in docs['c_validated']
    assert docs['c_validated']['class_id'] == 5


@pytest.mark.asyncio
async def test_apply_skips_items_changed_since_the_plan() -> None:
    fake = QueryFakeOpenSearch({INDEX: _docs()})
    plans = [p for p in await repair.plan_repairs(fake, index=INDEX) if p.applicable]
    fake.docs(INDEX)['a_class_range'].update(class_validated=True)
    counts = await repair.apply_repairs(fake, plans, index=INDEX)
    assert counts['skipped_changed'] == 1
    assert counts['repaired'] == 1
    assert fake.docs(INDEX)['a_class_range']['class_validated'] is True
    assert fake.docs(INDEX)['a_class_range']['class_id'] == 5


@pytest.mark.asyncio
async def test_script_dry_run_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.curation.repair_unmatched_class import build_parser, run

    fake = QueryFakeOpenSearch({INDEX: _docs()})
    before = copy.deepcopy(fake.docs(INDEX))
    args = build_parser().parse_args(['--index', INDEX, '--verbose'])
    assert await run(args, fake) == 0
    assert fake.docs(INDEX) == before
    out = capsys.readouterr().out
    assert '3 candidate(s); 2 repairable (1 in a class-range cluster), 1 skipped (locked)' in out
    assert 'Dry-run only' in out


@pytest.mark.asyncio
async def test_script_apply_writes(capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.curation.repair_unmatched_class import build_parser, run

    fake = QueryFakeOpenSearch({INDEX: _docs()})
    args = build_parser().parse_args(['--index', INDEX, '--apply'])
    assert await run(args, fake) == 0
    assert fake.docs(INDEX)['a_class_range']['class_id'] is None
