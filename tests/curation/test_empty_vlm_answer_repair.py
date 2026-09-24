"""Operator repair: undo ``vlm_unmatched`` stamped for an empty VLM class answer."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.services.curation import empty_vlm_answer_repair as repair


INDEX = 'test_items'
UPDATED = '2026-09-24T14:28:23+00:00'


@pytest.fixture(autouse=True)
def _profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        repair,
        'ingest_primary_profile',
        lambda: SimpleNamespace(name='det', detector_model='det_x', assigns_class=False),
    )
    monkeypatch.setattr(
        repair,
        'ingest_secondary_profile',
        lambda: SimpleNamespace(name='item', detector_model='item_x', assigns_class=True),
    )


def _doc(crop_id: str, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_source': 'vlm_unmatched',
        'label_source': 'vlm',
        'class_validated': False,
        'vlm_confidence': 'low',
        'vlm_raw_class': '',
        'vlm_raw_label': '',
        'class_labeler': 'ingest',
        'updated_at': UPDATED,
        **kw,
    }


def _docs() -> dict[str, dict[str, Any]]:
    return {
        'a_proposal': _doc('a_proposal', class_detector='det_x'),
        'b_classifier': _doc(
            'b_classifier',
            class_detector='item_x',
            class_id=3,
            class_name='widget',
            vlm_raw_class=None,
            vlm_raw_label='widget',
        ),
        'c_vlm_class': _doc(
            'c_vlm_class',
            class_detector='vlm',
            class_labeler='vlm',
            class_id=43,
            class_name='gadget',
            vlm_raw_label='gadget',
        ),
        'd_history': _doc(
            'd_history',
            class_detector='old_model',
            class_labeler='old_model',
            class_id_history=[
                {
                    'class_id': 5,
                    'class_name': 'sprocket',
                    'class_source': 'old_model',
                    'label_source': 'old_model',
                    'confidence': 0.5,
                    'writer': 'x',
                    'at': 't',
                }
            ],
        ),
        'e_unresolved': _doc('e_unresolved', class_detector='mystery', class_labeler='mystery'),
        # Not candidates:
        'f_real_unmatched': _doc(
            'f_real_unmatched', class_detector='det_x', vlm_raw_class='zeppelin'
        ),
        'g_validated': _doc('g_validated', class_detector='det_x', class_validated=True),
        'h_vlm': _doc('h_vlm', class_source='vlm', class_detector='det_x'),
    }


@pytest.mark.asyncio
async def test_plans_restore_the_pre_write_class_source() -> None:
    fake = QueryFakeOpenSearch({INDEX: _docs()})
    plans = {p.crop_id: p for p in await repair.plan_repairs(fake, index=INDEX, page_size=2)}

    assert set(plans) == {'a_proposal', 'b_classifier', 'c_vlm_class', 'd_history', 'e_unresolved'}
    assert (plans['a_proposal'].source, plans['a_proposal'].restore['class_source']) == (
        repair.SOURCE_INGEST,
        'det_proposal',
    )
    assert plans['a_proposal'].restore['label_source'] == 'det_proposal'
    assert plans['b_classifier'].restore['class_source'] == 'item_model'
    assert (plans['c_vlm_class'].source, plans['c_vlm_class'].restore['class_source']) == (
        repair.SOURCE_VLM,
        'vlm',
    )
    assert plans['d_history'].source == repair.SOURCE_HISTORY
    assert plans['d_history'].restore['class_id'] == 5
    assert plans['d_history'].restore['class_source'] == 'old_model'
    assert plans['e_unresolved'].applicable is False

    counts = repair.summarize(list(plans.values()))
    assert counts['by_source'][repair.SOURCE_INGEST] == 2
    assert counts['by_restored_class_source']['det_proposal'] == 1


def test_primary_with_a_class_is_not_guessed() -> None:
    # A proposer never writes a class_id; such a doc is inconsistent.
    assert repair.ingest_class_source({'class_detector': 'det_x', 'class_id': 2}) is None


@pytest.mark.asyncio
async def test_apply_restores_and_snapshots() -> None:
    fake = QueryFakeOpenSearch({INDEX: _docs()})
    plans = await repair.plan_repairs(fake, index=INDEX)
    counts = await repair.apply_repairs(fake, plans, index=INDEX)
    assert counts == {'repaired': 4, 'skipped_changed': 0, 'not_applicable': 1, 'errors': 0}

    docs = fake.docs(INDEX)
    a = docs['a_proposal']
    assert (a['class_source'], a['label_source']) == ('det_proposal', 'det_proposal')
    assert a.get('class_id') is None
    assert a.get('vlm_confidence') is None
    assert a.get('vlm_raw_class') is None
    assert a.get('vlm_raw_label') is None
    assert a['vlm_class_attempted_at'] == UPDATED
    assert a['vlm_class_empty_reason'] == 'no_answer'
    snap = a['class_id_history'][-1]
    assert (snap['class_source'], snap['restorable'], snap['writer']) == (
        'vlm_unmatched',
        True,
        repair.REPAIR_WRITER,
    )

    b = docs['b_classifier']
    assert (b['class_id'], b['class_source'], b['label_source']) == (3, 'item_model', 'item_model')
    # A non-empty raw label is kept.
    assert b['vlm_raw_label'] == 'widget'
    assert docs['c_vlm_class']['class_source'] == 'vlm'
    assert docs['d_history']['class_id'] == 5
    assert docs['e_unresolved']['class_source'] == 'vlm_unmatched'
    for untouched in ('f_real_unmatched', 'g_validated', 'h_vlm'):
        assert 'class_id_history' not in docs[untouched]


@pytest.mark.asyncio
async def test_apply_skips_items_changed_since_the_plan() -> None:
    fake = QueryFakeOpenSearch({INDEX: _docs()})
    plans = [p for p in await repair.plan_repairs(fake, index=INDEX) if p.applicable]
    fake.docs(INDEX)['a_proposal'].update(class_source='human', label_source='human')
    fake.docs(INDEX)['b_classifier'].update(vlm_raw_class='widget2')
    counts = await repair.apply_repairs(fake, plans, index=INDEX)
    assert counts['skipped_changed'] == 2
    assert counts['repaired'] == 2
    assert fake.docs(INDEX)['a_proposal']['class_source'] == 'human'


@pytest.mark.asyncio
async def test_script_dry_run_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    import copy

    from scripts.curation.repair_empty_vlm_answers import run

    fake = QueryFakeOpenSearch({INDEX: _docs()})
    before = copy.deepcopy(fake.docs(INDEX))
    args = argparse.Namespace(
        index=INDEX, crop_id_prefix=None, page_size=500, dry_run=True, verbose=True
    )
    assert await run(args, fake) == 0
    assert fake.docs(INDEX) == before
    out = capsys.readouterr().out
    assert '5 candidate(s); 4 repairable, 1 report-only' in out
    assert 'Dry-run only' in out
