"""K3: operator repair for a class-less item that still carries a
``label_source`` -- "who labeled this" recorded with no label to
attribute. Current writers were audited and no longer do this (an empty
VLM class answer only records the attempt, see vlm_class_attempt.py);
this repairs legacy data / any other bypass.
"""

from __future__ import annotations

from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.services.curation import stale_label_source_repair as repair


INDEX = 'test_items'


def _docs() -> dict[str, dict[str, Any]]:
    return {
        # Candidate: class-less, stale label_source from an old vlm_unmatched write.
        'a_stale': {
            'crop_id': 'a_stale',
            'class_source': 'vlm_unmatched',
            'label_source': 'vlm',
            'class_validated': False,
        },
        # Candidate: class-less, stale label_source with no class_source at all.
        'b_no_source': {
            'crop_id': 'b_no_source',
            'label_source': 'vlm',
            'class_validated': False,
        },
        # Not a candidate: has a real class_id.
        'c_classed': {
            'crop_id': 'c_classed',
            'class_id': 3,
            'class_name': 'widget',
            'class_source': 'vlm',
            'label_source': 'vlm',
            'class_validated': False,
        },
        # Not a candidate: class-less but label_source is already null.
        'd_clean': {
            'crop_id': 'd_clean',
            'class_source': 'item_proposal',
            'label_source': None,
            'class_validated': False,
        },
        # Not a candidate: human-validated -- never touched even if
        # (hypothetically) class-less with a label_source.
        'e_human_validated': {
            'crop_id': 'e_human_validated',
            'label_source': 'human',
            'class_validated': True,
        },
    }


@pytest.mark.asyncio
async def test_plan_repairs_finds_only_class_less_stale_label_source_docs() -> None:
    fake = QueryFakeOpenSearch({INDEX: _docs()})
    plans = {p.crop_id: p for p in await repair.plan_repairs(fake, index=INDEX)}
    assert set(plans) == {'a_stale', 'b_no_source'}
    assert plans['a_stale'].prior_class_source == 'vlm_unmatched'
    assert plans['a_stale'].label_source == 'vlm'
    assert plans['b_no_source'].prior_class_source is None


def test_is_candidate_excludes_classed_clean_and_locked_docs() -> None:
    docs = _docs()
    assert repair.is_candidate(docs['a_stale']) is True
    assert repair.is_candidate(docs['b_no_source']) is True
    assert repair.is_candidate(docs['c_classed']) is False
    assert repair.is_candidate(docs['d_clean']) is False
    assert repair.is_candidate(docs['e_human_validated']) is False


@pytest.mark.asyncio
async def test_apply_repairs_clears_label_source_and_snapshots_history() -> None:
    fake = QueryFakeOpenSearch({INDEX: _docs()})
    plans = await repair.plan_repairs(fake, index=INDEX)
    counts = await repair.apply_repairs(fake, plans, index=INDEX)
    assert counts == {'repaired': 2, 'skipped_changed': 0, 'errors': 0}

    a = fake.docs(INDEX)['a_stale']
    assert a['label_source'] is None
    # class_source is untouched -- only label_source is stale, not the
    # class provenance itself.
    assert a['class_source'] == 'vlm_unmatched'
    assert a['class_id_history'], 'a restorable snapshot must be recorded'
    assert a['class_id_history'][-1]['class_source'] == 'vlm_unmatched'

    b = fake.docs(INDEX)['b_no_source']
    assert b['label_source'] is None

    # Untouched docs stay exactly as they were.
    assert fake.docs(INDEX)['c_classed']['label_source'] == 'vlm'
    assert fake.docs(INDEX)['e_human_validated']['label_source'] == 'human'


@pytest.mark.asyncio
async def test_apply_repairs_skips_a_doc_changed_since_the_plan() -> None:
    fake = QueryFakeOpenSearch({INDEX: _docs()})
    plans = await repair.plan_repairs(fake, index=INDEX)
    # Simulate a concurrent write that gives the item a real class.
    fake.docs(INDEX)['a_stale']['class_id'] = 9
    counts = await repair.apply_repairs(fake, plans, index=INDEX)
    assert counts['skipped_changed'] == 1
    assert counts['repaired'] == 1
