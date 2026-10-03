"""A class the user named as an open-vocabulary target is not relabeled by the VLM:
the VLM's answer rides along as a suggestion and the class fields stay."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.curation.vlm_worker import _build_pending_query
from src.services.curation.class_sources import vlm_suggestion
from src.services.curation.label_batch_write import label_batch_merge
from src.services.curation.vlm_class_attempt import class_attempt_fields


NOW = '2026-10-02T00:00:00+00:00'


def _target(**kw: Any) -> dict[str, Any]:
    return {
        'class_id': 7,
        'class_name': 'cone',
        'class_source': 'open_vocab_target',
        'cluster_id': 7,
        'source_prompt': 'traffic cone',
        **kw,
    }


def _vlm_update(**kw: Any) -> dict[str, Any]:
    return {
        'class_id': 3,
        'class_name': 'barrel',
        'class_source': 'vlm',
        'class_detector': 'vlm',
        'class_labeler': 'vlm',
        'cluster_id': 3,
        'label_source': 'vlm',
        'vlm_confidence': 'high',
        'vlm_raw_label': 'barrel',
        **class_attempt_fields(NOW),
        'updated_at': NOW,
        **kw,
    }


def _written(update: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {**current, **label_batch_merge(update, current)}


def test_a_vlm_relabel_keeps_the_target_class_and_records_a_suggestion() -> None:
    doc = _written(_vlm_update(), _target())
    assert (doc['class_id'], doc['class_name'], doc['class_source']) == (
        7,
        'cone',
        'open_vocab_target',
    )
    assert doc['cluster_id'] == 7
    assert doc['source_prompt'] == 'traffic cone'
    assert doc['vlm_class_attempted_at'] == NOW
    assert vlm_suggestion(doc) == (None, 'barrel')


def test_an_unmatched_or_new_class_answer_cannot_clear_the_target_class() -> None:
    unmatched = _vlm_update(class_source='vlm_unmatched', vlm_raw_class='thingy')
    unmatched.pop('class_id')
    unmatched.pop('class_name')
    doc = _written(unmatched, _target())
    assert (doc['class_id'], doc['class_name'], doc['class_source']) == (
        7,
        'cone',
        'open_vocab_target',
    )
    assert vlm_suggestion(doc) == (None, 'thingy')

    pending = _vlm_update(
        class_source='vlm_new_class_pending', vlm_proposed_class='pylon', needs_new_class=True
    )
    pending.pop('class_id')
    pending.pop('class_name')
    doc = _written(pending, _target())
    assert (doc['class_name'], doc['class_source']) == ('cone', 'open_vocab_target')
    assert not doc.get('needs_new_class')
    assert vlm_suggestion(doc) == (None, 'pylon')


def test_agreement_leaves_no_suggestion() -> None:
    doc = _written(_vlm_update(class_id=7, class_name='cone'), _target(vlm_proposed_class='old'))
    assert vlm_suggestion(doc) == (None, None)


def test_other_items_are_relabeled_as_before() -> None:
    proposal = {'class_source': 'open_vocab_proposal', 'proposal_name': 'barrel'}
    doc = _written(_vlm_update(), proposal)
    assert (doc['class_name'], doc['class_source']) == ('barrel', 'vlm')


def test_the_worker_asks_a_target_item_once() -> None:
    must_not = _build_pending_query(0.8)['bool']['must_not']
    clause = next(
        c
        for c in must_not
        if 'bool' in c and {'terms': {'class_source': ['open_vocab_target']}} in c['bool']['filter']
    )
    assert {'exists': {'field': 'vlm_class_attempted_at'}} in clause['bool']['filter']


@pytest.mark.parametrize('source', ['open_vocab_target'])
def test_the_target_source_is_in_the_catalog(source: str) -> None:
    from src.services.curation.class_sources import class_source_catalog

    entry = next(e for e in class_source_catalog() if e['id'] == source)
    assert entry['role'] == 'open_vocab'
