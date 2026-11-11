"""The geometry lock: a box a person drew, moved or deleted is never moved, deleted or
re-created by an automated writer (re-detect, open-vocabulary run, dataset-import
reconcile, re-ingest). The predicate is :func:`src.clients.occ_locks.is_locked_geometry`;
every writer reaches it through ``is_locked_item``, so no writer can forget it."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest

from src.clients.occ_locks import (
    is_human_owned_item,
    is_locked_geometry,
    is_locked_item,
    lock_includes,
)


if TYPE_CHECKING:
    from pathlib import Path

MACHINE: dict[str, Any] = {'class_source': 'unlabeled_proposal', 'class_validated': False}
DELETED: dict[str, Any] = {
    'class_excluded': True,
    'excluded_by': 'human',
    'excluded_reason': 'not_an_object',
}

CASES: dict[str, tuple[dict[str, Any], bool]] = {
    'detector_box': ({**MACHINE, 'geometry_source': 'detector'}, False),
    'never_edited': (dict(MACHINE), False),
    'drawn_or_moved_by_a_person': ({**MACHINE, 'geometry_source': 'human'}, True),
    'human_deleted_as_not_an_object': ({**MACHINE, **DELETED}, True),
    'human_ignored_for_another_reason': (
        {**MACHINE, **DELETED, 'excluded_reason': 'blurry'},
        False,
    ),
    'machine_excluded_as_not_an_object': ({**MACHINE, **DELETED, 'excluded_by': 'vlm'}, False),
    'held_out': ({**MACHINE, 'test_holdout': True}, True),
    'validated_import': (
        {'class_source': 'external_label', 'class_validated': True, 'import_ids': ['i1']},
        True,
    ),
    'suggestion_import': (
        {'class_source': 'external_label', 'class_validated': False, 'import_ids': ['i1']},
        False,
    ),
    'import_proposal': ({**MACHINE, 'import_ids': ['i1'], 'proposed_by_import': 'i1'}, False),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_predicate_cases(name: str) -> None:
    doc, locked = CASES[name]
    assert is_locked_geometry(doc) is locked, name


@pytest.mark.parametrize('name', sorted(CASES))
def test_every_writer_that_asks_the_item_lock_gets_the_geometry_lock(name: str) -> None:
    doc, locked = CASES[name]
    if locked:
        assert is_locked_item(doc), name


@pytest.mark.parametrize('name', sorted(CASES))
def test_lock_includes_carries_every_field_the_predicates_read(name: str) -> None:
    """A query that fetches only ``lock_includes()`` must reach the same verdicts,
    or a writer that passes it would silently skip the lock."""
    doc, _ = CASES[name]
    wanted = set(lock_includes())
    fetched = {k: v for k, v in doc.items() if k in wanted}
    assert is_locked_geometry(fetched) is is_locked_geometry(doc)
    assert is_locked_item(fetched) is is_locked_item(doc)
    assert is_human_owned_item(fetched) is is_human_owned_item(doc)


@pytest.mark.parametrize(
    ('doc', 'owned'),
    [
        ({**MACHINE, 'geometry_source': 'human'}, True),
        ({**MACHINE, **DELETED}, True),
        ({**MACHINE, 'geometry_source': 'detector'}, False),
        ({'class_source': 'external_label', 'class_validated': True, 'import_ids': ['i1']}, False),
    ],
)
def test_a_person_owning_the_geometry_keeps_an_import_from_reconciling_it_away(
    doc: dict[str, Any], owned: bool
) -> None:
    assert is_human_owned_item(doc) is owned


# --------------------------------------------------------------------------- writers


async def _drawn_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """One image holding one detector item; the item is then claimed by a person."""
    from curation.test_reprocess_detect import D1, _world

    return await _world(tmp_path, monkeypatch, [D1])


def _redetect(fake: Any, service: Any, image_id: str) -> Any:
    from curation.test_reprocess_detect import _factory, _req
    from src.services.curation.reprocess import apply_reprocess

    return apply_reprocess(fake, _req(image_id), service_factory=_factory(service))


@pytest.mark.asyncio
@pytest.mark.parametrize('detects_it_again', [True, False])
async def test_redetect_keeps_a_drawn_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detects_it_again: bool
) -> None:
    from curation.reprocess_fixtures import docs
    from curation.test_reprocess_detect import D1

    fake, triton, service, image_id, (crop_id,) = await _drawn_world(tmp_path, monkeypatch)
    docs(fake)[crop_id].update({'geometry_source': 'human', 'geometry_revision': 1})
    before = copy.deepcopy(docs(fake)[crop_id])
    triton.detections = [D1] if detects_it_again else []

    await _redetect(fake, service, image_id)

    assert list(docs(fake)) == [crop_id]
    after = docs(fake)[crop_id]
    after.pop('proposal_chain', None)
    assert after == before


@pytest.mark.asyncio
async def test_redetect_keeps_a_moved_item_and_does_not_recreate_its_old_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from curation.reprocess_fixtures import docs

    fake, _triton, service, image_id, (crop_id,) = await _drawn_world(tmp_path, monkeypatch)
    original = list(docs(fake)[crop_id]['bbox_norm'])
    docs(fake)[crop_id].update(
        {
            'bbox_norm': [0.62, 0.62, 0.95, 0.95],
            'detector_bbox_norm': original,
            'geometry_source': 'human',
            'geometry_revision': 2,
        }
    )
    before = copy.deepcopy(docs(fake)[crop_id])

    await _redetect(fake, service, image_id)  # the detector still finds the original box

    assert list(docs(fake)) == [crop_id]
    after = docs(fake)[crop_id]
    assert after.pop('proposal_chain') == ['primary:match']
    assert after == before


@pytest.mark.asyncio
@pytest.mark.parametrize('detects_it_again', [True, False])
async def test_redetect_does_not_recreate_a_human_deleted_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detects_it_again: bool
) -> None:
    from curation.reprocess_fixtures import docs
    from curation.test_reprocess_detect import D1

    fake, triton, service, image_id, (crop_id,) = await _drawn_world(tmp_path, monkeypatch)
    docs(fake)[crop_id].update(DELETED)
    before = copy.deepcopy(docs(fake)[crop_id])
    triton.detections = [D1] if detects_it_again else []

    await _redetect(fake, service, image_id)

    assert list(docs(fake)) == [crop_id]
    after = docs(fake)[crop_id]
    after.pop('proposal_chain', None)
    assert after == before


def test_open_vocab_and_import_reconcile_respect_the_lock() -> None:
    from src.config.region_fields import get_region_fields
    from src.services.curation import open_vocab_run, reprocess_detect
    from src.services.curation.dataset_import.proposals import existing_items
    from src.services.curation.dataset_import.reconcile import reconcilable

    fields = get_region_fields()
    imported = {
        'class_source': 'external_label',
        'class_validated': True,
        'import_ids': ['old'],
    }
    assert reconcilable(imported, 'new', fields)  # control: an untouched import goes
    assert not reconcilable({**imported, 'geometry_source': 'human'}, 'new', fields)
    assert not reconcilable({**imported, **DELETED}, 'new', fields)

    wanted = set(lock_includes())
    assert wanted <= set(open_vocab_run._existing_includes())
    assert wanted <= set(reprocess_detect._existing_includes())

    (row,) = existing_items(
        [
            {
                'crop_id': 'a',
                'bbox_norm': [0.6, 0.6, 0.9, 0.9],
                **MACHINE,
                'geometry_source': 'human',
                'detector_bbox_norm': [0.1, 0.1, 0.5, 0.5],
            }
        ]
    )
    assert row.locked
    assert row.detector_bbox_norm == (0.1, 0.1, 0.5, 0.5)


def test_a_detection_on_the_machine_box_of_a_locked_item_is_merged_into_it() -> None:
    from src.services.curation.proposal_merge import ExistingItem, Proposal, merge_item_proposals

    moved = ExistingItem(
        crop_id='a',
        bbox_norm=(0.6, 0.6, 0.9, 0.9),
        class_name=None,
        locked=True,
        detector_bbox_norm=(0.1, 0.1, 0.5, 0.5),
    )
    at_old_spot = Proposal(bbox_norm=(0.1, 0.1, 0.5, 0.5), class_id=0, class_name='car', score=0.9)
    plan = merge_item_proposals([moved], [at_old_spot], remove_stale=True)
    assert plan.merged_into_locked == [('a', at_old_spot)]
    assert plan.created == []

    # IoU just under LABEL_IOU_MATCH on the old box (0.4 wide strip) still creates.
    away = Proposal(bbox_norm=(0.1, 0.1, 0.5, 0.2), class_id=0, class_name='car', score=0.9)
    assert merge_item_proposals([moved], [away]).created == [away]


def test_open_vocab_skips_a_hit_on_the_machine_box_of_a_locked_item() -> None:
    from src.services.detection.open_vocab_select import ExistingBox, locked_overlap

    moved = ExistingBox(
        bbox_norm=(0.6, 0.6, 0.9, 0.9),
        class_name=None,
        locked=True,
        detector_bbox_norm=(0.1, 0.1, 0.5, 0.5),
    )
    assert locked_overlap(moved, (0.1, 0.1, 0.5, 0.5))
    assert locked_overlap(moved, (0.6, 0.6, 0.9, 0.9))
    assert not locked_overlap(moved, (0.0, 0.7, 0.2, 0.9))
    unlocked = ExistingBox(bbox_norm=(0.6, 0.6, 0.9, 0.9), class_name=None, locked=False)
    assert not locked_overlap(unlocked, (0.6, 0.6, 0.9, 0.9))


@pytest.mark.asyncio
async def test_reingest_keeps_a_human_deleted_and_a_moved_item() -> None:
    from curation.query_fakes import QueryFakeOpenSearch
    from curation.reprocess_fixtures import docs, items_index
    from src.clients.occ_upsert import occ_upsert_bulk

    moved = {
        'crop_id': 'moved',
        'bbox_norm': [0.62, 0.62, 0.95, 0.95],
        'crop_area_norm': 0.11,
        'detector_bbox_norm': [0.05, 0.05, 0.6, 0.6],
        'geometry_source': 'human',
        'geometry_revision': 2,
        'embedding_state': 'deferred',
        **MACHINE,
    }
    deleted = {'crop_id': 'deleted', 'bbox_norm': [0.1, 0.1, 0.3, 0.3], **MACHINE, **DELETED}
    fake = QueryFakeOpenSearch({items_index(): {'moved': moved, 'deleted': deleted}})
    again = {
        'bbox_norm': [0.05, 0.05, 0.6, 0.6],
        'crop_area_norm': 0.3,
        'class_source': 'ingest',
        'label_source': 'ingest',
        'pe_embedding': [0.1, 0.2],
        'embedding_state': 'embedded',
    }
    await occ_upsert_bulk(
        fake,
        [
            {**again, 'crop_id': 'moved'},
            {**again, 'crop_id': 'deleted', 'bbox_norm': [0.1, 0.1, 0.3, 0.3]},
        ],
        human_field_guards=['class_source', 'label_source'],
        vector_excludes=[],
        vector_field='pe_embedding',
        state_field='embedding_state',
    )

    kept = docs(fake)['moved']
    assert kept['bbox_norm'] == moved['bbox_norm']
    assert kept['crop_area_norm'] == moved['crop_area_norm']
    assert 'pe_embedding' not in kept  # a vector of the detector box never lands on the moved box
    assert kept['embedding_state'] == 'deferred'
    assert kept['geometry_source'] == 'human'
    gone = docs(fake)['deleted']
    assert gone['class_excluded'] is True
    assert gone['excluded_reason'] == 'not_an_object'
