"""The pure half of every human write path: what the person did, as the claims to append."""

from __future__ import annotations

from typing import Any

import pytest

from src.domain import (
    claim_builders as cb,
    claim_builders_import as cbi,
    claim_builders_region as cbr,
)
from src.domain.claim_kinds import ActorKind, ClaimKind
from src.domain.claim_payloads import Origin
from src.domain.item_geometry import ItemBoxError
from src.domain.region_boxes import RegionBox, RegionBoxWriteError
from src.domain.region_edit_rules import BoxEdit
from tests.domain.fold_support import VIEW, Log, folded, inputs, opinion


def pl(draft: Any) -> Any:
    return draft.payload


def kinds(drafts) -> list[ClaimKind]:
    return [d.kind for d in drafts]


A = (0.10, 0.10, 0.30, 0.30)
B = (0.40, 0.40, 0.60, 0.60)
SHOWN = [RegionBox('b1', A, 'proposed'), RegionBox('b2', B, 'proposed')]


def test_labels_record_what_the_person_saw_and_how() -> None:
    (single,) = cb.label(VIEW, class_name='car', class_id_hint=1, op_id='o')
    assert single.kind is ClaimKind.CLASS_SET
    assert single.subject.bbox == VIEW.bbox
    assert single.subject.machine_bbox == VIEW.machine_bbox
    assert pl(single).via == 'single'
    (resolve,) = cb.resolve_new_class(VIEW, class_name='car', class_id_hint=1, op_id='o')
    assert (pl(resolve).via, pl(resolve).label_source) == ('resolve', 'new_class_proposal')
    (accept,) = cb.accept_machine(VIEW, class_name='car', class_id_hint=1, op_id='o')
    assert (pl(accept).via, pl(accept).label_source) == ('bulk_accept', 'human_bulk_accept')


def test_unlabel_steps_back_over_the_newest_label_operation_or_clears() -> None:
    log = Log()
    log.add(cb.label(VIEW, class_name='car', class_id_hint=1, op_id='o1'))
    log.add(cb.discard(VIEW, clear_class_too=True, dismiss_from_review=True, op_id='o2'))
    history = [
        cb.LiveOperation('o1', 'label', (1,)),
        cb.LiveOperation('o2', 'discard', (2, 3)),
    ]
    undo = cb.unlabel(VIEW, history, hidden=[], op_id='u')
    assert [d.withdraws_claim_id for d in undo] == [2, 3]  # a discard comes back whole
    accepted_only = [cb.LiveOperation('oa', 'accept_machine', (9,))]
    assert kinds(cb.unlabel(VIEW, accepted_only, hidden=[], op_id='u')) == [ClaimKind.CLASS_CLEARED]
    assert kinds(cb.unlabel(VIEW, [], hidden=[], op_id='u')) == [ClaimKind.CLASS_CLEARED]


def test_discard_variants() -> None:
    assert kinds(cb.discard(VIEW, clear_class_too=True, dismiss_from_review=False, op_id='o')) == [
        ClaimKind.CLASS_CLEARED
    ]
    assert kinds(cb.discard(VIEW, clear_class_too=False, dismiss_from_review=True, op_id='o')) == [
        ClaimKind.REVIEW_DISMISSED
    ]
    both = cb.discard(VIEW, clear_class_too=True, dismiss_from_review=True, op_id='o')
    assert kinds(both) == [ClaimKind.CLASS_CLEARED, ClaimKind.REVIEW_DISMISSED]
    assert {d.op_id for d in both} == {'o'}  # one operation: the undo brings both back


@pytest.mark.parametrize(
    ('drafts', 'kind'),
    [
        (
            cb.move_to_cluster(VIEW, partition_id='p', cluster_uid='u', op_id='o'),
            ClaimKind.CLASS_MOVED,
        ),
        (cb.exclude(VIEW, reason='blurry', op_id='o'), ClaimKind.EXCLUDED),
        (cb.restore(VIEW, op_id='o'), ClaimKind.RESTORED),
        (cb.review_dismiss(VIEW), ClaimKind.REVIEW_DISMISSED),
        (cb.review_undismiss(VIEW), ClaimKind.REVIEW_RESTORED),
        (cb.vlm_dismiss(VIEW, class_name='bus', class_id_hint=3), ClaimKind.VLM_DISMISSED),
        (cb.probe_dismiss(VIEW, class_name='bus', probe_analyser_id=2), ClaimKind.PROBE_DISMISSED),
        (cb.flag_new_class(VIEW, note='a boat'), ClaimKind.NEEDS_NEW_CLASS),
        (
            cb.audit_sample(VIEW, batch_id='b', label_name='car', label_source='vlm', op_id='o'),
            ClaimKind.AUDIT_SAMPLED,
        ),
        (
            cb.holdout_freeze(VIEW, freeze_id='f', machine_class=None, op_id='o'),
            ClaimKind.HOLDOUT_FROZEN,
        ),
        (
            cb.duplicate_verdict('a', 'b', level='crop', decision='same'),
            ClaimKind.DUPLICATE_VERDICT,
        ),
    ],
)
def test_single_claim_paths(drafts, kind) -> None:
    assert kinds(drafts) == [kind]
    assert drafts[0].actor_kind is ActorKind.HUMAN


def test_forgetting_a_freeze_and_a_duplicate_verdict_are_withdraws() -> None:
    drafts = cb.forget_holdout_freeze(VIEW, [4, 2], op_id='o')
    assert [d.withdraws_claim_id for d in drafts] == [2, 4]
    assert cb.withdraw(7, VIEW, op_id=None).kind is ClaimKind.WITHDRAW


def test_put_regions_is_a_diff() -> None:
    new = (0.7, 0.7, 0.9, 0.9)
    edits = [BoxEdit('b1', bbox=(0.12, 0.10, 0.30, 0.30)), BoxEdit(bbox=new, state='rejected')]
    write = cbr.put_regions(VIEW, SHOWN, edits, box_seq=2, region_seq=3, op_id='o')
    assert kinds(write.drafts) == [
        ClaimKind.BOX_MOVED,  # b1 moved
        ClaimKind.BOX_CREATED,  # the new box
        ClaimKind.BOX_DELETED,  # b2 was left out
    ]
    created = pl(write.drafts[1])
    assert (created.state, created.region_seq) == ('rejected', 3)
    assert write.drafts[1].subject.box_id == 'b3'  # minted from box_seq
    assert write.box_seq == 3
    assert write.drafts[0].subject.bbox == A  # the box as shown, not as moved


def test_a_confirmation_within_the_tolerance_is_no_claim() -> None:
    write = cbr.put_regions(
        VIEW,
        SHOWN,
        [BoxEdit('b1', bbox=(0.10005, 0.10, 0.30, 0.30)), BoxEdit('b2')],
        box_seq=2,
        region_seq=0,
        op_id='o',
    )
    assert write.drafts == []


def test_patch_box_and_batch_box_state_touch_only_the_named_boxes() -> None:
    patched = cbr.patch_box(
        VIEW, SHOWN, 'b2', state='accepted', text='AB1', box_seq=2, region_seq=0, op_id='o'
    )
    assert [(d.kind, d.subject.box_id) for d in patched.drafts] == [
        (ClaimKind.BOX_STATE_SET, 'b2'),
        (ClaimKind.BOX_TEXT_SET, 'b2'),
    ]
    batch = cbr.batch_box_state(VIEW, SHOWN, ['b1'], 'rejected', box_seq=2, region_seq=0, op_id='o')
    assert [d.subject.box_id for d in batch.drafts] == ['b1']
    assert pl(batch.drafts[0]).rejection_reason == 'human'


def test_a_whole_set_verdict_writes_a_state_claim_for_every_box() -> None:
    write = cbr.set_region_status(VIEW, SHOWN, 'detected', box_seq=2, region_seq=1, op_id='o')
    assert [(d.kind, d.subject.box_id, pl(d).state) for d in write.drafts] == [
        (ClaimKind.BOX_STATE_SET, 'b1', 'accepted'),
        (ClaimKind.BOX_STATE_SET, 'b2', 'accepted'),
    ]
    none_visible = cbr.set_region_status(
        VIEW, SHOWN, 'no_region_visible', box_seq=2, region_seq=1, op_id='o'
    )
    assert kinds(none_visible.drafts) == [
        ClaimKind.BOX_DELETED,
        ClaimKind.BOX_DELETED,
        ClaimKind.REGION_STATUS_SET,
    ]
    with pytest.raises(RegionBoxWriteError, match='no_boxes'):
        cbr.set_region_status(VIEW, [], 'detected', box_seq=0, region_seq=0, op_id='o')


def test_an_edit_that_closes_the_set_claims_every_box() -> None:
    write = cbr.patch_box(
        VIEW, SHOWN[:1], 'b1', state='accepted', box_seq=1, region_seq=0, op_id='o'
    )
    assert kinds(write.drafts) == [ClaimKind.BOX_STATE_SET]
    two = [RegionBox('b1', A, 'accepted'), RegionBox('b2', B, 'proposed')]
    closing = cbr.patch_box(VIEW, two, 'b2', state='rejected', box_seq=2, region_seq=0, op_id='o')
    assert [d.subject.box_id for d in closing.drafts] == ['b1', 'b2']  # b1 unchanged, claimed too


def test_geometry_edit_and_its_dropped_boxes() -> None:
    drafts = cbr.set_geometry(VIEW, (0.0, 0.0, 0.7, 0.7), drop=SHOWN[:1], region_seq=2, op_id='o')
    assert kinds(drafts) == [ClaimKind.GEOMETRY_SET, ClaimKind.BOX_DELETED]
    with pytest.raises(ItemBoxError):
        cbr.set_geometry(VIEW, (0.1, 0.1, 0.1005, 0.5), region_seq=0, op_id='o')


def test_a_drawn_object_inherits_the_holdout_of_its_photo() -> None:
    plain = cbr.draw_object(VIEW, class_name='car', class_id_hint=1, request_id='r', op_id='o')
    assert kinds(plain) == [ClaimKind.ITEM_DRAWN, ClaimKind.CLASS_SET]
    held = cbr.draw_object(
        VIEW, class_name='car', class_id_hint=1, request_id='r', freeze=('f1', None), op_id='o'
    )
    assert kinds(held)[-1] is ClaimKind.HOLDOUT_FROZEN
    log = Log()
    log.add(held)
    state = folded(inputs(log.claims, detections=[])).state
    assert (state.class_name, state.holdout, state.geometry_source) == ('car', True, 'human')


def test_import_and_combine_claims_carry_their_actor() -> None:
    (label,) = cbi.import_label(VIEW, import_id='i1', class_name='car', class_id_hint=1, op_id='o')
    assert (label.actor_kind, label.actor, pl(label).via) == (
        ActorKind.IMPORT,
        'import:i1',
        'import',
    )
    item = cbi.import_item(
        VIEW,
        import_id='i1',
        class_name='car',
        class_id_hint=1,
        standalone_region=True,
        origin=Origin(project='p', crop_id='c', image_id='i'),
        op_id='o',
    )
    assert kinds(item) == [ClaimKind.ITEM_DRAWN, ClaimKind.CLASS_SET]
    (box,) = cbi.import_box(
        VIEW, import_id='i1', box_id='b1', bbox=A, state='accepted', region_seq=0, op_id='o'
    )
    assert box.actor_kind is ActorKind.IMPORT
    (combine,) = cbi.combine_label(VIEW, job_id='j', class_name='car', class_id_hint=1, op_id='o')
    assert (combine.actor_kind, pl(combine).via) == (ActorKind.COMBINE, 'combine')
    (conflict,) = cbi.combine_conflict(
        VIEW, job_id='j', origins=['a', 'b'], merged_origins=['a'], op_id='o'
    )
    assert conflict.kind is ClaimKind.COMBINE_CONFLICT


def test_undo_of_an_operation_on_one_item_withdraws_every_claim_of_it() -> None:
    drafts = cb.withdraw_operation(VIEW, cb.LiveOperation('o', 'label', (9, 3, 5)), op_id='u')
    assert [d.withdraws_claim_id for d in drafts] == [3, 5, 9]


def test_accept_machine_follows_the_accept_rule_on_the_folded_class() -> None:
    log = Log()
    ops = [opinion('classifier', 'car', outcome='model', confidence=0.9)]
    state = folded(inputs(log.claims, opinions=ops)).state
    args: dict[str, Any] = {'detector_sources': {state.class_source}, 'vlm_sources': set()}
    assert cb.accept_machine_eligible(state, min_confidence=0.5, **args)
    assert not cb.accept_machine_eligible(state, min_confidence=0.95, **args)
    log.add(cb.exclude(VIEW, reason='blurry', op_id='e'))
    assert not cb.accept_machine_eligible(
        folded(inputs(log.claims, opinions=ops)).state, min_confidence=0.5, **args
    )
