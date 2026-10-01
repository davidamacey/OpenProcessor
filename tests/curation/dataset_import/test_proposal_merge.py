"""``merge_item_proposals``: a machine proposal never overwrites a locked
(human or imported) label, and the plan is deterministic."""

from __future__ import annotations

from src.services.curation.proposal_merge import (
    DISAGREEMENT_CLASS_MISMATCH,
    DISAGREEMENT_MISSED_LABEL,
    DISAGREEMENT_UNMATCHED_DETECTION,
    ExistingItem,
    Proposal,
    count_disagreements,
    merge_item_proposals,
)


BOX = (0.1, 0.1, 0.5, 0.5)
OTHER = (0.6, 0.6, 0.9, 0.9)


def _item(cid: str, box=BOX, cls: str | None = 'car', locked: bool = False) -> ExistingItem:
    return ExistingItem(crop_id=cid, bbox_norm=box, class_name=cls, locked=locked)


def _prop(box=BOX, cls: str | None = 'car', score: float = 0.9) -> Proposal:
    return Proposal(bbox_norm=box, class_id=0, class_name=cls, score=score)


def test_proposal_matching_a_locked_item_is_merged_not_created() -> None:
    plan = merge_item_proposals([_item('a', locked=True)], [_prop()])
    assert plan.merged_into_locked == [('a', _prop())]
    assert plan.created == []
    assert plan.replaced == []
    assert plan.disagreements == []


def test_class_difference_on_a_locked_match_is_a_mismatch_record() -> None:
    plan = merge_item_proposals([_item('a', cls='car', locked=True)], [_prop(cls='truck')])
    assert [d.kind for d in plan.disagreements] == [DISAGREEMENT_CLASS_MISMATCH]
    assert plan.disagreements[0].label_class == 'car'
    assert plan.disagreements[0].detector_class == 'truck'


def test_unlocked_item_same_box_is_refreshed_and_moved_box_is_replaced() -> None:
    same = merge_item_proposals([_item('a')], [_prop()])
    assert same.refreshed == [('a', _prop())]
    moved = merge_item_proposals([_item('a')], [_prop(box=(0.12, 0.12, 0.52, 0.52))])
    assert [cid for cid, _ in moved.replaced] == ['a']
    assert moved.refreshed == []


def test_unmatched_proposal_becomes_a_new_item() -> None:
    plan = merge_item_proposals([_item('a', locked=True)], [_prop(), _prop(box=OTHER)])
    assert plan.created == [_prop(box=OTHER)]
    assert [d.kind for d in plan.disagreements] == [DISAGREEMENT_UNMATCHED_DETECTION]


def test_locked_item_nothing_matched_is_a_missed_label_never_removed() -> None:
    plan = merge_item_proposals([_item('a', locked=True)], [], remove_stale=True)
    assert plan.removed == []
    assert [d.kind for d in plan.disagreements] == [DISAGREEMENT_MISSED_LABEL]


def test_stale_unlocked_item_is_removed_only_when_asked() -> None:
    keep = merge_item_proposals([_item('a')], [])
    assert keep.removed == []
    drop = merge_item_proposals([_item('a')], [], remove_stale=True)
    assert drop.removed == ['a']


def test_each_item_and_proposal_match_at_most_once_highest_iou_first() -> None:
    # Two proposals overlap one locked item; the closer one wins the match
    # and the other is a new item.
    far = (0.12, 0.12, 0.52, 0.52)
    plan = merge_item_proposals([_item('a', locked=True)], [_prop(box=far), _prop()])
    assert plan.merged_into_locked == [('a', _prop())]
    assert plan.created == [_prop(box=far)]


def test_plan_is_deterministic_under_input_order() -> None:
    items = [_item('b', box=OTHER, locked=True), _item('a', locked=True)]
    props = [_prop(box=OTHER), _prop()]
    forward = merge_item_proposals(items, props)
    backward = merge_item_proposals(list(reversed(items)), list(reversed(props)))
    assert sorted(forward.merged_into_locked, key=lambda m: m[0]) == sorted(
        backward.merged_into_locked, key=lambda m: m[0]
    )


def test_negative_frame_reports_every_detection_as_unmatched() -> None:
    plan = merge_item_proposals([], [_prop()], on_negative_frame=True)
    assert plan.created == [_prop()]
    assert count_disagreements(plan.disagreements) == {
        'mismatches': 0,
        'missed_labels': 0,
        'unmatched_detections': 1,
    }
    quiet = merge_item_proposals([], [_prop()])
    assert quiet.disagreements == []
