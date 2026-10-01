"""Tests for the W8 per-box storage primitives (``region_boxes.py``).

Scope note (honest, see final report): this covers the core list-shaped
storage primitives (``RegionBox``, ``read_boxes``, ``boxes_write_fields``,
``next_box_id``, ``derive_status``, ``box_query``) and the served
``box_states`` / ``tone`` catalog addition. It does NOT cover the worker
pipeline rewrite, VLM overlay, migration, embeddings or the human edit
routes — those are explicitly out of scope for this pass (see the final
report handed back to the coordinator).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.config.region_fields import RegionFields
from src.config.region_rejection import REJECT_REASON_HUMAN
from src.config.region_state import RegionStatus
from src.services.curation.region_boxes import (
    BOX_STATES,
    RegionBox,
    RegionBoxWriteError,
    accepted,
    apply_put_boxes,
    box_query,
    boxes_with_status,
    boxes_write_fields,
    derive_status,
    has_any_box_query,
    is_human_owned,
    next_box_id,
    read_boxes,
)


F = RegionFields()


def test_box_states_values() -> None:
    assert BOX_STATES == ('proposed', 'accepted', 'rejected', 'false_positive')


def test_read_boxes_empty_source() -> None:
    assert read_boxes({}) == []


def test_read_boxes_round_trips_to_doc() -> None:
    box = RegionBox(box_id='b1', bbox_norm=(0.1, 0.2, 0.3, 0.4), state='accepted', score=0.9)
    src = {F.boxes: [box.to_doc()]}
    boxes = read_boxes(src)
    assert boxes == [box]


def test_next_box_id_never_reuses_after_delete() -> None:
    # seq=3 means b3 was the highest ever assigned, even if no box with
    # that id currently exists (it was deleted).
    assert next_box_id([], seq=3) == 'b4'


def test_next_box_id_from_existing() -> None:
    existing = [RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', score=1.0)]
    assert next_box_id(existing, seq=0) == 'b2'


def test_boxes_write_fields_sets_counts_and_bumps_revision() -> None:
    boxes = [
        RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', score=0.9),
        RegionBox(box_id='b2', bbox_norm=(0, 0, 1, 1), state='rejected', score=0.5),
    ]
    current_src = {F.revision: 4, F.box_seq: 1}
    doc = boxes_write_fields(boxes, current_src=current_src)
    assert doc[F.count] == 1
    assert doc[F.rejected_count] == 1
    assert doc[F.max_score] == 0.9
    assert doc[F.revision] == 5
    assert doc[F.box_seq] == 2
    assert doc[F.boxes] == [b.to_doc() for b in boxes]


def test_boxes_write_fields_accepted_plus_rejected_has_no_item_level_reason() -> None:
    # W8-cleanup N1 regression: an ordinary multi-box `detected` item
    # (one accepted box, one rejected sibling) must NOT carry a
    # rejection reason on the item mirror -- that field is only for
    # box-less rejected/no_region_visible items. Otherwise the labeler
    # renders a red "Rejection" row instead of "needs confirmation".
    boxes = [
        RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', score=0.9),
        RegionBox(
            box_id='b2',
            bbox_norm=(0, 0, 1, 1),
            state='rejected',
            score=0.5,
            rejection_reason='region_visible_elsewhere',
        ),
    ]
    doc = boxes_write_fields(boxes, current_src={})
    assert doc[F.rejection_reason] is None


def test_boxes_write_fields_all_rejected_keeps_item_level_reason() -> None:
    boxes = [
        RegionBox(
            box_id='b1',
            bbox_norm=(0, 0, 1, 1),
            state='rejected',
            score=0.5,
            rejection_reason='blurry',
        ),
    ]
    doc = boxes_write_fields(boxes, current_src={})
    assert doc[F.rejection_reason] == 'blurry'


def test_boxes_write_fields_mirror_prefers_accepted_over_higher_scoring_fp() -> None:
    # W8-cleanup N2 regression: an accepted box must win the mirror even
    # when a false_positive sibling scores higher.
    boxes = [
        RegionBox(box_id='b1', bbox_norm=(0.1, 0.1, 0.2, 0.2), state='false_positive', score=0.95),
        RegionBox(box_id='b2', bbox_norm=(0.5, 0.5, 0.6, 0.6), state='accepted', score=0.4),
    ]
    doc = boxes_write_fields(boxes, current_src={})
    assert doc[F.bbox_norm] == [0.5, 0.5, 0.6, 0.6]
    assert doc[F.score] == 0.4


def test_boxes_write_fields_empty_list() -> None:
    doc = boxes_write_fields([], current_src={})
    assert doc[F.count] == 0
    assert doc[F.rejected_count] == 0
    assert doc[F.max_score] is None
    assert doc[F.revision] == 1
    assert doc[F.box_seq] == 0
    assert doc[F.boxes] == []


def test_derive_status_precedence() -> None:
    accepted_box = RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', score=0.9)
    fp_box = RegionBox(box_id='b2', bbox_norm=(0, 0, 1, 1), state='false_positive', score=0.9)
    proposed_box = RegionBox(box_id='b3', bbox_norm=(0, 0, 1, 1), state='proposed', score=0.9)
    rejected_box = RegionBox(box_id='b4', bbox_norm=(0, 0, 1, 1), state='rejected', score=0.9)

    assert derive_status([accepted_box, rejected_box], empty_status=RegionStatus.NO_REGION_BOX) is (
        RegionStatus.DETECTED
    )
    assert derive_status([fp_box, rejected_box], empty_status=RegionStatus.NO_REGION_BOX) is (
        RegionStatus.FALSE_POSITIVE
    )
    assert derive_status([proposed_box, rejected_box], empty_status=RegionStatus.NO_REGION_BOX) is (
        RegionStatus.PENDING_VERIFICATION
    )
    assert derive_status([rejected_box], empty_status=RegionStatus.NO_REGION_BOX) is (
        RegionStatus.VERIFY_REJECTED
    )
    assert derive_status([], empty_status=RegionStatus.NO_REGION_BOX) is RegionStatus.NO_REGION_BOX


def test_accepted_filters_by_state() -> None:
    a = RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', score=0.9)
    r = RegionBox(box_id='b2', bbox_norm=(0, 0, 1, 1), state='rejected', score=0.9)
    assert accepted([a, r]) == [a]


def test_box_query_wraps_in_nested() -> None:
    q = box_query({'term': {F.boxes_state: 'accepted'}})
    assert q == {'nested': {'path': F.boxes, 'query': {'term': {F.boxes_state: 'accepted'}}}}


def test_has_any_box_query_is_count_or_rejected() -> None:
    q = has_any_box_query()
    assert q == {
        'bool': {
            'should': [
                {'range': {F.count: {'gte': 1}}},
                {'range': {F.rejected_count: {'gte': 1}}},
            ],
            'minimum_should_match': 1,
        }
    }


# ---------------------------------------------------------------------------
# apply_put_boxes: sibling-preserving PUT merge (W8a, any_domain_plan.md
# §7.7 wire-write table + W8 pins 1-2)
# ---------------------------------------------------------------------------


def _existing_two_boxes() -> dict:
    b1 = RegionBox(box_id='b1', bbox_norm=(0.1, 0.1, 0.2, 0.2), state='accepted', score=0.9)
    b2 = RegionBox(box_id='b2', bbox_norm=(0.3, 0.3, 0.4, 0.4), state='proposed', score=0.5)
    return {F.boxes: [b1.to_doc(), b2.to_doc()], F.box_seq: 2, F.revision: 3}


def test_apply_put_boxes_untouched_sibling_by_id_only() -> None:
    current = _existing_two_boxes()
    # Only b1 referenced, by id alone -- b2 must survive untouched even
    # though it's omitted from most of the payload's attention.
    requested: list[dict[str, Any]] = [{'box_id': 'b1'}, {'box_id': 'b2'}]
    result = apply_put_boxes(current, requested, frame='source')
    assert [b.box_id for b in result] == ['b1', 'b2']
    assert result[0].bbox_norm == (0.1, 0.1, 0.2, 0.2)
    assert result[0].state == 'accepted'
    assert result[1].state == 'proposed'


def test_apply_put_boxes_omitting_a_stored_box_deletes_it() -> None:
    current = _existing_two_boxes()
    requested: list[dict[str, Any]] = [{'box_id': 'b1'}]
    result = apply_put_boxes(current, requested, frame='source')
    assert [b.box_id for b in result] == ['b1']


def test_apply_put_boxes_move_keeps_state() -> None:
    current = _existing_two_boxes()
    requested: list[dict[str, Any]] = [
        {'box_id': 'b1', 'bbox_norm': [0.15, 0.15, 0.25, 0.25]},
        {'box_id': 'b2'},
    ]
    result = apply_put_boxes(current, requested, frame='source')
    assert result[0].bbox_norm == (0.15, 0.15, 0.25, 0.25)
    assert result[0].state == 'accepted'


def test_apply_put_boxes_new_box_defaults_to_accepted() -> None:
    current = _existing_two_boxes()
    requested: list[dict[str, Any]] = [
        {'box_id': 'b1'},
        {'box_id': 'b2'},
        {'box_id': None, 'bbox_norm': [0.5, 0.5, 0.6, 0.6]},
    ]
    result = apply_put_boxes(current, requested, frame='source')
    assert result[2].box_id == 'b3'
    assert result[2].state == 'accepted'


def test_apply_put_boxes_new_box_explicit_state_honored() -> None:
    current = _existing_two_boxes()
    requested: list[dict[str, Any]] = [
        {'box_id': None, 'bbox_norm': [0.5, 0.5, 0.6, 0.6], 'state': 'rejected'},
    ]
    result = apply_put_boxes(current, requested, frame='source')
    assert result[0].state == 'rejected'


def test_apply_put_boxes_unknown_box_id_raises() -> None:
    current = _existing_two_boxes()
    with pytest.raises(RegionBoxWriteError):
        apply_put_boxes(current, [{'box_id': 'b99'}], frame='source')


def test_apply_put_boxes_patching_existing_box_to_rejected_stamps_human_reason() -> None:
    """W8c M3 fix: patching an EXISTING (machine-created) stored box to
    `rejected` via PUT is a human verdict too -- stamp the same reason
    `boxes_with_status`'s whole-set path uses so `is_human_owned`
    recognizes it later."""
    current = _existing_two_boxes()
    requested: list[dict[str, Any]] = [
        {'box_id': 'b1', 'state': 'rejected'},
        {'box_id': 'b2'},
    ]
    result = apply_put_boxes(current, requested, frame='source')
    assert result[0].state == 'rejected'
    assert result[0].rejection_reason == REJECT_REASON_HUMAN
    assert is_human_owned(result[0])


# ---------------------------------------------------------------------------
# is_human_owned (W8c M3)
# ---------------------------------------------------------------------------


def test_is_human_owned_true_for_human_created_box() -> None:
    box = RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', source='human')
    assert is_human_owned(box)


def test_is_human_owned_true_for_human_rejection_reason_on_a_machine_box() -> None:
    box = RegionBox(
        box_id='b1',
        bbox_norm=(0, 0, 1, 1),
        state='rejected',
        source='detector',
        detector='some_model',
        rejection_reason=REJECT_REASON_HUMAN,
    )
    assert is_human_owned(box)


def test_is_human_owned_true_for_human_transcribed_text_on_a_machine_box() -> None:
    box = RegionBox(
        box_id='b1',
        bbox_norm=(0, 0, 1, 1),
        state='accepted',
        source='detector',
        detector='some_model',
        text_source='human',
    )
    assert is_human_owned(box)


def test_is_human_owned_false_for_an_untouched_machine_box() -> None:
    box = RegionBox(
        box_id='b1',
        bbox_norm=(0, 0, 1, 1),
        state='rejected',
        source='detector',
        detector='some_model',
        rejection_reason='sanity_reject:aspect_ratio',
    )
    assert not is_human_owned(box)


# ---------------------------------------------------------------------------
# boxes_with_status: whole-set human status transitions (W8.7 table)
# ---------------------------------------------------------------------------


def test_boxes_with_status_detected_accepts_every_proposed() -> None:
    boxes = [
        RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='proposed', score=0.9),
        RegionBox(box_id='b2', bbox_norm=(0, 0, 1, 1), state='rejected', score=0.9),
    ]
    result = boxes_with_status('detected', boxes)
    assert result[0].state == 'accepted'
    assert result[1].state == 'rejected'


def test_boxes_with_status_detected_no_boxes_raises() -> None:
    with pytest.raises(RegionBoxWriteError):
        boxes_with_status('detected', [])


def test_boxes_with_status_detected_no_accepted_result_raises() -> None:
    boxes = [RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='rejected', score=0.9)]
    with pytest.raises(RegionBoxWriteError):
        boxes_with_status('detected', boxes)


def test_boxes_with_status_false_positive_flips_every_box() -> None:
    boxes = [
        RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', score=0.9),
        RegionBox(box_id='b2', bbox_norm=(0, 0, 1, 1), state='proposed', score=0.5),
    ]
    result = boxes_with_status('false_positive', boxes)
    assert all(b.state == 'false_positive' for b in result)


def test_boxes_with_status_verify_rejected_sets_human_reason() -> None:
    boxes = [RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', score=0.9)]
    result = boxes_with_status('verify_rejected', boxes)
    assert result[0].state == 'rejected'
    assert result[0].rejection_reason == 'human'


def test_boxes_with_status_no_region_visible_clears() -> None:
    boxes = [RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', score=0.9)]
    assert boxes_with_status('no_region_visible', boxes) == []


# ---------------------------------------------------------------------------
# Item-level wire serialization of the box list (W8a, any_domain_plan.md
# §7.7 "wire read")
# ---------------------------------------------------------------------------


def test_serialize_item_carries_region_boxes_and_stats() -> None:
    from src.services.curation.wire import serialize_item

    box = RegionBox(box_id='b1', bbox_norm=(0.1, 0.2, 0.3, 0.4), state='accepted', score=0.9)
    src = {
        'crop_id': 'x',
        F.boxes: [box.to_doc()],
        F.count: 1,
        F.rejected_count: 0,
        F.max_score: 0.9,
        F.set_complete: True,
        F.revision: 2,
    }
    item = serialize_item(src, 'x')
    # bbox_in_parent / thumbnail_url / locked are derived wire-only additions
    # on top of the stored element (W8.9, W10 lock rule) -- not part of
    # RegionBox.to_doc().
    assert item['region_boxes'] == [
        {
            **box.to_doc(),
            'bbox_in_parent': None,
            'locked': False,
            'thumbnail_url': item['region_boxes'][0]['thumbnail_url'],
        }
    ]
    assert item['region_boxes'][0]['thumbnail_url'].endswith('/crops/x/region_thumbnail?box_id=b1')
    assert item['region_count'] == 1
    assert item['region_rejected_count'] == 0
    assert item['region_max_score'] == 0.9
    assert item['region_set_complete'] is True
    assert item['region_revision'] == 2


def test_serialize_item_defaults_when_absent() -> None:
    from src.services.curation.wire import serialize_item

    item = serialize_item({'crop_id': 'x'}, 'x')
    assert item['region_boxes'] == []
    assert item['region_count'] == 0
    assert item['region_rejected_count'] == 0
    assert item['region_max_score'] is None
    assert item['region_set_complete'] is None
    assert item['region_revision'] == 0
