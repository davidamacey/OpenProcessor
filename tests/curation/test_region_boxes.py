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

from src.config.region_fields import RegionFields
from src.config.region_state import RegionStatus
from src.services.curation.region_boxes import (
    BOX_STATES,
    RegionBox,
    accepted,
    box_query,
    boxes_write_fields,
    derive_status,
    has_any_box_query,
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
