"""``bulk_writer._vector_entries``: this pass's per-box vectors, keyed onto
the box ids the write-time merge minted -- and only where the vector is
still true of the box that lands."""

from __future__ import annotations

import dataclasses

from scripts.curation.worker.bulk_writer import _vector_entries
from scripts.curation.worker.state import _ItemTask
from src.services.curation.region_boxes import RegionBox, finalize_box_ids, new_box_placeholder


def _task(pending: list[RegionBox], vectors: dict[str, list[float]]) -> _ItemTask:
    t = _ItemTask(
        crop_id='c1',
        image_path='',
        item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        region_status='pending_detection',
        class_name='',
    )
    t.pending_boxes = pending
    t.box_vectors = vectors
    return t


def _fresh(i: int, state: str, x: float) -> RegionBox:
    return RegionBox(box_id=new_box_placeholder(i), bbox_norm=(x, 0.1, x + 0.2, 0.4), state=state)


def test_vectors_follow_the_minted_ids_and_only_accepted_boxes_get_one() -> None:
    accepted, rejected = _fresh(0, 'accepted', 0.1), _fresh(1, 'rejected', 0.5)
    task = _task(
        [accepted, rejected],
        {accepted.box_id: [1.0], rejected.box_id: [2.0]},
    )
    merged = [accepted, rejected]
    finalized = finalize_box_ids(merged, existing=[], seq=0)

    entries = _vector_entries(task, merged, finalized)

    assert entries == [
        {'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.30000000000000004, 0.4], 'embedding': [1.0]}
    ]


def test_a_box_a_human_moved_mid_flight_does_not_inherit_the_old_vector() -> None:
    stored = RegionBox(box_id='b1', bbox_norm=(0.1, 0.1, 0.3, 0.4), state='proposed')
    verified = dataclasses.replace(stored, state='accepted')
    task = _task([verified], {'b1': [1.0]})
    # The merge kept the human's moved box instead of this pass's verdict.
    moved = dataclasses.replace(stored, bbox_norm=(0.6, 0.6, 0.8, 0.9), state='accepted')

    assert _vector_entries(task, [moved], [moved]) == []


def test_a_box_that_the_merge_left_not_accepted_gets_no_vector() -> None:
    stored = RegionBox(box_id='b1', bbox_norm=(0.1, 0.1, 0.3, 0.4), state='accepted')
    task = _task([stored], {'b1': [1.0]})
    humans_reject = dataclasses.replace(stored, state='rejected')

    assert _vector_entries(task, [humans_reject], [humans_reject]) == []
