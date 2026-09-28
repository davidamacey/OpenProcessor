"""W10.7/W10.17: region-box attachment to parent items."""

from __future__ import annotations

from src.services.curation.dataset_import.regions import ParentCandidate, attach_region_boxes
from src.services.curation.dataset_import.scan import LabelBox


def test_containment_picks_smallest_containing_parent() -> None:
    small_parent = ParentCandidate(key='small', bbox_norm=(0.0, 0.0, 0.5, 0.5))
    big_parent = ParentCandidate(key='big', bbox_norm=(0.0, 0.0, 1.0, 1.0))
    box = LabelBox(dataset_class='wheel', bbox_norm=(0.1, 0.1, 0.3, 0.3))
    result = attach_region_boxes([small_parent, big_parent], [box])
    assert len(result.attachments) == 1
    assert result.attachments[0].parent_key == 'small'
    assert result.standalone == []


def test_box_outside_every_parent_is_standalone() -> None:
    parent = ParentCandidate(key='p1', bbox_norm=(0.0, 0.0, 0.3, 0.3))
    box = LabelBox(dataset_class='wheel', bbox_norm=(0.6, 0.6, 0.9, 0.9))
    result = attach_region_boxes([parent], [box])
    assert result.attachments == []
    assert len(result.standalone) == 1


def test_containment_threshold_respected() -> None:
    parent = ParentCandidate(key='p1', bbox_norm=(0.0, 0.0, 0.5, 0.5))
    # Half the box sticks out -> containment 0.5, below the 0.9 default.
    box = LabelBox(dataset_class='wheel', bbox_norm=(0.4, 0.4, 0.6, 0.6))
    result = attach_region_boxes([parent], [box])
    assert result.attachments == []
    assert len(result.standalone) == 1


def test_parent_classes_filter(tmp_path=None) -> None:
    car = ParentCandidate(key='car', bbox_norm=(0.0, 0.0, 1.0, 1.0), class_name='car')
    truck = ParentCandidate(key='truck', bbox_norm=(0.0, 0.0, 1.0, 1.0), class_name='truck')
    box = LabelBox(dataset_class='wheel', bbox_norm=(0.1, 0.1, 0.3, 0.3))
    result = attach_region_boxes([car, truck], [box], parent_classes=frozenset({'car'}))
    assert result.attachments[0].parent_key == 'car'


def test_one_parent_can_take_many_boxes() -> None:
    parent = ParentCandidate(key='p1', bbox_norm=(0.0, 0.0, 1.0, 1.0))
    boxes = [
        LabelBox(dataset_class='wheel', bbox_norm=(0.0, 0.0, 0.2, 0.2)),
        LabelBox(dataset_class='wheel', bbox_norm=(0.8, 0.8, 1.0, 1.0)),
    ]
    result = attach_region_boxes([parent], boxes)
    assert len(result.attachments) == 2
    assert {a.parent_key for a in result.attachments} == {'p1'}


def test_deterministic_tie_break_by_key() -> None:
    p1 = ParentCandidate(key='b_parent', bbox_norm=(0.0, 0.0, 0.5, 0.5))
    p2 = ParentCandidate(key='a_parent', bbox_norm=(0.0, 0.0, 0.5, 0.5))
    box = LabelBox(dataset_class='wheel', bbox_norm=(0.1, 0.1, 0.3, 0.3))
    result = attach_region_boxes([p1, p2], [box])
    assert result.attachments[0].parent_key == 'a_parent'
