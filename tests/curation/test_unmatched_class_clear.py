"""Tests for ``unmatched_class_clear`` (IT-2).

A ``class_source='vlm_unmatched'`` write means "the VLM read a label that
isn't in the registry" -- it must not leave the item's prior class_id/name
in place, since the write's own class_source says nothing was matched. See
``src/services/curation/class_sources.py`` and the two write paths that call
this helper: ``scripts/curation/worker/bulk_writer.py`` (the worker's
combined detect+verify path) and ``src/routers/curation/vlm.py`` (the
label-batch endpoint).
"""

from __future__ import annotations

from src.services.curation.class_sources import unmatched_class_clear
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET


def _current(**overrides: object) -> dict[str, object]:
    base = {
        'class_id': 5,
        'class_name': 'foo',
        'class_source': 'coco_yolo11_model',
        'class_detector': 'coco_yolo11',
        'class_detector_version': '1',
        'class_labeler': 'coco_yolo11',
        'class_labeled_at': '2026-01-01T00:00:00Z',
        'cluster_id': 5,
    }
    base.update(overrides)
    return base


def test_class_range_cluster_is_reset_to_unassigned() -> None:
    out = unmatched_class_clear(_current(cluster_id=5))
    assert out['class_id'] is None
    assert out['class_name'] is None
    assert out['class_detector'] is None
    assert out['class_detector_version'] is None
    assert out['class_labeler'] is None
    assert out['class_labeled_at'] is None
    assert out['cluster_id'] == -1
    assert out['cluster_subid'] is None


def test_candidate_cluster_is_left_alone() -> None:
    out = unmatched_class_clear(_current(cluster_id=RESIDUAL_CLUSTER_ID_OFFSET + 42))
    assert out['class_id'] is None
    assert 'cluster_id' not in out
    assert 'cluster_subid' not in out


def test_unassigned_cluster_is_left_alone() -> None:
    out = unmatched_class_clear(_current(cluster_id=-1))
    assert out['class_id'] is None
    assert 'cluster_id' not in out


def test_validated_item_returns_empty() -> None:
    assert unmatched_class_clear(_current(class_validated=True)) == {}


def test_human_owned_item_returns_empty() -> None:
    assert unmatched_class_clear(_current(class_source='human')) == {}
