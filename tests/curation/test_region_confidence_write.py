"""IT-3 residual: the verifier's box confidence must land in
``RegionFields.confidence``.

Before the fix, ``reply.region_confidence`` / ``outcome.confidence`` (the
VLM's ``high``/``medium``/``low`` box verdict) was read for the
auto-confirm gate and for ``region_text_confidence`` (a *different* field --
grades the text reading, not the box) but never written to
``RegionFields.confidence`` itself, so the mapped field stayed 0/N forever.
It is already cleared on requeue (``region_requeue.py``'s
``detection_fields``), so a write here is the only missing piece.
"""

from __future__ import annotations

from scripts.curation.worker.verify import _combined_write_doc, _region_write_doc
from src.config import get_region_fields
from src.services.labeling.vlm_labeler import VlmCombinedReply


F = get_region_fields()


def test_region_write_doc_sets_confidence_field() -> None:
    doc = _region_write_doc(
        region_in_source=(0.1, 0.1, 0.2, 0.2),
        score=0.9,
        detector='det_model',
        detector_version='1',
        chain=[],
        confidence='high',
    )
    assert doc[F.confidence] == 'high'


def test_region_write_doc_omits_confidence_when_not_given() -> None:
    doc = _region_write_doc(
        region_in_source=(0.1, 0.1, 0.2, 0.2),
        score=0.9,
        detector='det_model',
        detector_version='1',
        chain=[],
    )
    assert F.confidence not in doc


def test_combined_write_doc_carries_region_confidence_into_confidence_field() -> None:
    doc = _combined_write_doc(
        reply=VlmCombinedReply(
            img_id='c1',
            region_visible=True,
            region_bbox_correct=True,
            region_confidence='high',
        ),
        candidate_in_source=(0.1, 0.1, 0.2, 0.2),
        candidate_score=0.9,
        detector='det_model',
        detector_version='1',
        chain=[],
        class_names=None,
    )
    assert doc[F.confidence] == 'high'
