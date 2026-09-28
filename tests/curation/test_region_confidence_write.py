"""IT-3 residual: the verifier's box confidence must land in
``RegionFields.confidence``.

Before the fix, ``reply.region_confidence`` / ``outcome.confidence`` (the
VLM's ``high``/``medium``/``low`` box verdict) was read for the
auto-confirm gate and for ``region_text_confidence`` (a *different* field --
grades the text reading, not the box) but never written to
``RegionFields.confidence`` itself, so the mapped field stayed 0/N forever.
It is already cleared on requeue (``region_requeue.py``'s
``detection_fields``), so a write here is the only missing piece.

W8: every live worker write path's box confidence now lands on
``RegionBox.confidence`` (per box, via ``verify.verdicts_to_boxes`` --
already covered by that function's own dedicated tests) instead of the
item-level ``RegionFields.confidence`` this file's now-deleted
``_combined_write_doc`` used to set. ``_region_write_doc`` itself is no
longer called from the live pipeline either (the segmenter
high-confidence auto-skip and no-VLM-configured paths both build a
``RegionBox`` directly now) -- it stays defined and tested here as the
legacy single-scalar write-doc builder Item 2 (legacy scalar field
removal) will reconcile, not deleted mid-pass while the scalar fields it
targets are still read by ~20 other consumers untouched this pass.
"""

from __future__ import annotations

from scripts.curation.worker.verify import _region_write_doc
from src.config import get_region_fields


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
