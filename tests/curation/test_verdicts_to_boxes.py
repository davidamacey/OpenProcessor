"""W8.5/W8.15: :func:`verdicts_to_boxes` -- verifier verdicts -> RegionBox list.

Standalone unit coverage of the pure verdict-to-storage function (not yet
wired into the streaming runner's stage consumers -- see the W8 handback
report). ``candidates`` are :class:`TaskBoxInput` (a bounded, decoupled
stand-in for the eventual ``TaskBox``).
"""

from __future__ import annotations

from scripts.curation.worker.verify import TaskBoxInput, verdicts_to_boxes
from src.config.region_rejection import (
    REJECT_REASON_NO_VERDICT,
    REJECT_REASON_SANITY_PREFIX,
    REJECT_REASON_VERIFIER,
)
from src.config.region_state import RegionStatus
from src.services.curation.region_boxes import new_box_placeholder
from src.services.labeling.region_overlay import VlmBoxVerdict


def _cand(
    bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.4, 0.4),
    *,
    box_id: str | None = None,
) -> TaskBoxInput:
    return TaskBoxInput(
        box_id=box_id,
        bbox_in_crop=bbox,
        bbox_in_source=bbox,
        score=0.9,
        detector='sam3',
        detector_version='1',
        source='segmenter',
    )


def test_accepted_box_from_true_verdict() -> None:
    boxes, status, _extra = verdicts_to_boxes(
        [_cand()],
        [VlmBoxVerdict(box=1, bbox_correct=True, confidence='high')],
    )
    assert status == RegionStatus.DETECTED
    assert len(boxes) == 1
    assert boxes[0].state == 'accepted'
    # W8 M1 fix: a fresh candidate gets a PLACEHOLDER id here -- a real
    # ``b<N>`` id is only minted at write time, against the CURRENT
    # stored ``region_box_seq`` (bulk_writer._merge / finalize_box_ids),
    # never against a snapshot this pure function has no access to.
    assert boxes[0].box_id == new_box_placeholder(0)
    assert boxes[0].confidence == 'high'


def test_rejected_box_from_false_verdict() -> None:
    boxes, status, _extra = verdicts_to_boxes(
        [_cand()],
        [VlmBoxVerdict(box=1, bbox_correct=False, confidence='low')],
    )
    assert status == RegionStatus.VERIFY_REJECTED
    assert boxes[0].state == 'rejected'
    assert boxes[0].rejection_reason == REJECT_REASON_VERIFIER


def test_no_verdict_box_rejected_with_no_verdict_reason() -> None:
    boxes, status, _extra = verdicts_to_boxes(
        [_cand()],
        [VlmBoxVerdict(box=1, bbox_correct=None, confidence=None)],
        force_resolve=True,
    )
    assert status == RegionStatus.VERIFY_REJECTED
    assert boxes[0].rejection_reason == REJECT_REASON_NO_VERDICT


def test_true_verdict_failing_sanity_gate_is_rejected_with_sanity_reason() -> None:
    # A degenerate (zero-area) bbox always fails the geometry gate.
    degenerate = (0.5, 0.5, 0.5, 0.5)
    boxes, status, _extra = verdicts_to_boxes(
        [_cand(bbox=degenerate)],
        [VlmBoxVerdict(box=1, bbox_correct=True, confidence='high')],
    )
    assert status == RegionStatus.VERIFY_REJECTED
    assert (boxes[0].rejection_reason or '').startswith(REJECT_REASON_SANITY_PREFIX)


def test_no_verdict_at_all_returns_none_status_for_retry() -> None:
    """When nothing got a verdict, the caller retries (no write) -- unless
    ``force_resolve`` (the no-verdict cap)."""
    boxes, status, extra = verdicts_to_boxes(
        [_cand()],
        [VlmBoxVerdict(box=1, bbox_correct=None, confidence=None)],
    )
    assert status is None
    assert extra.get('no_verdict') is True
    assert boxes == []


def test_multi_box_mixed_verdicts() -> None:
    cands = [_cand(bbox=(0.0, 0.0, 0.2, 0.2)), _cand(bbox=(0.5, 0.5, 0.8, 0.8))]
    verdicts = [
        VlmBoxVerdict(box=1, bbox_correct=True, confidence='high'),
        VlmBoxVerdict(box=2, bbox_correct=False, confidence='low'),
    ]
    boxes, status, _extra = verdicts_to_boxes(cands, verdicts)
    assert status == RegionStatus.DETECTED
    assert boxes[0].box_id == new_box_placeholder(0)
    assert boxes[0].state == 'accepted'
    assert boxes[1].box_id == new_box_placeholder(1)
    assert boxes[1].state == 'rejected'


def test_stored_pending_verification_box_ids_preserved() -> None:
    cand = _cand(box_id='b3')
    boxes, status, _extra = verdicts_to_boxes(
        [cand],
        [VlmBoxVerdict(box=1, bbox_correct=True, confidence='high')],
    )
    assert status == RegionStatus.DETECTED
    assert boxes[0].box_id == 'b3'


def test_box_id_present_on_every_verify_result() -> None:
    """Every entry in the returned list names its own box_id (Cropwright
    C3/Q15: box_id on per-box verify results)."""
    cands = [_cand(bbox=(0.0, 0.0, 0.2, 0.2)), _cand(bbox=(0.5, 0.5, 0.8, 0.8))]
    verdicts = [
        VlmBoxVerdict(box=1, bbox_correct=True, confidence='high'),
        VlmBoxVerdict(box=2, bbox_correct=None, confidence=None),
    ]
    boxes, _status, _extra = verdicts_to_boxes(cands, verdicts, force_resolve=True)
    assert all(b.box_id for b in boxes)
