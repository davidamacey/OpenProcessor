"""Per-box VLM region verification (``POST /vlm/verify_regions``).

One region crop is sent to the VLM per stored box that is still open to a
machine verdict; each reply is written back onto *its* box (``bbox_correct``
/ ``confidence`` / ``state``) and the item status is re-derived. Pure, no
I/O: the route reads, calls the VLM and hands the verdicts to
:func:`verify_regions_update` inside its OCC merge.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.clients.occ_locks import is_locked_box
from src.config.region_fields import RegionFields, get_region_fields
from src.config.region_rejection import REJECT_REASON_VERIFIER
from src.config.region_state import RegionStatus
from src.services.curation.region_box_edits import same_box
from src.services.curation.region_boxes import (
    RegionBox,
    boxes_write_fields,
    derive_status,
    read_boxes,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_VERIFIABLE_STATES = ('proposed', 'accepted')


@dataclass(frozen=True)
class BoxVerdict:
    """The VLM's verdict on one box, with the geometry it was shown (the
    merge drops a verdict whose box moved while the call was in flight)."""

    box_id: str
    bbox_norm: tuple[float, float, float, float]
    is_region: bool
    confidence: str | None
    reason: str | None


def verifiable_boxes(src: dict[str, Any], F: RegionFields | None = None) -> list[RegionBox]:
    """The stored boxes the verifier may judge: ``proposed`` or
    ``accepted``, and never a box a human or an import owns (a locked box is
    never sent for a machine verdict)."""
    return [b for b in read_boxes(src, F) if b.state in _VERIFIABLE_STATES and not is_locked_box(b)]


def verify_regions_update(
    current: dict[str, Any],
    verdicts: Sequence[BoxVerdict],
    *,
    now: str,
    pack_stamp: str | None,
    vlm_stamp: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The OCC-merge update for ``verdicts`` against the live ``current``
    doc, or ``{}`` when nothing applies.

    A human verdict on the item that landed while the VLM calls were in
    flight wins outright, and a verdict is dropped when its box is gone,
    has since been locked, or no longer has the geometry it was verified
    at. ``is_region`` accepts the box; otherwise it is rejected with the
    verifier reason. ``region_verified`` follows whether any box is now
    accepted.
    """
    F = get_region_fields()
    if current.get(F.verifier) == 'human' or current.get(F.label_source) == 'human':
        return {}
    by_id = {v.box_id: v for v in verdicts}
    boxes: list[RegionBox] = []
    applied: list[BoxVerdict] = []
    for box in read_boxes(current, F):
        verdict = by_id.get(box.box_id)
        if (
            verdict is None
            or box.state not in _VERIFIABLE_STATES
            or is_locked_box(box)
            or not same_box(box.bbox_norm, verdict.bbox_norm)
        ):
            boxes.append(box)
            continue
        applied.append(verdict)
        boxes.append(
            dataclasses.replace(
                box,
                state='accepted' if verdict.is_region else 'rejected',
                rejection_reason=None if verdict.is_region else REJECT_REASON_VERIFIER,
                bbox_correct=verdict.is_region,
                confidence=verdict.confidence,
            )
        )
    if not applied:
        return {}
    rejected = [v for v in applied if not v.is_region]
    update = dict(boxes_write_fields(boxes, current_src=current, F=F))
    update.update(
        {
            F.status: derive_status(boxes, empty_status=RegionStatus.NO_REGION_BOX).value,
            F.verified: any(b.state == 'accepted' for b in boxes),
            F.reason: (rejected or applied)[0].reason,
            'updated_at': now,
        }
    )
    if pack_stamp is not None:
        update['vlm_prompt_pack'] = pack_stamp
    if vlm_stamp:
        # Which endpoint/model answered (W9 provenance).
        update.update(vlm_stamp)
    return update


__all__ = ['BoxVerdict', 'verifiable_boxes', 'verify_regions_update']
