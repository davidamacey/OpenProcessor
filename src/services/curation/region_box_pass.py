"""The stored box-list write for one detection pass (W8 B1/M1, shared W5).

One function, :func:`box_pass_update`, decides what a pass's resolved boxes
become on top of the item's CURRENT stored doc: which stored boxes survive,
the real ids minted against the live ``region_box_seq`` high-water mark, the
derived item status, and the box-list fields (``boxes_write_fields``). The
worker's bulk writer calls it inside the OCC merge against the freshly read
doc; ``POST /region_profiles/test`` calls it against the stored item to
preview exactly that write, so the two cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from src.config.region_fields import RegionFields, get_region_fields
from src.services.curation.region_boxes import (
    RegionBox,
    boxes_write_fields,
    derive_status,
    finalize_box_ids,
    is_human_owned,
    merge_boxes_for_write,
    read_boxes,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.config.region_state import RegionStatus


@dataclass(frozen=True)
class BoxPassResult:
    #: The merged list BEFORE placeholder ids became real ones (the
    #: embedding step aligns its entries by position with ``finalized``).
    merged: list[RegionBox]
    finalized: list[RegionBox]
    #: ``{F.status, F.boxes, F.count, ...}``: the fields to write.
    update: dict[str, Any]


def box_pass_update(
    current: dict[str, Any],
    pending: Sequence[RegionBox],
    *,
    reverify: bool,
    merge_machine_boxes: bool,
    baseline: Sequence[RegionBox] | None,
    status: RegionStatus | None,
    empty_status: RegionStatus | None,
    F: RegionFields | None = None,
) -> BoxPassResult:
    """Resolve ``pending`` against the stored doc ``current``.

    ``reverify``: the pass re-verified stored ``proposed`` boxes; every other
    stored sibling survives by id, and ``baseline`` (the snapshot the pass
    read before its VLM call) detects a box a human moved or deleted meanwhile.
    ``merge_machine_boxes``: a fresh detection pass keeps only the boxes a
    human owns and replaces every machine-sourced one. Neither: ``pending``
    is the whole list. ``status`` (when not ``None``) overrides the derived
    item status; otherwise it is derived from the final list, with
    ``empty_status`` for an empty one; a pass with neither is a caller bug.

    The fresh-detection rule is the W8c M1 fix: a new answer to "where are
    the regions?" is not a partial update, so a stale MACHINE-sourced box (a
    prior pass's sanity-gate reject, say) must not accumulate and keep
    overriding the derived status; boxes a human created or acted on
    (``is_human_owned``) are never dropped.
    """
    if status is None and empty_status is None:
        msg = 'a box pass needs a status or an empty_status'
        raise ValueError(msg)
    F = F or get_region_fields()
    stored_now = read_boxes(current, F)
    if reverify:
        merged = merge_boxes_for_write(stored_now, pending, baseline=baseline)
    elif merge_machine_boxes:
        keep = [b for b in stored_now if is_human_owned(b)]
        merged = [*keep, *pending]
    else:
        merged = list(pending)
    finalized = finalize_box_ids(merged, existing=stored_now, seq=int(current.get(F.box_seq) or 0))
    resolved_status = (
        status
        if status is not None
        else derive_status(finalized, empty_status=cast('RegionStatus', empty_status))
    )
    update: dict[str, Any] = {F.status: resolved_status}
    update.update(boxes_write_fields(finalized, current_src=current, F=F))
    return BoxPassResult(merged=merged, finalized=finalized, update=update)


__all__ = ['BoxPassResult', 'box_pass_update']
