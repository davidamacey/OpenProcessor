"""Region-class labels (W10.7): write a dataset's region boxes onto their
parent items, make standalone items for boxes with no parent, and record the
reviewed-negative "no region here" verdict.

One OCC write per parent. The list written is the parent's human-owned boxes
(kept) plus the dataset's boxes; unlocked machine boxes are dropped because
the dataset's set is complete. A parent a human already owns (class, a box,
or a verdict) is never written: the conflict is reported instead.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_update_one
from src.config.region_source import CANDIDATE_IMPORT
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.dataset_import.item_labels import human_owned, import_stamp
from src.services.curation.edit_history import EDIT_HISTORY_FIELD, EditKind, record_edit
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.curation.item_doc import DetectedItem
from src.services.curation.region_box_edits import same_box
from src.services.curation.region_boxes import (
    RegionBox,
    boxes_write_fields,
    derive_status,
    is_human_owned,
    next_box_id,
    read_boxes,
)


if TYPE_CHECKING:
    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.scan import LabelBox, ScanEntry

logger = get_logger(__name__)


@dataclasses.dataclass
class ParentWrite:
    outcome: str
    """``written`` | ``noop`` (this import already wrote it) | ``locked``."""
    boxes: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    accepted_bbox: tuple[float, float, float, float] | None = None


def _validated(ctx: ImportContext) -> bool:
    return ctx.options.label_trust == 'validated'


def new_import_box(
    ctx: ImportContext, bbox: tuple[float, float, float, float], box_id: str, now: str
) -> RegionBox:
    return RegionBox(
        box_id=box_id,
        bbox_norm=bbox,
        state='accepted' if _validated(ctx) else 'proposed',
        score=1.0,
        detector='import',
        detector_version=ctx.import_id,
        source=CANDIDATE_IMPORT,
        detected_at=now,
    )


def region_state_fields(
    ctx: ImportContext, boxes: list[RegionBox], current: dict[str, Any], now: str
) -> dict[str, Any]:
    """Every region field an import writes for a final box list."""
    F = ctx.region_fields
    fields = boxes_write_fields(boxes, current_src=current, F=F, set_complete=True)
    status = derive_status(boxes, empty_status=RegionStatus.NO_REGION_VISIBLE)
    if ctx.options.processing == 'propose' and boxes:
        # The region worker proposes what the dataset missed around the
        # locked boxes (W10.7); the set stays validated until it adds one.
        status = RegionStatus.PENDING_DETECTION
    fields[F.status] = status.value
    fields[F.validated] = _validated(ctx)
    fields[F.label_source] = 'import'
    fields[F.verifier] = 'import'
    fields[F.verifier_version] = ctx.import_id
    fields[F.verified_at] = now
    if ctx.profile is not None:
        fields[F.profile] = ctx.profile.name
        fields[F.profile_revision] = ctx.profile.revision
    return fields


def region_human_locked(current: dict[str, Any], boxes: list[RegionBox], F: Any) -> bool:
    return (
        human_owned(current)
        or any(is_human_owned(b) for b in boxes)
        or (bool(current.get(F.validated)) and current.get(F.verifier) == 'human')
    )


async def write_parent_regions(
    ctx: ImportContext,
    entry: ScanEntry,
    parent_key: str,
    label_boxes: list[LabelBox],
    now: str,
) -> ParentWrite:
    """Write ``label_boxes`` (possibly none: a reviewed negative) onto one
    parent. See the module docstring for what is kept and dropped."""
    F = ctx.region_fields
    result = ParentWrite(outcome='written')

    def merger(current: dict[str, Any]) -> dict[str, Any]:
        stored = read_boxes(current, F)
        if region_human_locked(current, stored, F):
            result.outcome = 'locked'
            return {}
        keep = [b for b in stored if is_human_owned(b)]
        previous = [b for b in stored if b.source == CANDIDATE_IMPORT]
        seq = int(current.get(F.box_seq) or 0)
        written: list[RegionBox] = []
        for lb in label_boxes:
            reuse = next(
                (b for b in previous if same_box(list(b.bbox_norm), list(lb.bbox_norm))), None
            )
            box = reuse or new_import_box(
                ctx, lb.bbox_norm, next_box_id([*stored, *written], seq=seq), now
            )
            written.append(box)
        final = [*keep, *written]
        fields = region_state_fields(ctx, final, current, now)
        fields.update(import_stamp(ctx, entry, current, now))
        already = any(
            e.get('writer') == ctx.writer and e.get('kind') == EditKind.REGION.value
            for e in current.get(EDIT_HISTORY_FIELD) or []
            if isinstance(e, dict)
        )
        if not already:
            fields[EDIT_HISTORY_FIELD] = record_edit(
                current, kind=EditKind.REGION, writer=ctx.writer, now=now
            )
        result.boxes = [
            {'box_id': b.box_id, 'bbox_norm': list(b.bbox_norm), 'state': b.state} for b in written
        ]
        accepted = [b for b in written if b.state == 'accepted']
        result.accepted_bbox = accepted[0].bbox_norm if accepted else None
        return fields

    await occ_update_one(
        ctx.opensearch,
        doc_id=parent_key,
        merger=merger,
        index=ctx.items_index,
        writer_id=ctx.writer,
    )
    return result


def standalone_item(
    ctx: ImportContext,
    entry: ScanEntry,
    box: LabelBox,
    *,
    width: int,
    height: int,
    now: str,
) -> DetectedItem:
    """An item for a region box with no parent: its own bbox, no class, one
    accepted region box equal to itself (W10.7)."""
    bbox = box.bbox_norm
    region_box = new_import_box(ctx, bbox, 'b1', now)
    extra: dict[str, Any] = import_stamp(ctx, entry, {}, now)
    extra.update(region_state_fields(ctx, [region_box], {}, now))
    extra['import_standalone_region'] = True
    return DetectedItem(
        bbox_pixel=(bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height),
        score=1.0,
        class_source=LABEL_IMPORT_CLASS_SOURCE,
        extra_fields=extra,
    )


__all__ = [
    'ParentWrite',
    'new_import_box',
    'region_human_locked',
    'region_state_fields',
    'standalone_item',
    'write_parent_regions',
]
