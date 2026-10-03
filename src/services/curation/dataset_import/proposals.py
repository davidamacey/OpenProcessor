"""Machine proposals during an import (W10.6 step 7, W10.7 ``parents:
detect``): run the primary detector over the decoded image and merge its
boxes into the image's items through
:func:`~src.services.curation.proposal_merge.merge_item_proposals`.

A proposal never overwrites an imported or human label: one that overlaps a
locked item is noted on it (``proposal_chain``), the rest become new
unvalidated machine items tagged ``proposed_by_import``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_update_one
from src.clients.occ_locks import is_locked_item
from src.services.curation.proposal_merge import (
    ExistingItem,
    MergePlan,
    Proposal,
    merge_item_proposals,
)
from src.services.curation.region_scope import in_parent_classes
from src.services.detection.geometry import bbox_norm as bbox_norm_of


if TYPE_CHECKING:
    from PIL import Image

    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.item_doc import DetectedItem


async def run_detector(ctx: ImportContext, pil: Image.Image, image_path: str) -> list[DetectedItem]:
    """The ingest detectors' boxes for one image (primary, plus the optional
    secondary's class override), exactly as ingest runs them."""
    return (await ctx.service.detect_items(pil, image_path=image_path)).items


def detection_bbox_norm(item: DetectedItem, width: int, height: int) -> tuple[float, ...]:
    return tuple(bbox_norm_of(item.bbox_pixel, width, height))


def parent_detections(ctx: ImportContext, detections: list[DetectedItem]) -> list[DetectedItem]:
    """Detections in the profile's parent scope (any class when it names none)."""
    parent_classes = ctx.profile.parent_classes if ctx.profile else ()
    return [
        d
        for d in detections
        if in_parent_classes(parent_classes, class_name=d.class_name, proposal_name=d.proposal_name)
    ]


def to_proposals(detections: list[DetectedItem], width: int, height: int) -> list[Proposal]:
    return [
        Proposal(
            bbox_norm=detection_bbox_norm(d, width, height),  # type: ignore[arg-type]
            class_id=d.class_id,
            class_name=d.class_name or d.proposal_name,
            score=d.score,
        )
        for d in detections
    ]


def existing_items(docs: list[dict[str, Any]]) -> list[ExistingItem]:
    return [
        ExistingItem(
            crop_id=str(d['crop_id']),
            bbox_norm=tuple(d.get('bbox_norm') or (0.0, 0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            class_name=d.get('class_name'),
            locked=is_locked_item(d),
        )
        for d in docs
        if d.get('crop_id')
    ]


def plan_proposals(
    docs: list[dict[str, Any]],
    detections: list[DetectedItem],
    *,
    width: int,
    height: int,
    on_negative_frame: bool,
) -> tuple[MergePlan, list[DetectedItem]]:
    """The merge plan, plus the detections behind ``plan.created`` (same
    order): the items to create."""
    proposals = to_proposals(detections, width, height)
    by_proposal = {id(p): d for p, d in zip(proposals, detections, strict=True)}
    plan = merge_item_proposals(
        existing_items(docs), proposals, on_negative_frame=on_negative_frame
    )
    return plan, [by_proposal[id(p)] for p in plan.created]


async def note_matches(ctx: ImportContext, plan: MergePlan) -> None:
    """Append ``<profile>:match`` to ``proposal_chain`` of every locked item a
    proposal matched."""
    note = f'{ctx.service.profile.name}:match'
    for crop_id, _proposal in plan.merged_into_locked:

        def merger(current: dict[str, Any]) -> dict[str, Any]:
            chain = list(current.get('proposal_chain') or [])
            if note in chain:
                return {}
            return {'proposal_chain': [*chain, note]}

        await occ_update_one(
            ctx.opensearch,
            doc_id=crop_id,
            merger=merger,
            index=ctx.items_index,
            writer_id=ctx.writer,
        )


__all__ = [
    'detection_bbox_norm',
    'existing_items',
    'note_matches',
    'parent_detections',
    'plan_proposals',
    'run_detector',
    'to_proposals',
]
