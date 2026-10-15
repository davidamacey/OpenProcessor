"""What one imported image will hold: the item-label plan against what the
image already stores, the region parents, the region attachments and the
proposal merge.

Planning reads the image and the index (existing items, the detector) but
writes nothing; :mod:`~src.services.curation.dataset_import.chunk` applies the
plan.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from src.domain.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.curation.class_label import ItemLabel
from src.services.curation.dataset_import import reconcile, region_labels
from src.services.curation.dataset_import.item_labels import (
    ItemPlan,
    import_stamp,
    plan_item_labels,
)
from src.services.curation.dataset_import.proposals import (
    detection_bbox_norm,
    parent_detections,
    plan_proposals,
    run_detector,
)
from src.services.curation.dataset_import.regions import ParentCandidate, attach_region_boxes
from src.services.curation.dataset_import.scan import LabelBox
from src.services.curation.item_doc import DetectedItem
from src.services.curation.wire import item_source_excludes
from src.services.detection.geometry import crop_id as make_crop_id, stored_bbox_norm


if TYPE_CHECKING:
    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.mapping import MapTarget
    from src.services.curation.dataset_import.scan import ScanEntry
    from src.services.curation.ingest_index import ImageContext
    from src.services.curation.proposal_merge import MergePlan


@dataclasses.dataclass
class SplitBoxes:
    items: list[tuple[LabelBox, MapTarget]]
    regions: list[LabelBox]
    skipped: int


@dataclasses.dataclass
class ImagePlan:
    existing: list[dict[str, Any]]
    plans: list[ItemPlan]
    dropped: list[dict[str, Any]]
    parent_cands: list[ParentCandidate]
    detect_parents: list[DetectedItem]
    attach: Any
    merge_plan: MergePlan | None
    created_detections: list[DetectedItem]


def split_boxes(ctx: ImportContext, entry: ScanEntry) -> SplitBoxes:
    out = SplitBoxes([], [], 0)
    for box in entry.boxes:
        target = ctx.resolved.targets.get(box.dataset_class)
        if target is None or target.kind == 'skip':
            out.skipped += 1
        elif target.kind == 'item':
            out.items.append((box, target))
        else:
            out.regions.append(box)
    return out


async def _items_for_image(ctx: ImportContext, image_id: str) -> list[dict[str, Any]]:
    resp = await ctx.opensearch.search(
        index=ctx.items_index,
        body={
            'size': 1000,
            'query': {'term': {'image_id': image_id}},
            '_source': {'excludes': item_source_excludes()},
        },
    )
    return [h['_source'] for h in (resp.get('hits') or {}).get('hits') or []]


def _apply_prior_actions(plans: list[ItemPlan], prior: dict[str, Any] | None) -> None:
    """A resumed image keeps the actions its write-ahead row recorded: the
    docs now exist, so a fresh plan would call ``created`` items ``noop``."""
    if not prior:
        return
    recorded = {i['crop_id']: i for i in prior.get('items') or []}
    for plan in plans:
        item = recorded.get(plan.crop_id)
        if item is None:
            continue
        plan.action = item['action']
        plan.snapshot_index = item.get('snapshot_index')
        plan.holdout_prior = item.get('holdout_prior')


def _docs_after_import(
    ctx: ImportContext, existing: list[dict[str, Any]], plans: list[ItemPlan]
) -> list[dict[str, Any]]:
    """The image's items as they will be once this entry's labels are
    written: proposals are merged against the labels, not around them."""
    claimed = {p.crop_id for p in plans}
    docs = [d for d in existing if d.get('crop_id') not in claimed]
    validated = ctx.options.label_trust == 'validated'
    for plan in plans:
        base = dict(plan.existing or {})
        if plan.action in ('created', 'updated'):
            base.update(
                {
                    'crop_id': plan.crop_id,
                    'bbox_norm': list(plan.box.bbox_norm),
                    'class_name': plan.target.class_name,
                    'class_source': LABEL_IMPORT_CLASS_SOURCE,
                    'class_validated': validated,
                }
            )
        docs.append(base)
    return docs


def _parent_candidates(
    ctx: ImportContext,
    plans: list[ItemPlan],
    boxes: SplitBoxes,
    detections: list[DetectedItem],
    img: ImageContext,
    parent_classes: frozenset[str],
) -> tuple[list[ParentCandidate], list[DetectedItem]]:
    """Parents for region labels: the labeled items, or the detector's boxes."""
    region_targets = bool(boxes.regions) or any(
        t.kind == 'region' for t in ctx.resolved.targets.values()
    )
    parent_cands: list[ParentCandidate] = []
    detect_parents: list[DetectedItem] = []
    if region_targets and ctx.parents == 'labels':
        parent_cands = [
            ParentCandidate(p.crop_id, p.box.bbox_norm, p.target.class_name)
            for p in plans
            if not parent_classes
            or (p.target.class_name or '').lower() in {c.lower() for c in parent_classes}
        ]
    elif region_targets and ctx.parents == 'detect':
        detect_parents = parent_detections(ctx, detections)
        for d in detect_parents:
            nb = detection_bbox_norm(d, img.width, img.height)
            parent_cands.append(
                ParentCandidate(make_crop_id(img.image_id, list(nb)), nb, d.class_name)  # type: ignore[arg-type]
            )
    return parent_cands, detect_parents


async def plan_image(
    ctx: ImportContext,
    entry: ScanEntry,
    boxes: SplitBoxes,
    state: str,
    img: ImageContext,
    prior: dict[str, Any] | None,
) -> ImagePlan:
    """Plan one ingested image's import; nothing is written."""
    existing = await _items_for_image(ctx, img.image_id)
    stored_boxes = [
        (LabelBox(b.dataset_class, stored_bbox_norm(b.bbox_norm, img.width, img.height)), t)
        for b, t in boxes.items
    ]
    plans = plan_item_labels(existing, stored_boxes, image_id=img.image_id)
    _apply_prior_actions(plans, prior)
    dropped = await reconcile.dropped_entries(
        ctx, existing, entry, plans, image_id=img.image_id, prior=prior
    )

    detections: list[DetectedItem] = []
    if ctx.uses_detector:
        detections = await run_detector(ctx, img.require_pil(), img.image_path)

    parent_classes = frozenset(ctx.profile.parent_classes) if ctx.profile else frozenset()
    parent_cands, detect_parents = _parent_candidates(
        ctx, plans, boxes, detections, img, parent_classes
    )
    attach = attach_region_boxes(
        parent_cands,
        boxes.regions,
        containment_threshold=ctx.options.region_containment,
        parent_classes=parent_classes or None,
    )

    # Proposals (propose): merge against everything the image will hold.
    merge_plan: MergePlan | None = None
    created_detections: list[DetectedItem] = []
    if ctx.options.processing == 'propose':
        merge_plan, created_detections = plan_proposals(
            _docs_after_import(ctx, existing, plans),
            detections,
            width=img.width,
            height=img.height,
            on_negative_frame=state == 'negative',
        )
    return ImagePlan(
        existing=existing,
        plans=plans,
        dropped=dropped,
        parent_cands=parent_cands,
        detect_parents=detect_parents,
        attach=attach,
        merge_plan=merge_plan,
        created_detections=created_detections,
    )


def labeled_item(
    ctx: ImportContext, entry: ScanEntry, plan: ItemPlan, width: int, height: int, now: str
) -> DetectedItem:
    x1, y1, x2, y2 = plan.box.bbox_norm
    target = plan.target
    return DetectedItem(
        bbox_pixel=(x1 * width, y1 * height, x2 * width, y2 * height),
        score=1.0,
        class_id=target.class_id,
        class_name=target.class_name,
        class_source=LABEL_IMPORT_CLASS_SOURCE,
        label=ItemLabel.imported(
            import_id=ctx.import_id,
            class_id=target.class_id,  # type: ignore[arg-type]
            class_name=target.class_name,  # type: ignore[arg-type]
            trust=ctx.options.label_trust,
            now=now,
            writer=ctx.writer,
        ),
        extra_fields=import_stamp(ctx, entry, {}, now),
    )


def tag_proposed(ctx: ImportContext, item: DetectedItem, state: str) -> DetectedItem:
    item.extra_fields = {
        **item.extra_fields,
        'proposed_by_import': ctx.import_id,
        **({'on_negative_frame': True} if state == 'negative' else {}),
    }
    return item


@dataclasses.dataclass
class NewItems:
    labeled: list[DetectedItem]
    standalone: list[DetectedItem]
    parents: list[DetectedItem]
    machine: list[DetectedItem]

    def all(self) -> list[DetectedItem]:
        return [*self.labeled, *self.standalone, *self.parents, *self.machine]


def new_items(
    ctx: ImportContext,
    entry: ScanEntry,
    plan: ImagePlan,
    img: ImageContext,
    state: str,
    now: str,
) -> NewItems:
    detect_parents = plan.detect_parents
    return NewItems(
        labeled=[
            labeled_item(ctx, entry, p, img.width, img.height, now)
            for p in plan.plans
            if p.action == 'created'
        ],
        standalone=[
            region_labels.standalone_item(
                ctx, entry, box, width=img.width, height=img.height, now=now
            )
            for box in plan.attach.standalone
        ],
        parents=[tag_proposed(ctx, d, state) for d in detect_parents],
        machine=[
            tag_proposed(ctx, d, state)
            for d in plan.created_detections
            if not any(d is p for p in detect_parents)
        ],
    )
