"""The per-image import flow (W10.6) and its chunk loop.

For each image: load it, plan the item labels against what the image
already holds, write the plan ahead to the ledger, then apply it: item
labels through the single class-label writer, new items through the same
``index_items`` ingest uses, region boxes, the negative-frame state, and
(``propose``) machine proposals. Every write is idempotent, so a chunk the
process died in is simply run again; the write-ahead ledger row is what
keeps ``created`` vs ``updated`` truthful across that re-run.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_update_one
from src.core.logging import get_logger
from src.services.curation.class_label import ItemLabel
from src.services.curation.dataset_import import region_labels
from src.services.curation.dataset_import.image_load import ImageLoadError, load_image_bytes
from src.services.curation.dataset_import.item_labels import (
    ItemPlan,
    apply_item_updates,
    import_stamp,
    plan_item_labels,
    stamp_matched_noops,
)
from src.services.curation.dataset_import.op_export_stems import (
    project_item_crop_boxes,
    resolve_stems,
)
from src.services.curation.dataset_import.proposals import (
    detection_bbox_norm,
    note_matches,
    parent_detections,
    plan_proposals,
    run_detector,
)
from src.services.curation.dataset_import.regions import ParentCandidate, attach_region_boxes
from src.services.curation.dataset_import.report import ImportReport
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.curation.ingest_index import ImageContext
from src.services.curation.item_doc import DetectedItem
from src.services.curation.proposal_merge import count_disagreements
from src.services.detection.geometry import crop_id as make_crop_id


if TYPE_CHECKING:
    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.mapping import MapTarget
    from src.services.curation.dataset_import.scan import LabelBox, ScanEntry
    from src.services.curation.dataset_import.store import ImportStore
    from src.services.curation.proposal_merge import MergePlan

logger = get_logger(__name__)

_FINAL = ('ok', 'failed', 'skipped')
_ITEM_EXCLUDES = ['pe_embedding', 'backbone_embedding', 'region_embedding', 'region_box_embeddings']


@dataclasses.dataclass
class _Boxes:
    items: list[tuple[LabelBox, MapTarget]]
    regions: list[LabelBox]
    skipped: int


def _resolve_stem(ctx: ImportContext, entry: ScanEntry) -> ScanEntry:
    """An OpenProcessor-export entry whose stem names a doc this project
    already holds attaches to that doc's ORIGINAL image (the exported copy is
    resized, and coordinates are resize-invariant); an ``item_crop`` entry's
    boxes are first projected from the crop frame into the source frame."""
    res = ctx.stem_resolutions.get(entry.source_stem)
    if res is None:
        return entry
    boxes = project_item_crop_boxes(entry.boxes, res) if res.kind == 'item_crop' else entry.boxes
    return dataclasses.replace(entry, abs_image_path=Path(res.image_path), boxes=list(boxes))


def _split_boxes(ctx: ImportContext, entry: ScanEntry) -> _Boxes:
    out = _Boxes([], [], 0)
    for box in entry.boxes:
        target = ctx.resolved.targets.get(box.dataset_class)
        if target is None or target.kind == 'skip':
            out.skipped += 1
        elif target.kind == 'item':
            out.items.append((box, target))
        else:
            out.regions.append(box)
    return out


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


async def _items_for_image(ctx: ImportContext, image_id: str) -> list[dict[str, Any]]:
    resp = await ctx.opensearch.search(
        index=ctx.items_index,
        body={
            'size': 1000,
            'query': {'term': {'image_id': image_id}},
            '_source': {'excludes': _ITEM_EXCLUDES},
        },
    )
    return [h['_source'] for h in (resp.get('hits') or {}).get('hits') or []]


def _image_stamp(
    ctx: ImportContext, entry: ScanEntry, state: str, current: dict[str, Any]
) -> dict[str, Any]:
    ids = list(current.get('import_ids') or [])
    if ctx.import_id not in ids:
        ids.append(ctx.import_id)
    stamp: dict[str, Any] = {
        'import_ids': ids,
        'import_source_stem': entry.source_stem,
        'import_label_state': state,
    }
    if state == 'negative':
        stamp['negative_for'] = ctx.negative_for
    if entry.split:
        stamp['dataset_split'] = entry.split
    if entry.stratum:
        stamp['import_stratum'] = entry.stratum
    if entry.hard_negative:
        stamp['import_hard_negative'] = True
    return stamp


async def _stamp_existing_image(
    ctx: ImportContext, image_id: str, entry: ScanEntry, state: str
) -> None:
    def merger(current: dict[str, Any]) -> dict[str, Any]:
        return _image_stamp(ctx, entry, state, current)

    await occ_update_one(
        ctx.opensearch,
        doc_id=image_id,
        merger=merger,
        index=ctx.images_index,
        writer_id=ctx.writer,
    )


def _labeled_item(
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


async def _import_one(
    ctx: ImportContext,
    store: ImportStore,
    chunk: int,
    entry: ScanEntry,
    prior: dict[str, Any] | None,
) -> ImportReport:
    report = ImportReport()
    base = {
        'source_stem': entry.source_stem,
        'rel_path': entry.rel_path,
        'split': entry.split,
        'label_state': entry.label_state,
    }
    entry = _resolve_stem(ctx, entry)
    boxes = _split_boxes(ctx, entry)
    state = entry.label_state
    if state == 'labeled' and not boxes.items and not boxes.regions:
        state = 'unlabeled'  # every box was skipped: says nothing about the frame

    try:
        data, stored = load_image_bytes(ctx, entry)
    except ImageLoadError as exc:
        return _failed(store, chunk, base, exc.kind, str(exc), report)

    known = ctx.seen_images.get(_hash_of(data))
    img = await ctx.service.ingest_image(
        data,
        str(stored),
        ctx.options.source_tag,
        ingest_run_id=ctx.import_id,
        adopt_existing=True,
        known_existing=known,
    )
    if not isinstance(img, ImageContext):
        return _failed(
            store, chunk, base, img.error_kind or 'image_unreadable', img.error or '', report
        )
    if entry.declared_size is not None:
        actual = (img.width, img.height)
        if actual != entry.declared_size:
            return _skipped(store, chunk, base, 'image_size_mismatch', report)
    created_image = prior.get('image_created') if prior else img.created
    img.image_extra = _image_stamp(ctx, entry, state, {})
    now = _now()

    existing = await _items_for_image(ctx, img.image_id)
    plans = plan_item_labels(existing, boxes.items, image_id=img.image_id)
    _apply_prior_actions(plans, prior)

    detections: list[DetectedItem] = []
    if ctx.uses_detector:
        detections = await run_detector(ctx, img.pil, img.image_path)

    # Parents for region labels, and the standalone items for boxes with none.
    region_targets = bool(boxes.regions) or any(
        t.kind == 'region' for t in ctx.resolved.targets.values()
    )
    parent_classes = frozenset(ctx.profile.parent_classes) if ctx.profile else frozenset()
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

    new_items = [
        _labeled_item(ctx, entry, plan, img.width, img.height, now)
        for plan in plans
        if plan.action == 'created'
    ]
    standalone_items = [
        region_labels.standalone_item(ctx, entry, box, width=img.width, height=img.height, now=now)
        for box in attach.standalone
    ]
    parent_items = [_tag_proposed(ctx, d, state) for d in detect_parents]
    machine_new = [
        _tag_proposed(ctx, d, state)
        for d in created_detections
        if not any(d is p for p in detect_parents)
    ]
    parent_keys = {c.key for c in parent_cands}

    row: dict[str, Any] = {
        **base,
        'status': 'pending',
        'image_id': img.image_id,
        'image_path': img.image_path,
        'image_created': bool(created_image),
        'items': [
            {
                'crop_id': p.crop_id,
                'action': p.action,
                'snapshot_index': p.snapshot_index,
                'holdout_prior': p.holdout_prior,
            }
            for p in plans
        ],
    }
    store.append_ledger(chunk, row)

    # ---- apply
    all_new = [*new_items, *standalone_items, *parent_items, *machine_new]
    seed = ctx.options.processing == 'propose'
    outcome = await ctx.service.index_items(img, all_new, seed_region=seed)
    if outcome.result.status == 'failed':
        return _failed(
            store,
            chunk,
            base,
            outcome.result.error_kind or 'bulk_index',
            outcome.result.error or '',
            report,
        )
    ctx.seen_images[img.imohash] = {'image_id': img.image_id, 'image_path': img.image_path}
    if not img.created:
        await _stamp_existing_image(ctx, img.image_id, entry, state)
    await stamp_matched_noops(ctx, plans)
    updates = await apply_item_updates(ctx, entry, plans, now)

    ledger_items = list(row['items'])
    cids = outcome.crop_ids
    n_labeled, n_standalone, n_parent = len(new_items), len(standalone_items), len(parent_items)
    standalone_ids = cids[n_labeled : n_labeled + n_standalone]
    parent_ids = cids[n_labeled + n_standalone : n_labeled + n_standalone + n_parent]
    machine_ids = cids[n_labeled + n_standalone + n_parent :]
    ledger_items.extend({'crop_id': cid, 'action': 'standalone'} for cid in standalone_ids)
    ledger_items.extend({'crop_id': cid, 'action': 'parent'} for cid in parent_ids)
    ledger_items.extend({'crop_id': cid, 'action': 'proposal'} for cid in machine_ids)

    ledger_boxes: list[dict[str, Any]] = []
    await _write_regions(ctx, entry, attach, parent_keys, state, now, report, ledger_boxes)

    if merge_plan is not None:
        await note_matches(ctx, merge_plan)
        report.proposals_merged += len(merge_plan.merged_into_locked)
        report.proposals_created += len(machine_ids)
        for dis in merge_plan.disagreements:
            report.disagreement_samples.append(
                {'rel_path': entry.rel_path, **dataclasses.asdict(dis)}
            )
        for key, value in count_disagreements(merge_plan.disagreements).items():
            report.disagreement_counts[key] += value

    _count_plans(report, plans, updates, len(new_items))
    report.parents_detected += n_parent
    report.standalone_regions += n_standalone
    ledger_boxes.extend({'crop_id': cid, 'box_id': 'b1'} for cid in standalone_ids)
    if state == 'negative':
        report.negatives += 1
    elif state == 'unlabeled':
        report.unlabeled += 1
    report.images_created += 1 if created_image else 0
    report.images_reused += 0 if created_image else 1
    if ctx.freeze_test and entry.split == 'test':
        report.holdout_frozen += (
            len(new_items)
            + len(standalone_items)
            + len(parent_items)
            + sum(1 for p in plans if p.action == 'updated')
        )

    final = {
        **row,
        'status': 'ok',
        'items': ledger_items,
        'boxes': ledger_boxes,
        'report': report.to_dict(),
    }
    store.append_ledger(chunk, final)
    return report


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


def _hash_of(data: bytes) -> str:
    from src.services.curation.ingest import _imohash_bytes

    return _imohash_bytes(data)


def _tag_proposed(ctx: ImportContext, item: DetectedItem, state: str) -> DetectedItem:
    item.extra_fields = {
        **item.extra_fields,
        'proposed_by_import': ctx.import_id,
        **({'on_negative_frame': True} if state == 'negative' else {}),
    }
    return item


def _count_plans(
    report: ImportReport, plans: list[ItemPlan], updates: dict[str, str], created: int
) -> None:
    report.items_created += created
    report.labels_written += created
    for plan in plans:
        if plan.action == 'locked_conflict' or updates.get(plan.crop_id) == 'locked_conflict':
            report.label_conflicts_locked += 1
        elif plan.action == 'updated':
            report.items_updated += 1
            report.labels_written += 1
        elif plan.action == 'noop':
            report.items_noop += 1


async def _write_regions(
    ctx: ImportContext,
    entry: ScanEntry,
    attach: Any,
    parent_keys: set[str],
    state: str,
    now: str,
    report: ImportReport,
    ledger_boxes: list[dict[str, Any]],
) -> None:
    by_parent: dict[str, list[LabelBox]] = {}
    for a in attach.attachments:
        by_parent.setdefault(a.parent_key, []).append(a.box)
    negative_ok = ctx.options.region_negatives and state in ('labeled', 'negative')
    targets = list(by_parent)
    if negative_ok:
        targets += [k for k in sorted(parent_keys) if k not in by_parent]
    for key in targets:
        write = await region_labels.write_parent_regions(
            ctx, entry, key, by_parent.get(key, []), now
        )
        if write.outcome == 'locked':
            report.label_conflicts_locked += 1
            continue
        report.boxes_written += len(write.boxes)
        ledger_boxes.extend({'crop_id': key, **b} for b in write.boxes)


def _failed(
    store: ImportStore,
    chunk: int,
    base: dict[str, Any],
    kind: str,
    message: str,
    report: ImportReport,
) -> ImportReport:
    report.images_failed += 1
    store.append_ledger(
        chunk,
        {
            **base,
            'status': 'failed',
            'error_kind': kind,
            'error': message[:300],
            'report': report.to_dict(),
        },
    )
    return report


def _skipped(
    store: ImportStore, chunk: int, base: dict[str, Any], kind: str, report: ImportReport
) -> ImportReport:
    report.images_skipped += 1
    store.append_ledger(
        chunk, {**base, 'status': 'skipped', 'error_kind': kind, 'report': report.to_dict()}
    )
    return report


async def import_chunk(
    ctx: ImportContext, store: ImportStore, chunk: int, entries: list[ScanEntry]
) -> ImportReport:
    """Import one chunk; returns its report. Images a previous (crashed) run
    already finished are not redone."""
    prior_rows = store.chunk_rows(chunk)
    total = ImportReport()
    if ctx.source_format == 'openprocessor_export':
        ctx.stem_resolutions = await resolve_stems(
            ctx.opensearch, entries, images_index=ctx.images_index, items_index=ctx.items_index
        )
    for entry in entries:
        prior = prior_rows.get(entry.rel_path)
        if prior and prior.get('status') in _FINAL:
            total.add(ImportReport.from_dict(prior.get('report') or {}))
            continue
        total.add(await _import_one(ctx, store, chunk, entry, prior))
    for index in (ctx.images_index, ctx.items_index):
        await ctx.opensearch.indices.refresh(index=index)
    return total


__all__ = ['import_chunk']
