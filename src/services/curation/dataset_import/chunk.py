"""The per-image import flow and its chunk loop.

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
from typing import TYPE_CHECKING, Any, Literal

from src.clients.occ import occ_update_one
from src.core.logging import get_logger
from src.services.curation.dataset_import import reconcile, region_labels
from src.services.curation.dataset_import.image_load import ImageLoadError, load_image_bytes
from src.services.curation.dataset_import.image_plan import (
    ImagePlan,
    NewItems,
    new_items,
    plan_image,
    split_boxes,
)
from src.services.curation.dataset_import.item_labels import (
    ItemPlan,
    apply_item_updates,
    stamp_matched_noops,
)
from src.services.curation.dataset_import.op_export_stems import (
    project_item_crop_boxes,
    resolve_stems,
)
from src.services.curation.dataset_import.proposals import note_matches
from src.services.curation.dataset_import.report import ImportReport
from src.services.curation.ingest_index import ImageContext
from src.services.curation.proposal_merge import count_disagreements
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.scan import LabelBox, ScanEntry
    from src.services.curation.dataset_import.store import ImportStore
    from src.services.curation.proposal_merge import MergePlan

logger = get_logger(__name__)

_FINAL = ('ok', 'failed', 'skipped')


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


@dataclasses.dataclass
class _Rejected:
    """An image that could not be imported: why, and whether it counts as failed or skipped."""

    outcome: Literal['failed', 'skipped']
    kind: str
    message: str = ''


async def _ingest_entry_image(ctx: ImportContext, entry: ScanEntry) -> ImageContext | _Rejected:
    """Load and ingest the entry's image; a load error, an unreadable image or a
    declared size that disagrees with the pixels rejects the entry."""
    try:
        data, stored = load_image_bytes(ctx, entry)
    except ImageLoadError as exc:
        return _Rejected('failed', exc.kind, str(exc))

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
        return _Rejected('failed', img.error_kind or 'image_unreadable', img.error or '')
    if entry.declared_size is not None:
        actual = (img.width, img.height)
        if actual != entry.declared_size:
            return _Rejected('skipped', 'image_size_mismatch')
    return img


def _reject(
    store: ImportStore, chunk: int, base: dict[str, Any], rejected: _Rejected, report: ImportReport
) -> ImportReport:
    if rejected.outcome == 'skipped':
        return _skipped(store, chunk, base, rejected.kind, report)
    return _failed(store, chunk, base, rejected.kind, rejected.message, report)


def _pending_row(
    base: dict[str, Any], img: ImageContext, created_image: Any, plan: ImagePlan
) -> dict[str, Any]:
    """The write-ahead ledger row: what this image is about to do."""
    return {
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
            for p in plan.plans
        ]
        + plan.dropped,
    }


def _tally_proposals(
    report: ImportReport, entry: ScanEntry, merge_plan: MergePlan, created: int
) -> None:
    report.proposals_merged += len(merge_plan.merged_into_locked)
    report.proposals_created += created
    for dis in merge_plan.disagreements:
        report.disagreement_samples.append({'rel_path': entry.rel_path, **dataclasses.asdict(dis)})
    for key, value in count_disagreements(merge_plan.disagreements).items():
        report.disagreement_counts[key] += value


def _tally_image(
    report: ImportReport,
    ctx: ImportContext,
    entry: ScanEntry,
    state: str,
    created_image: Any,
    plan: ImagePlan,
    new: NewItems,
) -> None:
    """The per-image report counters that do not depend on the apply results."""
    if state == 'negative':
        report.negatives += 1
    elif state == 'unlabeled':
        report.unlabeled += 1
    report.images_created += 1 if created_image else 0
    report.images_reused += 0 if created_image else 1
    if ctx.freeze_test and entry.split == 'test':
        report.holdout_frozen += (
            len(new.labeled)
            + len(new.standalone)
            + len(new.parents)
            + sum(1 for p in plan.plans if p.action == 'updated')
        )


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
        'froze_test': bool(ctx.freeze_test and entry.split == 'test'),
    }
    entry = _resolve_stem(ctx, entry)
    boxes = split_boxes(ctx, entry)
    state = entry.label_state
    if state == 'labeled' and not boxes.items and not boxes.regions:
        state = 'unlabeled'  # every box was skipped: says nothing about the frame

    img = await _ingest_entry_image(ctx, entry)
    if isinstance(img, _Rejected):
        return _reject(store, chunk, base, img, report)
    created_image = prior.get('image_created') if prior else img.created
    img.image_extra = _image_stamp(ctx, entry, state, {})
    now = now_iso()

    plan = await plan_image(ctx, entry, boxes, state, img, prior)
    plans = plan.plans
    new = new_items(ctx, entry, plan, img, state, now)
    parent_keys = {c.key for c in plan.parent_cands}

    row = _pending_row(base, img, created_image, plan)
    store.append_ledger(chunk, row)

    # ---- apply
    seed = ctx.options.processing == 'propose'
    outcome = await ctx.service.index_items(img, new.all(), seed_region=seed)
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
    report.items_reconciled_removed += await reconcile.remove_dropped(ctx, plan.dropped)

    ledger_items = list(row['items'])
    cids = outcome.crop_ids
    n_labeled, n_standalone, n_parent = len(new.labeled), len(new.standalone), len(new.parents)
    standalone_ids = cids[n_labeled : n_labeled + n_standalone]
    parent_ids = cids[n_labeled + n_standalone : n_labeled + n_standalone + n_parent]
    machine_ids = cids[n_labeled + n_standalone + n_parent :]
    ledger_items.extend({'crop_id': cid, 'action': 'standalone'} for cid in standalone_ids)
    ledger_items.extend({'crop_id': cid, 'action': 'parent'} for cid in parent_ids)
    ledger_items.extend({'crop_id': cid, 'action': 'proposal'} for cid in machine_ids)

    ledger_boxes: list[dict[str, Any]] = []
    await _write_regions(ctx, entry, plan.attach, parent_keys, state, now, report, ledger_boxes)

    if plan.merge_plan is not None:
        await note_matches(ctx, plan.merge_plan)
        _tally_proposals(report, entry, plan.merge_plan, len(machine_ids))

    _count_plans(report, plans, updates, n_labeled)
    report.parents_detected += n_parent
    report.standalone_regions += n_standalone
    ledger_boxes.extend(
        {'crop_id': cid, 'box_id': 'b1', 'bbox_norm': list(box.bbox_norm)}
        for cid, box in zip(standalone_ids, plan.attach.standalone, strict=True)
    )
    _tally_image(report, ctx, entry, state, created_image, plan, new)

    final = {
        **row,
        'status': 'ok',
        'items': ledger_items,
        'boxes': ledger_boxes,
        'report': report.to_dict(),
    }
    store.append_ledger(chunk, final)
    return report


def _hash_of(data: bytes) -> str:
    from src.services.curation.ingest import _imohash_bytes

    return _imohash_bytes(data)


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
