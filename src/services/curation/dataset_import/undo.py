"""Undo one import batch (W10.12): remove only what the import wrote, never
a later human edit.

Every decision is a pure function of the document as the OCC write reads it
(``decide_*``), so the dry run and the apply run the same code and a human
edit that lands between them is respected. The ledger says what the import
did to each item (``created`` / ``updated`` / ``noop`` ...); the document says
who owns each part NOW.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_update_one
from src.clients.occ_locks import is_human_marker
from src.config.region_source import CANDIDATE_IMPORT
from src.config.region_state import RegionStatus
from src.services.curation.edit_history import (
    EDIT_HISTORY_FIELD,
    EditKind,
    region_state_fields,
    restore_edit_state,
)
from src.services.curation.history import CLASS_STATE_FIELDS, restore_class_state
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.curation.item_delete import delete_items
from src.services.curation.region_boxes import (
    RegionBox,
    boxes_write_fields,
    derive_status,
    is_human_owned,
    read_boxes,
)


if TYPE_CHECKING:
    from pathlib import Path

    from src.clients.curation_opensearch import ClassRegistry
    from src.config.region_fields import RegionFields
    from src.services.curation.dataset_import.store import ImportStore

_IMPORT_STAMP_FIELDS = (
    'imported_at',
    'import_dataset_name',
    'import_dataset_sha',
    'import_source_stem',
    'import_stratum',
    'import_hard_negative',
    'dataset_split',
    'proposal_chain',
    'proposed_by_import',
    'on_negative_frame',
    'import_standalone_region',
)
_MAX_SAMPLES = 20


@dataclass
class UndoReport:
    import_id: str
    dry_run: bool
    items_deleted: int = 0
    items_restored: int = 0
    items_kept_human_edited: int = 0
    items_kept_shared: int = 0
    class_labels_removed: int = 0
    boxes_removed: int = 0
    boxes_kept_human_edited: int = 0
    proposals_deleted: int = 0
    holdout_flags_cleared: int = 0
    images_deleted: int = 0
    images_kept: int = 0
    classes_deprecated: list[str] = field(default_factory=list)
    samples: dict[str, list[Any]] = field(
        default_factory=lambda: {'kept_human_edited': [], 'boxes_kept_human_edited': []}
    )

    def sample(self, key: str, value: Any) -> None:
        if len(self.samples[key]) < _MAX_SAMPLES:
            self.samples[key].append(value)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class UndoContext:
    import_id: str
    opensearch: Any
    images_index: str
    items_index: str
    crop_cache_dir: str | Path | None
    region_fields: RegionFields
    registry: ClassRegistry | None = None

    @property
    def writer(self) -> str:
        return f'import:{self.import_id}'


@dataclass
class Decision:
    kind: str
    """``delete`` | ``update`` | ``noop``."""
    fields: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    sample: tuple[str, Any] | None = None


def _without_import(doc: dict[str, Any], import_id: str) -> dict[str, Any]:
    ids = [i for i in doc.get('import_ids') or [] if i != import_id]
    out: dict[str, Any] = {'import_ids': ids}
    if not ids:
        out.update(dict.fromkeys(_IMPORT_STAMP_FIELDS))
    return out


def _class_is_import_owned(doc: dict[str, Any]) -> bool:
    return doc.get('class_source') == LABEL_IMPORT_CLASS_SOURCE and not is_human_marker(
        doc.get('label_source')
    )


def _import_boxes(
    boxes: list[RegionBox], import_id: str, ledger: list[dict[str, Any]]
) -> tuple[list[RegionBox], list[RegionBox]]:
    """``(still import's, edited by a human since)`` among the ledger's boxes."""
    wanted = {b['box_id']: b for b in ledger if b.get('box_id')}
    mine: list[RegionBox] = []
    edited: list[RegionBox] = []
    for b in boxes:
        row = wanted.get(b.box_id)
        if row is None:
            continue
        untouched = (
            b.source == CANDIDATE_IMPORT
            and b.detector_version == import_id
            and [round(v, 6) for v in b.bbox_norm] == [round(v, 6) for v in row['bbox_norm']]
            and not is_human_owned(b)
        )
        (mine if untouched else edited).append(b)
    return mine, edited


def _region_fields_after(
    doc: dict[str, Any],
    removed: list[RegionBox],
    F: RegionFields,
    writer: str,
    *,
    created: bool,
) -> dict[str, Any]:
    """Region fields once ``removed`` are gone: the snapshot when no human box
    remains (or cleared, for an item the import created), else the list
    rewritten and the status re-derived."""
    stored = read_boxes(doc, F)
    gone = {b.box_id for b in removed}
    remaining = [b for b in stored if b.box_id not in gone]
    if not any(is_human_owned(b) for b in remaining):
        if created:
            return dict.fromkeys(region_state_fields())
        for entry in reversed(doc.get(EDIT_HISTORY_FIELD) or []):
            if (
                isinstance(entry, dict)
                and entry.get('kind') == EditKind.REGION.value
                and entry.get('writer') == writer
            ):
                return restore_edit_state(entry, EditKind.REGION)
        return {}
    fields = boxes_write_fields(remaining, current_src=doc, F=F)
    fields[F.status] = derive_status(remaining, empty_status=RegionStatus.NO_REGION_VISIBLE).value
    return fields


def decide_created(
    doc: dict[str, Any], ctx: UndoContext, ledger_boxes: list[dict[str, Any]], action: str
) -> Decision:
    """An item the import created: delete it unless a human (or another
    import) also owns part of it."""
    F = ctx.region_fields
    if ctx.import_id not in (doc.get('import_ids') or []):
        return Decision('noop')
    others = [i for i in doc.get('import_ids') or [] if i != ctx.import_id]
    stored = read_boxes(doc, F)
    mine, edited = _import_boxes(stored, ctx.import_id, ledger_boxes)
    human_class = is_human_marker(doc.get('class_source')) or is_human_marker(
        doc.get('label_source')
    )
    human_box = any(is_human_owned(b) for b in stored)
    counts: dict[str, int] = {'boxes_removed': len(mine), 'boxes_kept_human_edited': len(edited)}
    if others:
        return Decision(
            'update', _without_import(doc, ctx.import_id), {**counts, 'items_kept_shared': 1}
        )
    if not (human_class or human_box or edited):
        key = 'proposals_deleted' if action in ('proposal', 'parent') else 'items_deleted'
        return Decision('delete', counts={**counts, key: 1})
    fields = _without_import(doc, ctx.import_id)
    if _class_is_import_owned(doc):
        fields.update(dict.fromkeys(CLASS_STATE_FIELDS))
        fields['class_validated'] = False
        counts['class_labels_removed'] = 1
    if mine:
        fields.update(_region_fields_after(doc, mine, F, ctx.writer, created=True))
    if doc.get('test_holdout'):
        fields['test_holdout'] = False
        counts['holdout_flags_cleared'] = 1
    counts['items_kept_human_edited'] = 1
    return Decision('update', fields, counts, sample=('kept_human_edited', doc.get('crop_id')))


def decide_updated(
    doc: dict[str, Any], ctx: UndoContext, entry: dict[str, Any], ledger_boxes: list[dict[str, Any]]
) -> Decision:
    """An existing item the import relabeled or wrote boxes on: restore the
    pre-import class state when the class is still the import's."""
    F = ctx.region_fields
    if ctx.import_id not in (doc.get('import_ids') or []):
        return Decision('noop')
    fields = _without_import(doc, ctx.import_id)
    counts: dict[str, int] = {}
    history = doc.get('class_id_history') or []
    snap_idx = next((i for i, h in enumerate(history) if h.get('writer') == ctx.writer), None)
    later_human = snap_idx is not None and any(
        not str(h.get('writer', '')).startswith('import:') for h in history[snap_idx + 1 :]
    )
    if snap_idx is not None and _class_is_import_owned(doc) and not later_human:
        fields.update(restore_class_state(history[snap_idx]))
        counts['items_restored'] = 1
        counts['class_labels_removed'] = 1
    elif snap_idx is not None:
        counts['items_kept_human_edited'] = 1
    if entry.get('holdout_prior') is False and doc.get('test_holdout'):
        fields['test_holdout'] = False
        counts['holdout_flags_cleared'] = 1
    stored = read_boxes(doc, F)
    mine, edited = _import_boxes(stored, ctx.import_id, ledger_boxes)
    if mine:
        fields.update(_region_fields_after(doc, mine, F, ctx.writer, created=False))
        counts['boxes_removed'] = len(mine)
    if edited:
        counts['boxes_kept_human_edited'] = len(edited)
    return Decision('update', fields, counts)


def decide_noop(doc: dict[str, Any], ctx: UndoContext) -> Decision:
    if ctx.import_id not in (doc.get('import_ids') or []):
        return Decision('noop')
    return Decision('update', _without_import(doc, ctx.import_id))


def _decide(
    doc: dict[str, Any], ctx: UndoContext, entry: dict[str, Any], ledger_boxes: list[dict[str, Any]]
) -> Decision:
    action = entry.get('action')
    if action in ('created', 'standalone', 'parent', 'proposal'):
        return decide_created(doc, ctx, ledger_boxes, action)
    if action == 'updated':
        return decide_updated(doc, ctx, entry, ledger_boxes)
    if action == 'noop':
        # A parent that only received region boxes is 'noop' for its class
        # but still owns import boxes.
        if ledger_boxes:
            return decide_updated(doc, ctx, entry, ledger_boxes)
        return decide_noop(doc, ctx)
    return Decision('noop')


def _tally(report: UndoReport, decision: Decision) -> None:
    for key, value in decision.counts.items():
        setattr(report, key, getattr(report, key) + value)
    if decision.sample is not None:
        report.sample(*decision.sample)


async def _mget(ctx: UndoContext, index: str, ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    resp = await ctx.opensearch.mget(body={'docs': [{'_id': i, '_index': index} for i in ids]})
    return {
        d['_id']: d['_source']
        for d in resp.get('docs') or []
        if d.get('found') and d.get('_source')
    }


async def undo_import(
    ctx: UndoContext,
    store: ImportStore,
    *,
    dry_run: bool,
    remove_images: bool = True,
    deprecate_created_classes: bool = True,
    created_classes: dict[str, int] | None = None,
) -> UndoReport:
    """Undo (or, with ``dry_run``, count) one import from its ledger."""
    report = UndoReport(import_id=ctx.import_id, dry_run=dry_run)
    to_delete: list[str] = []
    created_images: list[str] = []
    for row in store.ledger_rows():
        if row.get('status') == 'failed':
            continue
        entries = row.get('items') or []
        boxes_by_crop: dict[str, list[dict[str, Any]]] = {}
        for b in row.get('boxes') or []:
            boxes_by_crop.setdefault(b['crop_id'], []).append(b)
        crop_ids = list(dict.fromkeys([e['crop_id'] for e in entries] + list(boxes_by_crop)))
        docs = await _mget(ctx, ctx.items_index, crop_ids)
        seen: set[str] = set()
        for entry in entries:
            cid = entry['crop_id']
            if cid in seen or cid not in docs:
                continue
            seen.add(cid)
            decision = _decide(docs[cid], ctx, entry, boxes_by_crop.get(cid, []))
            _tally(report, decision)
            if decision.kind == 'delete':
                to_delete.append(cid)
            elif decision.kind == 'update' and not dry_run:
                await _write(ctx, cid, entry, boxes_by_crop.get(cid, []))
        if row.get('image_created') and row.get('image_id'):
            created_images.append(row['image_id'])
    if not dry_run and to_delete:
        await delete_items(
            ctx.opensearch,
            to_delete,
            items_index=ctx.items_index,
            crop_cache_dir=ctx.crop_cache_dir,
        )
    if remove_images:
        await _undo_images(ctx, report, created_images, set(to_delete), dry_run=dry_run)
    if deprecate_created_classes and ctx.registry is not None:
        await _deprecate_classes(
            ctx, report, created_classes or {}, set(to_delete), dry_run=dry_run
        )
    return report


async def _write(
    ctx: UndoContext, crop_id: str, entry: dict[str, Any], boxes: list[dict[str, Any]]
) -> None:
    def merger(current: dict[str, Any]) -> dict[str, Any]:
        return _decide(current, ctx, entry, boxes).fields

    await occ_update_one(
        ctx.opensearch, doc_id=crop_id, merger=merger, index=ctx.items_index, writer_id=ctx.writer
    )


async def _undo_images(
    ctx: UndoContext,
    report: UndoReport,
    created: list[str],
    deleting: set[str],
    *,
    dry_run: bool,
) -> None:
    """Delete an image the import created when nothing else is on it."""
    for image_id in created:
        resp = await ctx.opensearch.search(
            index=ctx.items_index,
            body={'size': 1000, 'query': {'term': {'image_id': image_id}}, '_source': ['crop_id']},
        )
        remaining = {
            h['_source'].get('crop_id') or h['_id']
            for h in (resp.get('hits') or {}).get('hits') or []
        } - deleting
        img = (await _mget(ctx, ctx.images_index, [image_id])).get(image_id)
        if img is None:
            continue  # already gone: a second undo reports nothing
        others = [i for i in img.get('import_ids') or [] if i != ctx.import_id]
        if remaining or others:
            report.images_kept += 1
            continue
        report.images_deleted += 1
        if not dry_run:
            await ctx.opensearch.bulk(
                body=[{'delete': {'_index': ctx.images_index, '_id': image_id}}], refresh=False
            )


async def _deprecate_classes(
    ctx: UndoContext,
    report: UndoReport,
    created: dict[str, int],
    deleting: set[str],
    *,
    dry_run: bool,
) -> None:
    """Deprecate classes the import created when no item references them."""
    assert ctx.registry is not None
    for _dataset_class, class_id in sorted(created.items()):
        resp = await ctx.opensearch.search(
            index=ctx.items_index,
            body={'size': 1000, 'query': {'term': {'class_id': class_id}}, '_source': ['crop_id']},
        )
        users = {
            h['_source'].get('crop_id') or h['_id']
            for h in (resp.get('hits') or {}).get('hits') or []
        } - deleting
        if users:
            continue
        entry = next((c for c in ctx.registry.load().classes if c.class_id == class_id), None)
        if entry is None or entry.deprecated:
            continue
        report.classes_deprecated.append(entry.class_name)
        if not dry_run:
            ctx.registry.set_deprecated(class_id, True)


__all__ = [
    'Decision',
    'UndoContext',
    'UndoReport',
    'decide_created',
    'decide_updated',
    'undo_import',
]
