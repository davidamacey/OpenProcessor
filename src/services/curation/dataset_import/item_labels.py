"""Item-class labels of one image (W10.6 steps 2-4): match the dataset's
boxes to the image's existing items, decide per box what the import does,
and apply it through the single class-label writer.

Planning (:func:`plan_item_labels`) is pure and reads a snapshot of the
image's items; applying (:func:`apply_item_updates`) re-checks ownership on
the document the OCC write actually reads, so a human edit that lands
between the plan and the write still wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.clients.occ import occ_update_one
from src.clients.occ_locks import is_human_marker
from src.services.curation.class_label import ItemLabel, class_label_update
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.detection.geometry import crop_id as make_crop_id, iou


if TYPE_CHECKING:
    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.mapping import MapTarget
    from src.services.curation.dataset_import.scan import LabelBox, ScanEntry

LABEL_IOU_MATCH = 0.5

Action = Literal['created', 'updated', 'noop', 'locked_conflict']


def human_owned(src: dict[str, Any]) -> bool:
    """A human wrote this item's class (or froze it into the holdout): an
    import never overwrites it. A holdout item whose class an earlier
    import set is the import's own and stays correctable by a newer one."""
    if is_human_marker(src.get('class_source')) or is_human_marker(src.get('label_source')):
        return True
    return bool(src.get('test_holdout')) and src.get('class_source') != LABEL_IMPORT_CLASS_SOURCE


@dataclass
class ItemPlan:
    box: LabelBox
    target: MapTarget
    crop_id: str
    action: Action
    existing: dict[str, Any] | None = None
    snapshot_index: int | None = None
    holdout_prior: bool | None = None


def plan_item_labels(
    existing: list[dict[str, Any]],
    boxes: list[tuple[LabelBox, MapTarget]],
    *,
    image_id: str,
) -> list[ItemPlan]:
    """One :class:`ItemPlan` per labeled box, in box order.

    A box matches the existing item with the same ``crop_id``, else the
    unclaimed one with the highest IoU >= :data:`LABEL_IOU_MATCH` (ties:
    lowest ``crop_id``); each item is claimed at most once.
    """
    by_id = {str(d['crop_id']): d for d in existing if d.get('crop_id')}
    claimed: set[str] = set()
    plans: list[ItemPlan] = []
    for box, target in boxes:
        cid = make_crop_id(image_id, list(box.bbox_norm))
        match: dict[str, Any] | None = None
        if cid in by_id and cid not in claimed:
            match = by_id[cid]
        else:
            best = 0.0
            for other_id in sorted(by_id):
                if other_id in claimed:
                    continue
                other_box = by_id[other_id].get('bbox_norm') or (0.0, 0.0, 0.0, 0.0)
                score = iou(tuple(other_box), box.bbox_norm)  # type: ignore[arg-type]
                if score >= LABEL_IOU_MATCH and score > best:
                    best, match = score, by_id[other_id]
        if match is None:
            plans.append(ItemPlan(box, target, cid, 'created'))
            continue
        claimed.add(str(match['crop_id']))
        plans.append(_plan_existing(box, target, match))
    return plans


def _plan_existing(box: LabelBox, target: MapTarget, src: dict[str, Any]) -> ItemPlan:
    cid = str(src['crop_id'])
    same_class = src.get('class_id') == target.class_id
    holdout = bool(src.get('test_holdout'))
    if human_owned(src):
        return ItemPlan(box, target, cid, 'noop' if same_class else 'locked_conflict', src)
    if src.get('class_source') == LABEL_IMPORT_CLASS_SOURCE and same_class:
        return ItemPlan(box, target, cid, 'noop', src)
    history = src.get('class_id_history') or []
    return ItemPlan(
        box, target, cid, 'updated', src, snapshot_index=len(history), holdout_prior=holdout
    )


def import_stamp(
    ctx: ImportContext, entry: ScanEntry, current: dict[str, Any], now: str
) -> dict[str, Any]:
    """The provenance/split fields an import stamps on an item it writes."""
    ids = list(current.get('import_ids') or [])
    if ctx.import_id not in ids:
        ids.append(ctx.import_id)
    stamp: dict[str, Any] = {
        'import_ids': ids,
        'imported_at': now,
        'import_dataset_name': ctx.options.name,
        'import_dataset_sha': ctx.source_sha,
        'import_source_stem': entry.source_stem,
    }
    if entry.split:
        stamp['dataset_split'] = entry.split
    if entry.stratum:
        stamp['import_stratum'] = entry.stratum
    if entry.hard_negative:
        stamp['import_hard_negative'] = True
    if ctx.freeze_test and entry.split == 'test':
        stamp['test_holdout'] = True
    return stamp


async def apply_item_updates(
    ctx: ImportContext, entry: ScanEntry, plans: list[ItemPlan], now: str
) -> dict[str, str]:
    """Write every ``updated`` plan through :func:`class_label_update`.

    Returns ``{crop_id: outcome}`` where outcome is ``updated``,
    ``locked_conflict`` (a human edit landed after the plan) or ``noop``
    (this import already wrote it: a resumed chunk).
    """
    outcomes: dict[str, str] = {}
    for plan in plans:
        if plan.action != 'updated':
            continue
        label = ItemLabel.imported(
            import_id=ctx.import_id,
            class_id=plan.target.class_id,  # type: ignore[arg-type]
            class_name=plan.target.class_name,  # type: ignore[arg-type]
            trust=ctx.options.label_trust,
            now=now,
            writer=ctx.writer,
        )
        state = {'outcome': 'updated'}

        def merger(
            current: dict[str, Any], _label: ItemLabel = label, _state: dict[str, str] = state
        ) -> dict[str, Any]:
            if human_owned(current):
                _state['outcome'] = 'locked_conflict'
                return {}
            if any(h.get('writer') == ctx.writer for h in current.get('class_id_history') or []):
                _state['outcome'] = 'noop'
                return {}
            fields = class_label_update(current, _label)
            fields.update(import_stamp(ctx, entry, current, now))
            return fields

        await occ_update_one(
            ctx.opensearch,
            doc_id=plan.crop_id,
            merger=merger,
            index=ctx.items_index,
            writer_id=ctx.writer,
        )
        outcomes[plan.crop_id] = state['outcome']
    return outcomes


async def stamp_matched_noops(ctx: ImportContext, plans: list[ItemPlan]) -> None:
    """Append this import's id to an import-owned item whose label already
    matches (a different import wrote it): provenance only, never the class."""
    for plan in plans:
        if plan.action != 'noop' or plan.existing is None:
            continue
        src = plan.existing
        if src.get('class_source') != LABEL_IMPORT_CLASS_SOURCE or ctx.import_id in (
            src.get('import_ids') or []
        ):
            continue

        def merger(current: dict[str, Any]) -> dict[str, Any]:
            ids = list(current.get('import_ids') or [])
            if ctx.import_id in ids:
                return {}
            return {'import_ids': [*ids, ctx.import_id]}

        await occ_update_one(
            ctx.opensearch,
            doc_id=plan.crop_id,
            merger=merger,
            index=ctx.items_index,
            writer_id=ctx.writer,
        )


__all__ = [
    'LABEL_IOU_MATCH',
    'ItemPlan',
    'apply_item_updates',
    'human_owned',
    'import_stamp',
    'plan_item_labels',
    'stamp_matched_noops',
]
