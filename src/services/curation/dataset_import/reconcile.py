"""Reconcile a newer dataset version over an earlier import (W10.11).

An image never carries two versions' labels: an item an earlier import wrote
and this version no longer has is removed, unless a human (or a holdout
freeze) has touched it since. The removed document is kept whole in this
import's write-ahead ledger row (``action: reconciled``), so undoing this
import puts it back, and the removal goes through the one delete path that
re-checks ownership on the fresh document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.occ_locks import is_human_marker
from src.services.curation.dataset_import.item_labels import match_existing
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.curation.item_delete import delete_items
from src.services.curation.region_boxes import is_human_owned, read_boxes


if TYPE_CHECKING:
    from src.config.region_fields import RegionFields
    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.item_labels import ItemPlan
    from src.services.curation.dataset_import.scan import ScanEntry

ACTION = 'reconciled'


def reconcilable(doc: dict[str, Any], import_id: str, fields: RegionFields) -> bool:
    """An item only earlier imports wrote, and nothing since has touched: its
    class is an import's and not a human's, it is not frozen into the
    holdout, and it holds no human box."""
    ids = doc.get('import_ids') or []
    return (
        bool(ids)
        and import_id not in ids
        and doc.get('class_source') == LABEL_IMPORT_CLASS_SOURCE
        and not is_human_marker(doc.get('label_source'))
        and not doc.get('test_holdout')
        and not doc.get('import_standalone_region')
        and not any(is_human_owned(b) for b in read_boxes(doc, fields))
    )


async def dropped_entries(
    ctx: ImportContext,
    existing: list[dict[str, Any]],
    entry: ScanEntry,
    plans: list[ItemPlan],
    *,
    image_id: str,
    prior: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Ledger entries (``{crop_id, action, doc}``) for every item this
    version drops. A box the dataset still has, even one mapped to ``skip``,
    keeps its item; so does a frame with no label file (it says nothing).
    Entries a crashed earlier attempt already recorded are carried over: the
    docs they name may be gone by now."""
    carried = [i for i in (prior or {}).get('items') or [] if i.get('action') == ACTION]
    if entry.label_state == 'unlabeled':
        return carried
    kept = {p.crop_id for p in plans}
    kept.update(
        str(m['crop_id'])
        for m in match_existing(existing, list(entry.boxes), image_id=image_id)
        if m is not None
    )
    have = {i['crop_id'] for i in carried}
    wanted = [
        str(d['crop_id'])
        for d in existing
        if str(d['crop_id']) not in kept
        and str(d['crop_id']) not in have
        and reconcilable(d, ctx.import_id, ctx.region_fields)
    ]
    if not wanted:
        return carried
    resp = await ctx.opensearch.mget(
        body={'docs': [{'_id': cid, '_index': ctx.items_index} for cid in wanted]}
    )
    full = [
        {'crop_id': d['_id'], 'action': ACTION, 'doc': d['_source']}
        for d in resp.get('docs') or []
        if d.get('found') and d.get('_source')
    ]
    return [*carried, *full]


async def remove_dropped(ctx: ImportContext, dropped: list[dict[str, Any]]) -> int:
    """Delete the dropped items that are still reconcilable on the fresh
    document; returns how many are gone."""
    if not dropped:
        return 0

    async def still_reconcilable(_crop_id: str, doc: dict[str, Any]) -> bool:
        return reconcilable(doc, ctx.import_id, ctx.region_fields)

    result = await delete_items(
        ctx.opensearch,
        [d['crop_id'] for d in dropped],
        items_index=ctx.items_index,
        crop_cache_dir=ctx.crop_cache_dir,
        deletable=still_reconcilable,
    )
    if result['errors']:
        raise RuntimeError(f'reconcile delete failed: {result["errors"][:3]}')
    return len(dropped) - len(result['skipped'])


__all__ = ['ACTION', 'dropped_entries', 'reconcilable', 'remove_dropped']
