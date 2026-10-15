"""OCC bulk relabel for the crop write routes: one mget page plus one bulk
call per batch instead of one ``occ_update_one`` round-trip per crop."""

from __future__ import annotations

from typing import Any

from src.clients.occ_bulk import occ_update_bulk
from src.config.curation_accessors import items_index
from src.core.logging import get_logger


logger = get_logger(__name__)


async def occ_bulk_human_relabel(
    opensearch: Any,
    crop_ids: list[str],
    merger: Any,
    *,
    writer_id: str,
    max_retries: int = 5,
) -> dict[str, Any]:
    """Batch OCC relabel: one mget page + one bulk call instead of
    one ``occ_update_one`` round-trip per crop. ``refresh='wait_for'`` is
    attached only to the final bulk call of each retry round (see
    :func:`src.clients.occ_bulk.occ_update_bulk`), not per crop.

    ``current_source`` on each conflict entry is looked up with one
    follow-up ``mget`` restricted to the (normally empty) conflict set,
    to preserve the pre-existing response shape without re-adding a
    per-crop read.
    """
    status_map = await occ_update_bulk(
        opensearch,
        index=items_index(),
        ids=list(crop_ids),
        merge_fn=lambda _crop_id, source: merger(source),
        max_retries=max_retries,
        refresh='wait_for',
    )
    updated_ids = [cid for cid in crop_ids if status_map.get(cid) == 'updated']
    conflict_ids = [cid for cid in crop_ids if status_map.get(cid) != 'updated']

    conflicts: list[dict[str, Any]] = []
    if conflict_ids:
        from src.clients.curation_opensearch.crops import mget_crops

        docs = await mget_crops(
            opensearch,
            conflict_ids,
            index=items_index(),
            source_includes=['class_source'],
        )
        for cid in conflict_ids:
            source = (docs.get(cid) or {}).get('_source') or {}
            conflicts.append({'crop_id': cid, 'current_source': source.get('class_source')})
            logger.warning(f'{writer_id}_update_failed', crop_id=cid)

    return {'updated': len(updated_ids), 'updated_ids': updated_ids, 'conflicts': conflicts}
