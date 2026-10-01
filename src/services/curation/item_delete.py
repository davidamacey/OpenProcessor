"""Delete items: the one place an items doc (and its crop-cache file) is
removed.

Curation never deletes items in normal operation; the two callers are
import undo (an item the import created, nothing else owns) and the
``detect`` reprocess scope (an unlocked machine item a re-run no longer
finds). Both pass only ids they have already decided are unlocked; this
module does not re-decide that.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)

_BULK_PAGE = 500


async def delete_items(
    opensearch: AsyncOpenSearch,
    crop_ids: list[str],
    *,
    items_index: str,
    crop_cache_dir: str | Path | None,
) -> dict[str, Any]:
    """Bulk-delete ``crop_ids`` from ``items_index`` and drop their cached
    crop files. Returns ``{'deleted': int, 'errors': [{crop_id, error}]}``.

    A doc that is already gone counts as deleted (the goal is "absent"). Any
    other bulk failure is reported per id, never swallowed.
    """
    deleted = 0
    errors: list[dict[str, Any]] = []
    for start in range(0, len(crop_ids), _BULK_PAGE):
        page = crop_ids[start : start + _BULK_PAGE]
        body = [{'delete': {'_index': items_index, '_id': cid}} for cid in page]
        try:
            resp = await opensearch.bulk(body=body, refresh=False)
        except Exception as exc:
            errors.extend({'crop_id': cid, 'error': str(exc)} for cid in page)
            continue
        for cid, item in zip(page, resp.get('items') or [], strict=False):
            result = item.get('delete') or {}
            status = int(result.get('status', 0))
            if status in (200, 404):
                deleted += 1
                _drop_cached_crop(crop_cache_dir, cid)
            else:
                errors.append({'crop_id': cid, 'error': str(result.get('error') or status)})
    return {'deleted': deleted, 'errors': errors}


def _drop_cached_crop(cache_dir: str | Path | None, crop_id: str) -> None:
    if not cache_dir:
        return
    with contextlib.suppress(OSError):
        (Path(cache_dir) / f'{crop_id}.jpg').unlink()


__all__ = ['delete_items']
