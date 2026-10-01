"""Delete documents (and an item's crop-cache file) only if they are still
deletable when the delete happens.

Curation never deletes in normal operation; the callers are import undo
(items and images the import created), and the ``detect`` reprocess scope
(an unlocked machine item a re-run no longer finds). Each decides on a
snapshot taken earlier, and a human can edit in between, so this is the one
place a delete is made: it re-reads every document fresh, asks the caller's
``deletable`` predicate about the fresh source, and deletes with
``if_seq_no`` / ``if_primary_term`` so an edit that lands after the re-read
turns the delete into a conflict that is re-read, never a lost edit.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.clients.occ import OCC_BULK_MGET_SOURCE_EXCLUDES
from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

_BULK_PAGE = 500
_MAX_ROUNDS = 4

Deletable = Callable[[str, dict[str, Any]], Awaitable[bool]]
"""Called with a document's id and fresh ``_source``; ``True`` keeps it on the
delete list. Async so a predicate may consult the index (an image is only
deletable while no item stands on it)."""


async def delete_items(
    opensearch: AsyncOpenSearch,
    ids: list[str],
    *,
    items_index: str,
    crop_cache_dir: str | Path | None,
    deletable: Deletable,
) -> dict[str, Any]:
    """Delete each of ``ids`` from ``items_index`` that is still
    ``deletable`` at write time. Returns ``{'deleted': int, 'skipped':
    [id], 'errors': [{crop_id, error}]}``.

    A document that is already gone counts as deleted (the goal is
    "absent"). A document the predicate rejects, or one that keeps changing
    under every retry, is ``skipped`` (the human edit wins). Any other bulk
    failure is reported per id, never swallowed.
    """
    deleted = gone = 0
    skipped: list[str] = []
    errors: list[dict[str, Any]] = []
    for start in range(0, len(ids), _BULK_PAGE):
        pending = ids[start : start + _BULK_PAGE]
        for _ in range(_MAX_ROUNDS):
            if not pending:
                break
            done, missing, conflicted = await _delete_round(
                opensearch, pending, items_index, crop_cache_dir, deletable, skipped, errors
            )
            deleted += done
            gone += missing
            pending = conflicted
        skipped.extend(pending)
    return {'deleted': deleted, 'gone': gone, 'skipped': skipped, 'errors': errors}


async def _fresh(
    opensearch: AsyncOpenSearch, index: str, ids: list[str]
) -> dict[str, tuple[dict[str, Any], int, int]]:
    resp = await opensearch.mget(
        body={'docs': [{'_id': i, '_index': index} for i in ids]},
        _source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES,
    )
    return {
        d['_id']: (d.get('_source') or {}, int(d['_seq_no']), int(d['_primary_term']))
        for d in resp.get('docs') or []
        if d.get('found')
    }


async def _delete_round(
    opensearch: AsyncOpenSearch,
    page: list[str],
    index: str,
    crop_cache_dir: str | Path | None,
    deletable: Deletable,
    skipped: list[str],
    errors: list[dict[str, Any]],
) -> tuple[int, int, list[str]]:
    """One re-read + conditional bulk delete. Returns ``(deleted, already
    gone, ids that hit a version conflict and need another round)``."""
    try:
        fresh = await _fresh(opensearch, index, page)
    except Exception as exc:
        errors.extend({'crop_id': cid, 'error': str(exc)} for cid in page)
        return 0, 0, []
    deleted = gone = 0
    body: list[dict[str, Any]] = []
    targets: list[str] = []
    for cid in page:
        if cid not in fresh:
            gone += 1
            _drop_cached_crop(crop_cache_dir, cid)
            continue
        source, seq_no, term = fresh[cid]
        if not await deletable(cid, source):
            skipped.append(cid)
            continue
        targets.append(cid)
        body.append(
            {'delete': {'_index': index, '_id': cid, 'if_seq_no': seq_no, 'if_primary_term': term}}
        )
    if not body:
        return deleted, gone, []
    try:
        resp = await opensearch.bulk(body=body, refresh=False)
    except Exception as exc:
        errors.extend({'crop_id': cid, 'error': str(exc)} for cid in targets)
        return deleted, gone, []
    conflicted: list[str] = []
    for cid, item in zip(targets, resp.get('items') or [], strict=False):
        result = item.get('delete') or {}
        status = int(result.get('status', 0))
        if status in (200, 404):
            deleted += 1
            _drop_cached_crop(crop_cache_dir, cid)
        elif status == 409:
            conflicted.append(cid)
        else:
            errors.append({'crop_id': cid, 'error': str(result.get('error') or status)})
    return deleted, gone, conflicted


def _drop_cached_crop(cache_dir: str | Path | None, crop_id: str) -> None:
    if not cache_dir:
        return
    with contextlib.suppress(OSError):
        (Path(cache_dir) / f'{crop_id}.jpg').unlink()


__all__ = ['Deletable', 'delete_items']
