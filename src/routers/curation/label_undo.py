"""Curation router sub-module — undo of human class writes, the other
recorded per-item decisions (discard, VLM-suggestion dismissal) and the
item's class history.

Every human class write (``PUT /crops/{id}/label``, ``PUT
/crops/batch_label``, ``POST /crops/move``) records the item's full
pre-write class state in ``class_id_history`` (see
:func:`src.services.curation.history.record_class_snapshot`). The routes
here restore that state, so the labeler never has to decide itself
whether "undo" means re-applying an earlier label or reverting to the
machine suggestion — the backend knows what the item looked like.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    CropDiscardBatchRequest,
    CropDiscardRequest,
    CropUndoBatchRequest,
    OpenSearchDep,
    _now_iso,
    is_not_found,
    logger,
    router,
)
from src.services.curation.class_sources import vlm_suggestion
from src.services.curation.edit_history import EDIT_HISTORY_FIELD, EditKind, record_edit
from src.services.curation.exclusion import park_restored_state_while_excluded
from src.services.curation.history import (
    CLASS_STATE_FIELDS,
    HUMAN_DISCARD_WRITER,
    HUMAN_UNLABEL_WRITER,
    REVIEW_DISMISS_FIELDS,
    find_undo_snapshot,
    record_class_snapshot,
    restore_class_state,
)
from src.services.curation.wire import item_source_excludes, serialize_item


class NothingToUndoError(Exception):
    """The item has no un-undone human class write on record."""


def _undo_merger(*, require_history: bool) -> Any:
    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        snapshot = find_undo_snapshot(current.get('class_id_history'))
        if snapshot is None and require_history:
            raise NothingToUndoError
        restored = restore_class_state(snapshot)
        if current.get('class_excluded'):
            # Still excluded: the restored validation/placement is what
            # un-exclude should bring back, not what applies right now.
            restored = park_restored_state_while_excluded(restored)
        history = record_class_snapshot(current, writer=HUMAN_UNLABEL_WRITER, restorable=False)
        return {**restored, 'class_id_history': history, 'updated_at': _now_iso()}

    return _merge


async def _undo_one(opensearch: Any, crop_id: str, *, require_history: bool) -> None:
    """Restore one crop. Raises NothingToUndoError, OCCFinalConflictError,
    or HTTPException(404)."""
    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=_undo_merger(require_history=require_history),
            refresh='wait_for',
            writer_id=HUMAN_UNLABEL_WRITER,
        )
    except (NothingToUndoError, OCCFinalConflictError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc


async def _items_by_ids(opensearch: Any, ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    resp = await opensearch.mget(
        index=CURATION_ITEMS_INDEX,
        body={'ids': ids},
        _source_excludes=item_source_excludes(),
    )
    return [
        serialize_item(d.get('_source') or {}, d.get('_id', ''))
        for d in (resp.get('docs') or [])
        if d.get('found')
    ]


@router.post('/crops/{crop_id}/label/undo')
async def undo_crop_label(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Restore the crop to its state before its most recent human class write.

    Whatever that state was — an earlier validated human label, a VLM
    suggestion, an ingest proposal, unlabeled — class, provenance,
    validation and cluster placement come back exactly as recorded.
    Repeated calls step back through successive human writes. Returns the
    restored item (shared wire format). ``409`` when the crop has no human
    class write left to undo.
    """
    try:
        await _undo_one(opensearch, crop_id, require_history=True)
    except NothingToUndoError as exc:
        raise HTTPException(status_code=409, detail=f'nothing to undo for crop {crop_id}') from exc
    items = await _items_by_ids(opensearch, [crop_id])
    if not items:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}')
    return items[0]


@router.post('/crops/label/undo_batch')
async def undo_crop_labels(
    payload: CropUndoBatchRequest, opensearch: OpenSearchDep
) -> dict[str, Any]:
    """Batch form of ``POST /crops/{crop_id}/label/undo``.

    Each crop is restored independently to its own state before its most
    recent human class write — so undoing a ``batch_label`` or ``move``
    means passing the same ``crop_ids``. Returns ``items`` (restored wire
    items), ``undone``, and per-crop ``nothing_to_undo`` / ``conflicts``
    / ``not_found`` id lists. ``409`` when no crop had anything to undo.
    """
    crop_ids = list(dict.fromkeys(payload.crop_ids))
    if not crop_ids:
        return {'items': [], 'undone': 0, 'nothing_to_undo': [], 'conflicts': [], 'not_found': []}

    async def _one(crop_id: str) -> str:
        try:
            await _undo_one(opensearch, crop_id, require_history=True)
        except NothingToUndoError:
            return 'nothing_to_undo'
        except OCCFinalConflictError:
            return 'conflicts'
        except HTTPException:
            return 'not_found'
        except Exception as exc:
            logger.warning('undo_label_failed', crop_id=crop_id, error=str(exc))
            return 'conflicts'
        return 'undone'

    outcomes = await asyncio.gather(*(_one(cid) for cid in crop_ids))
    by_outcome: dict[str, list[str]] = {
        'undone': [],
        'nothing_to_undo': [],
        'conflicts': [],
        'not_found': [],
    }
    for crop_id, outcome in zip(crop_ids, outcomes, strict=True):
        by_outcome[outcome].append(crop_id)
    if (
        not by_outcome['undone']
        and by_outcome['nothing_to_undo']
        and not (by_outcome['conflicts'] or by_outcome['not_found'])
    ):
        raise HTTPException(status_code=409, detail='nothing to undo for any crop')
    return {
        'items': await _items_by_ids(opensearch, by_outcome['undone']),
        'undone': len(by_outcome['undone']),
        'nothing_to_undo': by_outcome['nothing_to_undo'],
        'conflicts': by_outcome['conflicts'],
        'not_found': by_outcome['not_found'],
    }


@router.delete('/crops/{crop_id}/label')
async def unlabel_crop(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Undo the most recent human class label (legacy Undo route).

    Same restore as ``POST /crops/{crop_id}/label/undo``; the one
    difference is that with no human write on record it resets the crop
    to unlabeled (class and provenance cleared, nothing invented) instead
    of answering ``409``. Only the class side is touched; the region-side
    validated flag is independent.
    """
    await _undo_one(opensearch, crop_id, require_history=False)
    return {'crop_id': crop_id, 'reset': True}


def _discard_merger(payload: CropDiscardRequest) -> Any:
    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        now = _now_iso()
        update: dict[str, Any] = {
            'class_id_history': record_class_snapshot(
                current, writer=HUMAN_DISCARD_WRITER, restorable=True
            ),
            'updated_at': now,
        }
        if payload.clear_class:
            update.update(dict.fromkeys(CLASS_STATE_FIELDS))
            update['class_validated'] = False
        if payload.dismiss_from_review:
            update['review_dismissed_at'] = now
            update['review_dismissed_by'] = 'human'
        return update

    return _merge


def _require_effect(payload: CropDiscardRequest) -> None:
    if not (payload.clear_class or payload.dismiss_from_review):
        raise HTTPException(
            status_code=422, detail='discard needs clear_class and/or dismiss_from_review'
        )


async def _discard_one(opensearch: Any, crop_id: str, payload: CropDiscardRequest) -> None:
    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=_discard_merger(payload),
            refresh='wait_for',
            writer_id=HUMAN_DISCARD_WRITER,
        )
    except OCCFinalConflictError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc


@router.post('/crops/{crop_id}/discard')
async def discard_crop(
    crop_id: str, payload: CropDiscardRequest, opensearch: OpenSearchDep
) -> dict[str, Any]:
    """Discard an item, as a recorded human write.

    ``clear_class`` (default ``true``): the item doesn't belong in its
    class/cluster — class, provenance and validation are cleared and it
    drops to the residual pool (``cluster_id: null``).
    ``dismiss_from_review`` (default ``false``): hide it from every
    ``/review`` tab (``review_dismissed_at``). The pre-write state is
    snapshotted, so ``POST /crops/{crop_id}/label/undo`` restores it
    exactly. Returns the post-write item. ``422`` when neither is set.
    """
    _require_effect(payload)
    await _discard_one(opensearch, crop_id, payload)
    items = await _items_by_ids(opensearch, [crop_id])
    if not items:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}')
    return items[0]


@router.post('/crops/discard_batch')
async def discard_crops(
    payload: CropDiscardBatchRequest, opensearch: OpenSearchDep
) -> dict[str, Any]:
    """Batch form of ``POST /crops/{crop_id}/discard``. Returns ``items``
    (post-write wire items), ``discarded``, and ``conflicts`` /
    ``not_found`` id lists. Undo with ``POST /crops/label/undo_batch``
    passing the discarded ids."""
    _require_effect(payload)
    crop_ids = list(dict.fromkeys(payload.crop_ids))

    async def _one(crop_id: str) -> str:
        try:
            await _discard_one(opensearch, crop_id, payload)
        except OCCFinalConflictError:
            return 'conflicts'
        except HTTPException:
            return 'not_found'
        except Exception as exc:
            logger.warning('discard_failed', crop_id=crop_id, error=str(exc))
            return 'conflicts'
        return 'discarded'

    outcomes = await asyncio.gather(*(_one(cid) for cid in crop_ids))
    by: dict[str, list[str]] = {'discarded': [], 'conflicts': [], 'not_found': []}
    for crop_id, outcome in zip(crop_ids, outcomes, strict=True):
        by[outcome].append(crop_id)
    return {
        'items': await _items_by_ids(opensearch, by['discarded']),
        'discarded': len(by['discarded']),
        'conflicts': by['conflicts'],
        'not_found': by['not_found'],
    }


class _NoSuggestionError(Exception):
    pass


@router.post('/crops/{crop_id}/vlm_dismiss')
async def dismiss_vlm_suggestion(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Reject the VLM's class suggestion on this item.

    Records ``vlm_dismissed_class_id`` / ``vlm_dismissed_class_name`` /
    ``vlm_dismissed_at``; while the VLM's suggestion is the dismissed one,
    ``vlm_proposed_class_*`` are null and ``proposed_class_*`` no longer
    apply it. The class itself is untouched (label or discard it as a
    separate write). ``POST /crops/{crop_id}/vlm_dismiss/undo`` reverses it.
    Returns the post-write item; ``409`` when the item has no VLM
    suggestion.
    """

    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        class_id, class_name = vlm_suggestion(current)
        if class_name is None:
            raise _NoSuggestionError
        return {
            'vlm_dismissed_class_id': class_id,
            'vlm_dismissed_class_name': class_name,
            'vlm_dismissed_at': _now_iso(),
            EDIT_HISTORY_FIELD: record_edit(
                current, kind=EditKind.VLM_DISMISS, writer='human:vlm_dismiss'
            ),
            'updated_at': _now_iso(),
        }

    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=_merge,
            refresh='wait_for',
            writer_id='human:vlm_dismiss',
        )
    except _NoSuggestionError as exc:
        raise HTTPException(status_code=409, detail=f'no VLM suggestion on {crop_id}') from exc
    except OCCFinalConflictError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    items = await _items_by_ids(opensearch, [crop_id])
    if not items:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}')
    return items[0]


_HISTORY_KEYS: tuple[str, ...] = (*CLASS_STATE_FIELDS, *REVIEW_DISMISS_FIELDS, 'writer', 'at')


@router.get('/crops/{crop_id}/history')
async def crop_history(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """The item's class history, oldest first: ``{crop_id, entries}``.

    Each entry is the item's class state *before* one write
    (``class_id``, ``class_name``, ``class_source``, ``label_source``,
    ``confidence``, ``class_detector*``, ``class_labeler``,
    ``class_labeled_at``, ``class_validated``, ``cluster_id``,
    ``cluster_subid``; ``review_dismissed_*`` on discards) plus ``writer``
    (who made that write, e.g. ``human:label_crop``, ``vlm_pipeline``) and
    ``at``. Keys a writer didn't record are ``null``.
    """
    try:
        resp = await opensearch.get(
            index=CURATION_ITEMS_INDEX, id=crop_id, _source_includes=['class_id_history']
        )
    except Exception as exc:
        if is_not_found(exc):
            raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}') from exc
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    history = (resp.get('_source') or {}).get('class_id_history') or []
    entries = [
        {k: entry.get(k) for k in _HISTORY_KEYS} for entry in history if isinstance(entry, dict)
    ]
    return {'crop_id': crop_id, 'entries': entries}


@router.post('/crops/{crop_id}/review_undismiss')
async def review_undismiss(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Return an item hidden from review to the queues (clears
    ``review_dismissed_at`` / ``review_dismissed_by``). For a dismissal made
    by ``POST /crops/{id}/discard``, ``label/undo`` does the same and also
    restores the class. Returns the post-write item."""
    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=lambda _c: {
                'review_dismissed_at': None,
                'review_dismissed_by': None,
                'updated_at': _now_iso(),
            },
            refresh='wait_for',
            writer_id='human:review_undismiss',
        )
    except OCCFinalConflictError:
        raise
    except Exception as exc:
        if is_not_found(exc):
            raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}') from exc
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    items = await _items_by_ids(opensearch, [crop_id])
    if not items:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}')
    return items[0]
