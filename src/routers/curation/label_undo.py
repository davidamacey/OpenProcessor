"""Curation router sub-module — undo of human class writes, the other
recorded per-item decisions (discard, VLM-suggestion dismissal) and the
item's class history.

Every human class write (``PUT /crops/{id}/label``, ``PUT
/crops/batch_label``, ``POST /crops/move``) records the item's full
pre-write class state in ``class_id_history`` (see
:func:`src.domain.history.record_class_snapshot`). The routes
here restore that state, so the labeler never has to decide itself
whether "undo" means re-applying an earlier label or reverting to the
machine suggestion — the backend knows what the item looked like.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from src.clients.occ import OCCFinalConflictError
from src.clients.occ_bulk import failed_entry
from src.routers._errors import server_error
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._crop_write_models import (
    CropDiscardBatchRequest,
    CropDiscardRequest,
    CropUndoBatchRequest,
)
from src.schemas.curation.label_ops import CropUndoBatchResponse
from src.services.curation import label_undo as svc
from src.services.curation.class_undo import undo_class_writes


if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _crop_errors(crop_id: str) -> Iterator[None]:
    """Map the service's typed failures to HTTP. Only a typed not-found is a 404; a concurrent
    write that outlasts the retries is left to the app's 409 handler."""
    try:
        yield
    except svc.CropNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}') from exc
    except svc.NothingToUndoError as exc:
        raise HTTPException(status_code=409, detail=f'nothing to undo for crop {crop_id}') from exc
    except svc.NoVlmSuggestionError as exc:
        raise HTTPException(status_code=409, detail=f'no VLM suggestion on {crop_id}') from exc
    except OCCFinalConflictError:
        raise
    except Exception as exc:
        # Fail closed: anything else is the store being unavailable, never "crop not found".
        raise server_error(503, 'opensearch unavailable', exc) from exc


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
    with _crop_errors(crop_id):
        await svc.undo_one(opensearch, crop_id, require_history=True)
        return await svc.item_by_id(opensearch, crop_id)


@router.post('/crops/label/undo_batch', response_model=CropUndoBatchResponse)
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
    by_outcome = await undo_class_writes(opensearch, crop_ids)
    if (
        not by_outcome['undone']
        and by_outcome['nothing_to_undo']
        and not (by_outcome['conflicts'] or by_outcome['not_found'])
    ):
        raise HTTPException(status_code=409, detail='nothing to undo for any crop')
    return {
        'items': await svc.items_by_ids(opensearch, by_outcome['undone']),
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
    with _crop_errors(crop_id):
        await svc.undo_one(opensearch, crop_id, require_history=False)
    return {'crop_id': crop_id, 'reset': True}


def _require_effect(payload: CropDiscardRequest) -> None:
    if not (payload.clear_class or payload.dismiss_from_review):
        raise HTTPException(
            status_code=422, detail='discard needs clear_class and/or dismiss_from_review'
        )


@router.post('/crops/{crop_id}/discard')
async def discard_crop(
    crop_id: str, payload: CropDiscardRequest, opensearch: OpenSearchDep
) -> dict[str, Any]:
    """Discard an item, as a recorded human write.

    ``clear_class`` (default ``true``): the item doesn't belong in its
    class/cluster — class, provenance and validation are cleared and it
    drops to the residual pool (no ``cluster_uid``).
    ``dismiss_from_review`` (default ``false``): hide it from every
    ``/review`` tab (``review_dismissed_at``). The pre-write state is
    snapshotted, so ``POST /crops/{crop_id}/label/undo`` restores it
    exactly. Returns the post-write item. ``422`` when neither is set.
    """
    _require_effect(payload)
    with _crop_errors(crop_id):
        await svc.discard_one(
            opensearch,
            crop_id,
            clear_class=payload.clear_class,
            dismiss_from_review=payload.dismiss_from_review,
        )
        return await svc.item_by_id(opensearch, crop_id)


@router.post('/crops/discard_batch')
async def discard_crops(
    payload: CropDiscardBatchRequest, opensearch: OpenSearchDep
) -> dict[str, Any]:
    """Batch form of ``POST /crops/{crop_id}/discard``. Returns ``items``
    (post-write wire items), ``discarded``, ``conflicts`` / ``not_found`` id lists and
    ``failed`` (``{crop_id, reason: 'write_failed'}`` per item the write errored on). Undo with ``POST /crops/label/undo_batch``
    passing the discarded ids."""
    _require_effect(payload)
    by = await svc.discard_many(
        opensearch,
        list(dict.fromkeys(payload.crop_ids)),
        clear_class=payload.clear_class,
        dismiss_from_review=payload.dismiss_from_review,
    )
    return {
        'items': await svc.items_by_ids(opensearch, by['discarded']),
        'discarded': len(by['discarded']),
        'conflicts': by['conflicts'],
        'not_found': by['not_found'],
        'failed': [failed_entry(i) for i in by['failed']],
    }


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
    with _crop_errors(crop_id):
        await svc.dismiss_vlm_suggestion(opensearch, crop_id)
        return await svc.item_by_id(opensearch, crop_id)


@router.get('/crops/{crop_id}/history')
async def crop_history(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """The item's class history, oldest first: ``{crop_id, entries}``.

    Each entry is the item's class state *before* one write
    (``class_id``, ``class_name``, ``class_source``, ``label_source``,
    ``confidence``, ``class_detector*``, ``class_labeler``,
    ``class_labeled_at``, ``class_validated``, the item's ``cluster_*``
    placement; ``review_dismissed_*`` on discards) plus ``writer``
    (who made that write, e.g. ``human:label_crop``, ``vlm_pipeline``) and
    ``at``. Keys a writer didn't record are ``null``.
    """
    with _crop_errors(crop_id):
        return {'crop_id': crop_id, 'entries': await svc.crop_history(opensearch, crop_id)}


@router.post('/crops/{crop_id}/review_undismiss')
async def review_undismiss(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Return an item hidden from review to the queues (clears
    ``review_dismissed_at`` / ``review_dismissed_by``). For a dismissal made
    by ``POST /crops/{id}/discard``, ``label/undo`` does the same and also
    restores the class. Returns the post-write item."""
    with _crop_errors(crop_id):
        await svc.review_undismiss(opensearch, crop_id)
        return await svc.item_by_id(opensearch, crop_id)
