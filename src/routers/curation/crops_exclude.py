"""Curation router sub-module — item (crop) exclude, un-exclude and review-dismiss."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.routers.curation._common import OpenSearchDep, items_index, router
from src.routers.curation._common_models import CropExcludeRequest, CropUnexcludeRequest
from src.routers.curation._selection import selected_crop_ids
from src.routers.curation._selection_write_models import (
    BatchExcludeResponse,
    BatchUnexcludeResponse,
    SelectionDryRunResponse,
)
from src.utils.timeutil import now_iso


@router.post(
    '/crops/batch_exclude',
    response_model=None,
    responses={200: {'model': BatchExcludeResponse | SelectionDryRunResponse}},
)
async def batch_exclude_crops(
    payload: CropExcludeRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Exclude crops from training + clustering (reversible).

    Sets ``class_excluded=true`` plus provenance (``excluded_at``,
    ``excluded_by``, ``excluded_reason``) and moves the crop out of its
    candidate bucket (``cluster_id=-2``, the excluded sentinel). The
    residual-pool fetch, the trainer export, the cluster listing, and
    the default crop browse all filter ``class_excluded=true`` out, so
    an excluded crop never re-appears in a cluster or training set until
    it's un-excluded.

    Non-destructive: the crop document stays in OpenSearch for audit and
    provenance, and the pre-exclusion validation + cluster placement are
    recorded (``excluded_prior_*``) so un-exclude can restore them.
    ``reason`` defaults to ``'ignore'``; pass a tag like
    ``'blurry'`` to record why (e.g. a whole cluster of blurry items).
    """
    ids = await selected_crop_ids(opensearch, payload)
    if payload.dry_run:
        return {'dry_run': True, 'selected': len(ids)}
    if not ids:
        return {'excluded': 0, 'updated_ids': [], 'errors': 0}
    from src.services.curation.exclusion import exclusion_update

    now = now_iso()
    reason = payload.reason or 'ignore'
    updated_ids, n_errors = await _occ_bulk_human_write(
        opensearch,
        ids,
        lambda _id, cur: exclusion_update(cur, reason=reason, now=now),
        writer_id='human:batch_exclude_crops',
    )
    return {'excluded': len(updated_ids), 'updated_ids': updated_ids, 'errors': n_errors}


async def _occ_bulk_human_write(
    opensearch: Any,
    crop_ids: list[str],
    merger: Any,
    *,
    writer_id: str,
) -> tuple[list[str], int]:
    """Read-modify-write ``crop_ids`` via OCC bulk; return ``(written ids,
    failure count)``.

    A version conflict (a concurrent write landed between read and write)
    counts as a failure the caller reports, never a silent success; an item
    the merger left unchanged is in neither.
    """
    from src.clients.occ import occ_skip_on_conflict_bulk

    try:
        resp = await occ_skip_on_conflict_bulk(
            opensearch,
            doc_ids=list(crop_ids),
            merger=merger,
            index=items_index(),
            refresh='wait_for',
            writer_id=writer_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc
    failures = len(resp.get('errors') or []) + int(resp.get('skipped_due_to_conflict') or 0)
    return list(resp.get('updated_ids') or []), failures


@router.post(
    '/crops/batch_unexclude',
    response_model=None,
    responses={200: {'model': BatchUnexcludeResponse | SelectionDryRunResponse}},
)
async def batch_unexclude_crops(
    payload: CropUnexcludeRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Reverse an exclusion (Undo path for Ignore).

    Clears ``class_excluded`` + provenance and restores the validation
    recorded at exclude time. A validated crop goes straight back to its
    class cluster (``cluster_id == class_id``); an unvalidated one returns
    to the candidate cluster it was excluded from while that cluster still
    has members, else drops to the residual pool (``cluster_id=null``) for
    a fresh assignment on the next recluster. Crops that aren't excluded
    are left untouched.
    """
    ids = await selected_crop_ids(opensearch, payload)
    if payload.dry_run:
        return {'dry_run': True, 'selected': len(ids)}
    if not ids:
        return {'unexcluded': 0, 'updated_ids': [], 'errors': 0}
    from src.services.curation.exclusion import live_candidate_ids, unexclusion_update

    try:
        live = await live_candidate_ids(opensearch, items_index(), ids)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc
    now = now_iso()
    updated_ids, n_errors = await _occ_bulk_human_write(
        opensearch,
        ids,
        lambda _id, cur: unexclusion_update(cur, now=now, live_candidate_ids=live),
        writer_id='human:batch_unexclude_crops',
    )
    return {'unexcluded': len(updated_ids), 'updated_ids': updated_ids, 'errors': n_errors}


@router.post('/crops/{crop_id}/review_dismiss')
async def review_dismiss_crop(crop_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Permanently dismiss a crop from every /review queue.

    Stamps ``review_dismissed_at`` + ``review_dismissed_by``. The
    review_queue handler excludes any crop where ``review_dismissed_at``
    exists, so this is one-way: once an operator says 'I never want to
    see this again in review', it's gone from every tab. The crop's
    underlying class / region state is left intact — only the review
    visibility changes.
    """
    body = {
        'doc': {
            'review_dismissed_at': now_iso(),
            'review_dismissed_by': 'human',
            'updated_at': now_iso(),
        }
    }
    try:
        await opensearch.update(
            index=items_index(),
            id=crop_id,
            body=body,
            refresh='wait_for',
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    return {'crop_id': crop_id, 'dismissed': True}
