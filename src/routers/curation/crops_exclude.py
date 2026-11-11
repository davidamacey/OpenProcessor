"""Curation router sub-module — item (crop) exclude, un-exclude and review-dismiss."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.routers._errors import server_error
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._crop_write_models import CropExcludeRequest, CropUnexcludeRequest
from src.routers.curation._selection import selected_crop_ids
from src.routers.curation._selection_write_models import (
    BatchExcludeResponse,
    BatchUnexcludeResponse,
    SelectionDryRunResponse,
)
from src.schemas.curation.exclusion import ExcludeReasonList, exclude_reason_list
from src.services.curation.exclusion import exclude_items, unexclude_items
from src.services.curation.label_undo import CropNotFoundError, review_dismiss


@router.get('/exclude_reasons', response_model=ExcludeReasonList, tags=['Crops'])
async def list_exclude_reasons() -> ExcludeReasonList:
    """The reasons an item can be excluded for, in display order, with their labels."""
    return exclude_reason_list()


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
    ``excluded_by``, ``excluded_reason``); the crop keeps its place in the
    serving partition. The
    residual-pool fetch, the trainer export, the cluster listing, and
    the default crop browse all filter ``class_excluded=true`` out, so
    an excluded crop never re-appears in a cluster or training set until
    it's un-excluded.

    Non-destructive: the crop document stays in OpenSearch for audit and
    provenance, and the pre-exclusion validation is
    recorded (``excluded_prior_*``) so un-exclude can restore them.
    ``reason`` defaults to ``'ignore'``; pass a tag like
    ``'blurry'`` to record why (e.g. a whole cluster of blurry items).
    """
    ids = await selected_crop_ids(opensearch, payload)
    if payload.dry_run:
        return {'dry_run': True, 'selected': len(ids)}
    if not ids:
        return {'excluded': 0, 'updated_ids': [], 'errors': 0}
    try:
        updated_ids, n_errors = await exclude_items(
            opensearch, ids, reason=payload.reason or 'ignore'
        )
    except Exception as exc:
        raise server_error(503, 'opensearch error', exc) from exc
    return {'excluded': len(updated_ids), 'updated_ids': updated_ids, 'errors': n_errors}


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
    recorded at exclude time; the crop never left its cluster. Crops that aren't excluded
    are left untouched.
    """
    ids = await selected_crop_ids(opensearch, payload)
    if payload.dry_run:
        return {'dry_run': True, 'selected': len(ids)}
    if not ids:
        return {'unexcluded': 0, 'updated_ids': [], 'errors': 0}
    try:
        updated_ids, n_errors = await unexclude_items(opensearch, ids)
    except Exception as exc:
        raise server_error(503, 'opensearch error', exc) from exc
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
    try:
        await review_dismiss(opensearch, crop_id)
    except CropNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}') from exc
    except Exception as exc:
        raise server_error(503, 'opensearch error', exc) from exc
    return {'crop_id': crop_id, 'dismissed': True}
