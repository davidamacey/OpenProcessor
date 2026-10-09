"""Curation router sub-module: the accuracy audit.

``POST /audit/start`` draws a stratified sample of machine-labelled crops,
``GET /audit/queue`` lists the drawn crops still waiting for a human (who labels
them through the normal label routes) and ``GET /audit/report`` reports how often
the detector and the VLM agreed with those human labels. See
:mod:`src.services.curation.audit`.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from src.routers.curation._audit_models import (
    AuditQueueResponse,
    AuditReport,
    AuditStartRequest,
    AuditStartResponse,
)
from src.routers.curation._common import OpenSearchDep, _ensure_indexes, items_index, router
from src.routers.curation._config_common_models import ApiErrorResponse, api_error
from src.services.curation.audit import (
    DEFAULT_MIN_PER_CLASS,
    load_report,
    pending_query,
    start_audit,
)
from src.services.curation.wire import item_list_source_excludes, serialize_item


@router.post(
    '/audit/start',
    response_model=AuditStartResponse,
    responses={409: {'model': ApiErrorResponse}},
)
async def audit_start(body: AuditStartRequest, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Draw a sample. Eligible crops carry a machine class and the detector's answer
    and are not validated, held out, excluded, dismissed, imported suggestions or
    already drawn. ``409 audit_no_candidates`` when none are eligible. Marking a crop
    never changes its class."""
    await _ensure_indexes(opensearch)
    result = await start_audit(
        opensearch, min_per_class=body.min_per_class, sample_size=body.sample_size
    )
    if result['sampled'] == 0:
        raise api_error(
            409,
            'audit_no_candidates',
            'no machine-labelled crop with a detector answer is eligible',
        )
    return result


@router.get('/audit/queue', response_model=AuditQueueResponse)
async def audit_queue(
    opensearch: OpenSearchDep,
    batch_id: str | None = Query(None, description='Only this batch.'),
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=200),
) -> dict[str, Any]:
    """Drawn crops still waiting for a human label, in a stable order."""
    await _ensure_indexes(opensearch)
    try:
        resp = await opensearch.search(
            index=items_index(),
            body={
                'from': (page - 1) * page_size,
                'size': page_size,
                'query': pending_query(batch_id),
                'sort': [{'crop_id': 'asc'}],
                'track_total_hits': True,
                '_source': {'excludes': item_list_source_excludes()},
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    hits = (resp.get('hits') or {}).get('hits') or []
    return {
        'total': int(((resp.get('hits') or {}).get('total') or {}).get('value', 0)),
        'page': page,
        'page_size': page_size,
        'items': [serialize_item(h.get('_source') or {}, h.get('_id', '')) for h in hits],
    }


@router.get('/audit/report', response_model=AuditReport)
async def audit_report(
    opensearch: OpenSearchDep,
    min_per_class: int = Query(
        DEFAULT_MIN_PER_CLASS, ge=1, le=1000, description='Audited crops a class needs.'
    ),
) -> dict[str, Any]:
    """Per-class precision of the detector and the VLM with Wilson 95% intervals, the
    confusion matrix and the outcome counts. A class with fewer than ``min_per_class``
    audited crops is flagged ``insufficient_sample``."""
    await _ensure_indexes(opensearch)
    return await load_report(opensearch, min_per_class=min_per_class)
