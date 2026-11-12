"""Curation router sub-module: the accuracy audit.

``POST /audit/start`` draws a stratified sample of machine-labelled crops, the
``audit`` review queue (``GET /review/audit``) lists the drawn crops still waiting for a
human (who labels them through the normal label routes) and ``GET /audit/report`` reports
how often the detector and the VLM agreed with those human labels. See
:mod:`src.services.curation.audit`.
"""

from __future__ import annotations

from typing import Any

from fastapi import Query

from src.routers.curation._api_error import api_error
from src.routers.curation._audit_models import AuditReport, AuditStartRequest, AuditStartResponse
from src.routers.curation._common import OpenSearchDep, router
from src.schemas.curation.config_common import ApiErrorResponse
from src.services.curation.audit import DEFAULT_MIN_PER_CLASS, load_report, start_audit


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
    return await load_report(opensearch, min_per_class=min_per_class)
