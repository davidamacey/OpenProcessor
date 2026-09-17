"""Curation auto-label job control endpoints (status + cancel).

Split out of :mod:`pipeline` to keep that module under the 700-LOC hook
ceiling. Side-effect import: registers ``@router`` handlers on the shared
``_common.router``.
"""

from __future__ import annotations

from typing import Any

from src.routers.curation._common import router


@router.get('/pipeline/auto_label/status')
async def pipeline_auto_label_status() -> dict[str, Any]:
    """Poll the current/last auto_label job state."""
    from src.services.curation.autolabel import job as auto_label_job

    return auto_label_job.get_state()


@router.post('/pipeline/auto_label/cancel')
async def pipeline_auto_label_cancel() -> dict[str, Any]:
    """Request cancellation of the active auto_label job."""
    from src.services.curation.autolabel import job as auto_label_job

    cancelled = auto_label_job.cancel_job()
    return {'cancelled': cancelled, **auto_label_job.get_state()}
