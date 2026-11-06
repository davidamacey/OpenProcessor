"""The ingest service bound to the running app's encoder and Triton pool."""

from __future__ import annotations

from typing import Any

from src.core.exceptions import RequestRejectedError
from src.services.curation.ingest import CurationIngestService
from src.services.curation.ingest_detector import (
    IngestProfileUnresolvedError,
    refreshed_ingest_profiles,
)
from src.services.curation.ingest_policy_store import get_ingest_policy


async def get_ingest_service(opensearch: Any, registry: Any) -> CurationIngestService:
    """Refuses with 503 until the app has started and a detector is configured."""
    from src.app.resources import get_async_triton_pool
    from src.main import app

    pe_encoder = getattr(app.state, 'pe_encoder', None)
    if pe_encoder is None:
        raise RequestRejectedError(
            status_code=503,
            detail='PE encoder not initialized; ingest is unavailable until app startup completes',
        )
    try:
        profiles = await refreshed_ingest_profiles()
    except (ValueError, IngestProfileUnresolvedError) as exc:
        raise RequestRejectedError(status_code=503, detail=f'ingest misconfigured: {exc}') from exc
    profile, secondary = profiles.primary, profiles.secondary
    policy = await get_ingest_policy()
    if not profile.detector_model:
        raise RequestRejectedError(
            status_code=503,
            detail=(
                'No detector configured for ingest — activate an ingest profile for this project '
                '(POST /curation/projects/{project}/ingest_profiles/{name}/activate) or set '
                'OP_INGEST_PRIMARY_DETECTOR_MODEL to a Triton model name that serves item '
                'proposals for this deployment'
            ),
        )
    return CurationIngestService(
        opensearch=opensearch,
        triton_pool=get_async_triton_pool(),
        registry=await registry.snapshot(),
        profile=profile,
        secondary_profile=secondary,
        pe_encoder=pe_encoder,
        policy=policy,
    )
