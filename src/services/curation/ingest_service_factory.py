"""The ingest service bound to the running app's encoder and Triton pool."""

from __future__ import annotations

from typing import Any

from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile
from src.core.exceptions import RequestRejectedError
from src.services.curation.ingest import CurationIngestService
from src.services.curation.ingest_detector import project_ingest_profile


async def get_ingest_service(opensearch: Any, registry: Any) -> CurationIngestService:
    """Refuses with 503 until the app has started and a detector is configured."""
    from src.main import app, get_async_triton_pool

    pe_encoder = getattr(app.state, 'pe_encoder', None)
    if pe_encoder is None:
        raise RequestRejectedError(
            status_code=503,
            detail='PE encoder not initialized; ingest is unavailable until app startup completes',
        )
    try:
        base_profile = ingest_primary_profile()
        secondary = ingest_secondary_profile()
    except ValueError as exc:
        raise RequestRejectedError(status_code=503, detail=f'ingest misconfigured: {exc}') from exc
    profile, policy = await project_ingest_profile(opensearch, base_profile)
    if not profile.detector_model:
        raise RequestRejectedError(
            status_code=503,
            detail=(
                'No detector configured for ingest — set OP_INGEST_PRIMARY_DETECTOR_MODEL '
                'to a Triton model name that serves item proposals for this deployment'
            ),
        )
    return CurationIngestService(
        opensearch=opensearch,
        triton_pool=get_async_triton_pool(),
        registry=registry,
        profile=profile,
        secondary_profile=secondary,
        pe_encoder=pe_encoder,
        policy=policy,
    )
