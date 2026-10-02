"""Curation health: the project-scoped ``{prefix}/health`` (on the shared
curation ``router``) and the probes it shares with the global, project-less
``{api_prefix}/health`` in :mod:`src.routers.curation.global_status`.

Which keys of the scoped response are the bound project's (review delta 1):

- ``project`` -- the bound project's slug;
- ``opensearch.indexes`` -- existence of the bound project's index set;
- ``registry`` -- the bound project's class registry file;
- ``region_profile`` -- the active region profile the project's
  region-scoped UI keys on.

Every other key (``status``'s Triton/VLM inputs, ``triton``,
``opensearch.reachable``, ``vlm``, ``mlflow_public_url``) is a deployment
fact, identical for every project and served on the global ``/health``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from src.config import get_curation_config
from src.core.dependencies import AsyncTritonDep  # noqa: TC001
from src.routers.curation._common import (
    HealthResponse,
    OpenSearchDep,
    classes_index,
    get_class_registry,
    images_index,
    items_index,
    labels_confirmed_index,
    router,
)
from src.routers.curation.vlm import _get_vlm_labeler


async def triton_status(triton: Any) -> dict[str, Any]:
    """``{reachable, detail}`` for the shared Triton client."""
    status: dict[str, Any] = {'reachable': False, 'detail': ''}
    try:
        # AsyncTritonDep is a single AsyncInferenceServerClient, not the pool.
        # Both have is_server_live; fall back to is_server_ready.
        if hasattr(triton, 'is_server_live'):
            ok = await triton.is_server_live()
        elif hasattr(triton, 'health_check'):
            ok = await triton.health_check()
        else:
            ok = False
        status['reachable'] = bool(ok)
    except Exception as exc:
        status['detail'] = str(exc)
    return status


async def vlm_status(client: Any = None, *, scoped: bool = True) -> dict[str, Any]:
    """``{reachable, model?, last_error?, detail?}`` for a VLM endpoint: the
    bound project's active one (``scoped``), or -- for the project-less
    global health -- the deployment's ``env`` built-in."""
    status: dict[str, Any] = {'reachable': False}
    try:
        from src.services.config_store import get_global_config_store
        from src.services.labeling.vlm_endpoints import (
            active_vlm_endpoint,
            env_builtin,
            refresh_vlm_state,
        )

        if scoped:
            await refresh_vlm_state(client)
            endpoint = active_vlm_endpoint()
        else:
            from src.services.projects.guard import make_curation_opensearch

            await get_global_config_store().ensure_fresh(client or await make_curation_opensearch())
            endpoint = env_builtin()
        if endpoint is None:
            status['detail'] = 'no VLM endpoint is configured'
            return status
        if scoped:
            labeler = _get_vlm_labeler(endpoint=endpoint)
        else:
            # A health probe sends no prompt, and the project-scoped pack
            # lookup `_get_vlm_labeler` does needs a bound project.
            from src.services.labeling.vlm_factory import labeler_for
            from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

            labeler = labeler_for(endpoint, GENERIC_ITEM_PACK)
        h = await labeler.health()
        status['reachable'] = h.reachable
        status['model'] = h.model
        if h.last_error:
            status['last_error'] = h.last_error
    except Exception as exc:
        status['detail'] = str(exc)
    return status


@router.get('/health', response_model=HealthResponse)
async def curation_health(
    opensearch: OpenSearchDep,
    triton_pool: AsyncTritonDep,
) -> HealthResponse:
    """Aggregated health for the bound project: Triton + OpenSearch (the
    project's indexes) + VLM + the project's registry file. See the
    module docstring for which keys are project-bound."""
    triton = await triton_status(triton_pool)

    # opensearch dep here is the project's wrapper. Reach the raw async client
    # via attributes commonly exposed; fall back to assuming `opensearch` IS
    # an AsyncOpenSearch.
    raw_os = getattr(opensearch, 'client', None) or opensearch

    os_status: dict[str, Any] = {'reachable': False, 'indexes': {}}
    try:
        for idx_name in (
            images_index(),
            items_index(),
            labels_confirmed_index(),
            classes_index(),
        ):
            os_status['indexes'][idx_name] = bool(await raw_os.indices.exists(index=idx_name))
        os_status['reachable'] = True
    except Exception as exc:
        os_status['detail'] = str(exc)

    vlm = await vlm_status(raw_os)

    reg_path = get_class_registry().path
    registry_status: dict[str, Any] = {
        'path': str(reg_path),
        'exists': reg_path.exists(),
    }
    if reg_path.exists():
        try:
            registry_status['mtime'] = datetime.fromtimestamp(
                reg_path.stat().st_mtime, tz=UTC
            ).isoformat()
        except OSError as exc:
            registry_status['detail'] = str(exc)

    overall: Literal['ok', 'degraded', 'down']
    if triton['reachable'] and os_status['reachable'] and registry_status.get('exists'):
        overall = 'ok' if vlm['reachable'] else 'degraded'
    elif os_status['reachable']:
        overall = 'degraded'
    else:
        overall = 'down'

    from src.services.curation.region_vocabulary import region_profile_summary
    from src.services.detection.profile_registry import get_active_region_profile

    active_profile = get_active_region_profile()
    region_profile = region_profile_summary(active_profile) if active_profile is not None else None

    cfg = get_curation_config()
    return HealthResponse(
        status=overall,
        project=cfg.project_slug,
        triton=triton,
        opensearch=os_status,
        vlm=vlm,
        registry=registry_status,
        region_profile=region_profile,
        mlflow_public_url=cfg.mlflow_public_url,
    )
