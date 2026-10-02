"""``/curation/vlm/catalog`` and ``/vlm/local*``: the local model catalog and
the in-compose vLLM's desired model (W9.7). Registered on the projects
``global_router`` (deployment-wide, no project bound).

The API never restarts vLLM: it serves one model per process and changing it
means recreating the container with new arguments, a host-side compose
operation. So ``POST /vlm/local/select`` only records the DESIRED model and
returns ``restart_required`` plus the host command; ``serving`` flips only
after the host command ran and a probe recorded the new model root.
"""

from __future__ import annotations

import os
from typing import Any

from src.routers.curation._common import OpenSearchDep  # noqa: TC001 - FastAPI resolves it
from src.routers.curation._config_common_models import api_error
from src.routers.curation._vlm_endpoint_models import (
    CATALOG_STATUS_LABELS,
    Choice,
    VlmCatalogEntry,
    VlmCatalogLabels,
    VlmCatalogResponse,
    VlmLocalSelectRequest,
    VlmLocalStatus,
)
from src.routers.curation.projects import global_router
from src.services.config_store import get_global_config_store
from src.services.config_store.vlm_endpoints import clear_local_desired, set_local_desired
from src.services.labeling.vlm_catalog import (
    CatalogEntry,
    catalog_entry,
    fits,
    gpu_total_gb_from_env,
    load_catalog,
    local_vlm_status,
)
from src.services.labeling.vlm_endpoints import get_vlm_endpoint


def _local_endpoint_name() -> str:
    return os.environ.get('OP_LOCAL_VLM_ENDPOINT', '').strip()


async def _local_status(client: Any) -> VlmLocalStatus:
    store = get_global_config_store()
    await store.refresh(client)
    name = _local_endpoint_name()
    endpoint = get_vlm_endpoint(name) if name else None
    probe = endpoint.last_probe if endpoint else None
    return VlmLocalStatus.model_validate(
        local_vlm_status(
            endpoint_name=name if endpoint else '',
            served_model=endpoint.body.model if endpoint else None,
            served_root=probe.root if probe else None,
            served_max_model_len=probe.max_model_len if probe else None,
            desired=store.current.local_vlm_desired,
            gpu_total_gb=gpu_total_gb_from_env(os.environ.get('OP_LOCAL_VLM_GPU_TOTAL_MIB')),
        )
    )


def _entry_wire(entry: CatalogEntry, local: VlmLocalStatus) -> VlmCatalogEntry:
    return VlmCatalogEntry(
        id=entry.id,
        choice=Choice(id=entry.id, label=entry.hf_repo),
        hf_repo=entry.hf_repo,
        family=entry.family,
        license=entry.license,
        license_url=entry.license_url,
        gated=entry.gated,
        params_b=entry.params_b,
        quantization=entry.quantization,
        context_max=entry.context_max,
        max_model_len=entry.max_model_len,
        max_images=entry.max_images,
        vram_gb=entry.vram_gb,
        disk_gb=entry.disk_gb,
        status=entry.status,  # type: ignore[arg-type]
        rank=entry.rank,
        multi_box_verified=entry.multi_box_verified,
        text_reading_verified=entry.text_reading_verified,
        fits=fits(entry, local.gpu_total_gb),
        serving=bool(local.served and local.served.root == entry.hf_repo),
        desired=bool(local.desired and local.desired.catalog_id == entry.id),
    )


@global_router.get('/vlm/catalog', response_model=VlmCatalogResponse, tags=['VLM'])
async def get_vlm_catalog(client: OpenSearchDep) -> VlmCatalogResponse:
    """The local catalog with ``fits`` / ``serving`` / ``desired`` per entry."""
    local = await _local_status(client)
    return VlmCatalogResponse(
        entries=[_entry_wire(e, local) for e in load_catalog()],
        local=local,
        labels=VlmCatalogLabels(status=CATALOG_STATUS_LABELS),
    )


@global_router.get('/vlm/local', response_model=VlmLocalStatus, tags=['VLM'])
async def get_local_vlm(client: OpenSearchDep) -> VlmLocalStatus:
    return await _local_status(client)


@global_router.post(
    '/vlm/local/select', response_model=VlmLocalStatus, status_code=202, tags=['VLM']
)
async def select_local_vlm(payload: VlmLocalSelectRequest, client: OpenSearchDep) -> VlmLocalStatus:
    """Record the desired local model (202): the host applies it with
    ``openprocessor vlm use <id>`` (``restart_required`` says so)."""
    local = await _local_status(client)
    if not local.configured:
        raise api_error(409, 'no_local_vlm', 'this deployment has no in-compose VLM to switch')
    entry = catalog_entry(payload.catalog_id)
    if entry is None:
        raise api_error(
            422,
            'unknown_catalog_id',
            f'{payload.catalog_id!r} is not in the local catalog',
            valid_ids=[e.id for e in load_catalog()],
        )
    if fits(entry, local.gpu_total_gb) is False and not payload.force:
        raise api_error(
            422,
            'vlm_catalog_does_not_fit',
            f'{entry.id} needs about {entry.vram_gb:g} GB; the card has {local.gpu_total_gb:g} GB',
        )
    await set_local_desired(client, catalog_id=entry.id)
    return await _local_status(client)


@global_router.delete('/vlm/local/select', response_model=VlmLocalStatus, tags=['VLM'])
async def clear_local_vlm_selection(client: OpenSearchDep) -> VlmLocalStatus:
    """Clear the desired local model."""
    await clear_local_desired(client)
    return await _local_status(client)
