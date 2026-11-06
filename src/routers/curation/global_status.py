"""Global, project-less ``{api_prefix}/health`` and ``{api_prefix}/events``
(review delta 1), for screens outside any project (the project list, the
create form, the combine wizard).

Registered on the projects ``global_router``: they answer with nothing
bound. The project's own ``/health`` and ``/events`` live under
``{api_prefix}/projects/{project}``. The global stream carries only
``project: null`` events (``project.*``, ``combine.*``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import Header, Query
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel, Field

from src.config import get_settings
from src.core.dependencies import AsyncTritonDep  # noqa: TC001  # FastAPI resolves the annotation at runtime
from src.routers.curation._event_streams import GlobalSubscriptionDep, to_sse
from src.routers.curation.health import triton_status, vlm_status
from src.routers.curation.projects import global_router
from src.services.events.sse import event_stream
from src.services.projects.guard import make_curation_opensearch
from src.services.resource_links import service_url


class GlobalHealthResponse(BaseModel):
    """Deployment facts only -- nothing here depends on a project. The
    scoped ``{prefix}/health`` serves the same facts plus the bound
    project's."""

    status: Literal['ok', 'degraded', 'down']
    triton: dict[str, Any]
    opensearch: dict[str, Any] = Field(description='``{reachable, detail?}``; no index facts.')
    vlm: dict[str, Any]
    mlflow_public_url: str | None = None
    version: str
    api_version: str


async def _opensearch_reachable() -> dict[str, Any]:
    status: dict[str, Any] = {'reachable': False}
    try:
        client = await make_curation_opensearch()
        status['reachable'] = bool(await client.ping())
    except Exception as exc:
        status['detail'] = str(exc)
    return status


@global_router.get('/health', response_model=GlobalHealthResponse)
async def global_health(triton_pool: AsyncTritonDep) -> GlobalHealthResponse:
    """Deployment health: Triton, OpenSearch reachability, VLM, versions."""
    triton = await triton_status(triton_pool)
    opensearch = await _opensearch_reachable()
    vlm = await vlm_status(scoped=False)

    overall: Literal['ok', 'degraded', 'down']
    if triton['reachable'] and opensearch['reachable']:
        overall = 'ok' if vlm['reachable'] else 'degraded'
    elif opensearch['reachable']:
        overall = 'degraded'
    else:
        overall = 'down'

    return GlobalHealthResponse(
        status=overall,
        triton=triton,
        opensearch=opensearch,
        vlm=vlm,
        mlflow_public_url=service_url('mlflow'),
        version=get_settings().api_version,
        api_version='v1',
    )


@global_router.get('/events', response_class=EventSourceResponse)
async def global_events(
    sub: GlobalSubscriptionDep,
    topic: str | None = Query(default=None, description='Optional topic filter.'),
    last_event_id: Annotated[
        str | None,
        Header(description='Resume point: the `id` of the last event the client received.'),
    ] = None,
) -> Any:
    """SSE stream of global events only (``project: null``): project
    lifecycle (``project.*``) and ``combine.*`` (``GLOBAL_EVENT_PREFIXES``).
    A ``combine.*`` event names the project it builds in ``target``. No
    project's own events are ever delivered here; a project's config
    changes (``config.changed``) ride that project's own stream. Frames carry
    ``id`` and resume through ``Last-Event-ID`` like the project stream."""
    async for frame in event_stream(sub, last_event_id, topic=topic):
        yield to_sse(frame)
