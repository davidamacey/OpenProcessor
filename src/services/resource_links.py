"""Served ``resource_links`` (``GET /settings``): the operator's one list of
docs and monitoring/service UIs.

URL rule (one function, :func:`service_url`, shared with the MLflow run
link): a ``kind: 'service'`` URL is (1) the explicit ``OP_<X>_URL`` verbatim
when set (reverse-proxied deployments); else (2) ``<scheme>://<host the
client used>:<service host port>`` from the request (``Host`` /
``X-Forwarded-Host`` / ``X-Forwarded-Proto``, see
:mod:`src.core.request_origin`), so a LAN client gets its own host, never
``localhost``; else ``null`` (no request context, or the port is 0). A
``kind: 'docs'`` URL is PATH-RELATIVE to the API origin (``/docs``).

``reachable`` means "the service answers FROM THE SERVER" (probed at its
compose-internal address, 1 s timeout, background task cached ``_TTL_S``),
not "your browser can reach it". ``null`` = not probed yet / no URL.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal, NamedTuple

import httpx
from pydantic import BaseModel

from src.core.request_origin import current_origin


_TTL_S = 15.0
_PROBE_TIMEOUT_S = 1.0


class ResourceLink(BaseModel):
    id: str
    label: str
    url: str | None
    kind: Literal['service', 'docs']
    status: Literal['configured', 'not_configured']
    hint: str
    reachable: bool | None = None


class _Service(NamedTuple):
    id: str
    label: str
    name: str  # CurationConfig attribute stem: <name>_url / <name>_port
    env: str  # OP_<env>_URL / OP_<env>_PORT
    probe_url: str  # compose-internal address


_DOCS = (
    ('swagger', 'Swagger UI', '/docs', 'Interactive API docs served by the API.'),
    ('redoc', 'ReDoc', '/redoc', 'Reference API docs served by the API.'),
    ('openapi_json', 'OpenAPI JSON', '/openapi.json', 'Machine-readable API schema.'),
)

# Probe URLs are the compose service names/container ports.
_SERVICES = (
    _Service('grafana', 'Grafana', 'grafana', 'GRAFANA', 'http://grafana:3000/api/health'),
    _Service(
        'prometheus', 'Prometheus', 'prometheus', 'PROMETHEUS', 'http://prometheus:9090/-/healthy'
    ),
    _Service(
        'opensearch_dashboards',
        'OpenSearch Dashboards',
        'dashboards',
        'DASHBOARDS',
        'http://opensearch-dashboards:5601/api/status',
    ),
    _Service('mlflow', 'MLflow', 'mlflow', 'MLFLOW', 'http://curation-mlflow:5000/health'),
)


class _Cache:
    def __init__(self) -> None:
        self.reachable: dict[str, bool] = {}
        self.checked_at = 0.0
        self.inflight: asyncio.Task[None] | None = None


_cache = _Cache()


def reset_reachability_cache() -> None:
    _cache.__init__()  # type: ignore[misc]


async def _probe(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as client:
            return (await client.get(url)).status_code < 500
    except httpx.HTTPError:
        return False


def _url_attr(spec: _Service) -> str:
    return 'mlflow_public_url' if spec.id == 'mlflow' else f'{spec.name}_url'


def _url_env(spec: _Service) -> str:
    return 'OP_MLFLOW_PUBLIC_URL' if spec.id == 'mlflow' else f'OP_{spec.env}_URL'


def _spec(service_id: str) -> _Service:
    return next(s for s in _SERVICES if s.id == service_id)


def service_url(service_id: str) -> str | None:
    """The browser URL for a service id (the URL rule in the module doc)."""
    from src.config import get_curation_config

    cfg = get_curation_config()
    spec = _spec(service_id)
    explicit = getattr(cfg, _url_attr(spec))
    if explicit:
        return explicit
    port = getattr(cfg, f'{spec.name}_port')
    origin = current_origin()
    if not port or origin is None:
        return None
    return f'{origin.scheme}://{origin.host}:{port}'


def _enabled(spec: _Service) -> bool:
    """Has a URL source at all (explicit URL or a non-zero port), request or not."""
    from src.config import get_curation_config

    cfg = get_curation_config()
    return bool(getattr(cfg, _url_attr(spec)) or getattr(cfg, f'{spec.name}_port'))


async def refresh_reachability() -> None:
    targets = [s for s in _SERVICES if _enabled(s)]
    results = await asyncio.gather(*(_probe(s.probe_url) for s in targets))
    _cache.reachable = {s.id: ok for s, ok in zip(targets, results, strict=True)}
    _cache.checked_at = time.monotonic()


def schedule_reachability_refresh() -> None:
    """Kick a background refresh when the cache is stale; never awaits."""
    if _cache.inflight is not None and not _cache.inflight.done():
        return
    if _cache.checked_at and time.monotonic() - _cache.checked_at < _TTL_S:
        return
    _cache.inflight = asyncio.get_running_loop().create_task(refresh_reachability())


async def wait_for_inflight_refresh() -> None:
    if _cache.inflight is not None:
        await _cache.inflight


def resource_links_from_config() -> list[ResourceLink]:
    links = [
        ResourceLink(id=id_, label=label, url=path, kind='docs', status='configured', hint=hint)
        for id_, label, path, hint in _DOCS
    ]
    for s in _SERVICES:
        url = service_url(s.id)
        links.append(
            ResourceLink(
                id=s.id,
                label=s.label,
                url=url,
                kind='service',
                status='configured' if url else 'not_configured',
                hint=(
                    f'Set {_url_env(s)} to override; default is this host on OP_{s.env}_PORT.'
                    if url
                    else f'Set OP_{s.env}_PORT (0 disables) or {_url_env(s)}.'
                ),
                reachable=_cache.reachable.get(s.id) if url else None,
            )
        )
    return links
