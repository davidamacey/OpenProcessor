"""Served ``resource_links`` (``GET /settings``): the operator's one list of
docs and monitoring/service UIs.

URL rule (one function, :func:`service_url`, shared with the MLflow run
link): a ``kind: 'service'`` URL is (1) the explicit ``OP_<X>_URL`` verbatim
when set (reverse-proxied deployments); else ``null`` when the service's
``OP_<X>_PORT`` is 0 (disabled); else (2) with ``OP_GATEWAY_SUBPATHS`` the
gateway sub-path (``/grafana/``, ``/prometheus/``, ``/dashboards/``,
``/mlflow/``), which needs no request context because the browser resolves it
against the origin it already uses; else (3) ``<scheme>://<host the client
used>:<service host port>`` from the request (``Host`` /
``X-Forwarded-Host`` / ``X-Forwarded-Proto``, see
:mod:`src.core.request_origin`), so a LAN client gets its own host, never
``localhost``; else ``null`` (no request context). A
``kind: 'docs'`` URL is PATH-RELATIVE to the API origin (``/docs``).

``reachable`` means "the service answers FROM THE SERVER" (probed at its
compose-internal address, 1 s timeout), not "your browser can reach it". The probe is the beat
task ``op.periodic.resource_reachability`` (:mod:`src.services.periodic.resource_reachability`):
it writes the answer to the Valkey key :data:`REACHABILITY_KEY` with a 90 s expiry and every API
process only reads it, so all processes serve the same answer. ``null`` = no answer (the tick has
not run, the key expired, Valkey is down) or no URL; never "down" by default.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import httpx
import redis
import redis.asyncio as aioredis
from pydantic import BaseModel

from src.config.queue import QueueSettings
from src.core.logging import get_logger
from src.core.request_origin import current_origin
from src.tasks.runtime.event_wakeup import client_kwargs


if TYPE_CHECKING:
    from collections.abc import Mapping

logger = get_logger(__name__)

_PROBE_TIMEOUT_S = 1.0
REACHABILITY_KEY = 'op:resource_reachability'
REACHABILITY_TTL_S = 90
"""Three beat ticks (30 s): one missed tick keeps the answer, a dead beat lets it expire."""
_VALKEY_TIMEOUT_S = 1.0


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
    gateway_path: str  # sub-path the frontend nginx serves it under (gateway mode)


_DOCS = (
    ('swagger', 'Swagger UI', '/docs', 'Interactive API docs served by the API.'),
    ('redoc', 'ReDoc', '/redoc', 'Reference API docs served by the API.'),
    ('openapi_json', 'OpenAPI JSON', '/openapi.json', 'Machine-readable API schema.'),
)

# Probe URLs are the compose service names/container ports.
_SERVICES = (
    _Service(
        'grafana', 'Grafana', 'grafana', 'GRAFANA', 'http://grafana:3000/api/health', '/grafana/'
    ),
    _Service(
        'prometheus',
        'Prometheus',
        'prometheus',
        'PROMETHEUS',
        'http://prometheus:9090/-/healthy',
        '/prometheus/',
    ),
    _Service(
        'opensearch_dashboards',
        'OpenSearch Dashboards',
        'dashboards',
        'DASHBOARDS',
        'http://opensearch-dashboards:5601/api/status',
        '/dashboards/',
    ),
    _Service(
        'mlflow', 'MLflow', 'mlflow', 'MLFLOW', 'http://curation-mlflow:5000/health', '/mlflow/'
    ),
)


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
        typed: str | None = explicit
        return typed
    port = getattr(cfg, f'{spec.name}_port')
    if not port:
        return None
    if cfg.gateway_subpaths:
        return spec.gateway_path
    origin = current_origin()
    if origin is None:
        return None
    return f'{origin.scheme}://{origin.host}:{port}'


def _enabled(spec: _Service) -> bool:
    """Has a URL source at all (explicit URL or a non-zero port), request or not."""
    from src.config import get_curation_config

    cfg = get_curation_config()
    return bool(getattr(cfg, _url_attr(spec)) or getattr(cfg, f'{spec.name}_port'))


def valkey_client() -> aioredis.Redis:
    """A Valkey client for one read or write (the caller closes it)."""
    settings = QueueSettings.from_env()
    return aioredis.Redis.from_url(
        settings.url,
        socket_timeout=_VALKEY_TIMEOUT_S,
        socket_connect_timeout=_VALKEY_TIMEOUT_S,
        **client_kwargs(settings),
    )


async def probe_enabled() -> dict[str, bool]:
    """Probe each enabled service once, one after the other (a beat task, not a request)."""
    return {s.id: await _probe(s.probe_url) for s in _SERVICES if _enabled(s)}


async def publish_reachability(reachable: Mapping[str, bool], *, valkey: Any = None) -> None:
    """``SET op:resource_reachability <json> EX 90`` (https://valkey.io/commands/set/)."""
    client = valkey_client() if valkey is None else valkey
    try:
        await client.set(REACHABILITY_KEY, json.dumps(dict(reachable)), ex=REACHABILITY_TTL_S)
    finally:
        await client.aclose()


async def read_reachability() -> dict[str, bool]:
    """The last published answer; ``{}`` (unknown) when there is none or Valkey does not answer."""
    client = valkey_client()
    try:
        raw = await client.get(REACHABILITY_KEY)
    except (redis.RedisError, OSError) as exc:
        logger.warning('resource_reachability_unreadable', error=str(exc))
        return {}
    finally:
        await client.aclose()
    if raw is None:
        return {}
    answer = json.loads(raw)
    return {str(k): bool(v) for k, v in answer.items()}


def resource_links_from_config(reachable: Mapping[str, bool]) -> list[ResourceLink]:
    from src.config import get_curation_config

    gateway = get_curation_config().gateway_subpaths
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
                    f'Set {_url_env(s)} to override; served by the gateway at {s.gateway_path}.'
                    if url == s.gateway_path and gateway
                    else f'Set {_url_env(s)} to override; default is this host on OP_{s.env}_PORT.'
                    if url
                    else f'Set OP_{s.env}_PORT (0 disables) or {_url_env(s)}.'
                ),
                reachable=reachable.get(s.id) if url else None,
            )
        )
    return links
