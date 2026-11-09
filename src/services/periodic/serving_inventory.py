"""``op.periodic.serving_inventory``: what each Triton server of the active profile really serves.

Every minute the tick reads each server of the active ``serving_profile`` (Triton's
``/v2/repository/index`` for the model states, the Prometheus metrics for the cards it names) and
writes ``serving_reports`` (one row per server, model and version; a model gone from the index is
deleted) and refreshes ``gpu_devices.seen_at`` for each card the metrics carry in a ``gpu_uuid``
label. A server that does not answer has all its rows marked ``unavailable``: the router sends
nothing to a server whose report lacks the model, so an unreachable one fails closed.

Triton's ports are those of its image (HTTP 8000); the profile names the gRPC and metrics
addresses, so the HTTP address is the gRPC host on port 8000. Reads use ``httpx`` and the
metrics text is parsed with ``prometheus_client.parser``
(https://prometheus.github.io/client_python/parser/).
"""

from __future__ import annotations

from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families
from sqlalchemy import text

from src.core.logging import get_logger
from src.db.session import async_session_factory, deployment_tx
from src.services.config_store.research_kinds import active_serving_profile


logger = get_logger(__name__)

HTTP_PORT = 8000
TIMEOUT_S = 5.0
_STATES = {'READY': 'ready', 'LOADING': 'loading', 'UNLOADING': 'unloading'}

_UPSERT = text(
    'INSERT INTO serving_reports (server, model, version, state, instances, seen_at) '
    'VALUES (:server, :model, :version, :state, :instances, now()) '
    'ON CONFLICT (server, model, version) DO UPDATE SET state = excluded.state, '
    'instances = excluded.instances, seen_at = now()'
)
_DELETE_GONE = text(
    "DELETE FROM serving_reports WHERE server = :server AND NOT (model || '/' || version) "
    '= ANY(:present)'
)
_UNAVAILABLE = text(
    "UPDATE serving_reports SET state = 'unavailable', seen_at = now() WHERE server = :server"
)
_SEEN = text('UPDATE gpu_devices SET seen_at = now() WHERE gpu_uuid = ANY(:uuids)')


def _host(address: str) -> str:
    return address.rsplit(':', 1)[0]


def gpu_uuids(metrics_text: str) -> set[str]:
    """The ``gpu_uuid`` label values of every sample in a Prometheus text exposition."""
    return {
        sample.labels['gpu_uuid']
        for family in text_string_to_metric_families(metrics_text)
        for sample in family.samples
        if 'gpu_uuid' in sample.labels
    }


async def _read_server(
    client: httpx.AsyncClient, server: dict[str, Any]
) -> tuple[list[dict[str, Any]], set[str]]:
    response = await client.post(
        f'http://{_host(server["grpc_url"])}:{HTTP_PORT}/v2/repository/index'
    )
    response.raise_for_status()
    index: list[dict[str, Any]] = response.json()
    metrics = await client.get(f'http://{server["metrics_url"]}/metrics')
    metrics.raise_for_status()
    return index, gpu_uuids(metrics.text)


async def tick(*, client: httpx.AsyncClient | None = None) -> dict[str, int]:
    """Refresh every server of the active profile; ``{'servers': n, 'unreachable': m}``."""
    active = await active_serving_profile()
    if active is None:
        return {'servers': 0, 'unreachable': 0}
    body = active[0].body
    owned = client is None
    http = client or httpx.AsyncClient(timeout=TIMEOUT_S)
    unreachable = 0
    try:
        for server in body['servers']:
            name = server['name']
            try:
                index, uuids = await _read_server(http, server)
            except (httpx.HTTPError, ValueError) as exc:
                unreachable += 1
                logger.warning('serving_inventory_unreachable', server=name, error=str(exc))
                async with async_session_factory()() as session, deployment_tx(session, 'serving'):
                    await session.execute(_UNAVAILABLE, {'server': name})
                continue
            rows = [
                {
                    'server': name,
                    'model': entry['name'],
                    'version': int(entry.get('version') or 0),
                    'state': _STATES.get(entry.get('state', ''), 'unavailable'),
                    'instances': body.get('instances', {}).get(entry['name']),
                }
                for entry in index
            ]
            async with async_session_factory()() as session, deployment_tx(session, 'serving'):
                if rows:
                    await session.execute(_UPSERT, rows)
                await session.execute(
                    _DELETE_GONE,
                    {'server': name, 'present': [f'{r["model"]}/{r["version"]}' for r in rows]},
                )
                if uuids:
                    await session.execute(_SEEN, {'uuids': sorted(uuids)})
    finally:
        if owned:
            await http.aclose()
    return {'servers': len(body['servers']), 'unreachable': unreachable}
