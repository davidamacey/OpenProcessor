"""How a bulk run sends one batch of paths through the ingest route.

The runner talks to a :class:`BatchSender`; in production that is
:class:`HttpBatchSender` aimed at the API's own listening socket, so batches land
on every ``api`` worker process rather than only the one that owns the run.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from prometheus_client.parser import text_string_to_metric_families

from src.services.curation.bulk_ingest.limits import BATCH_REQUEST_TIMEOUT_S, MAX_SENDER_CONNECTIONS


_STAGE_METRIC = 'op_pipeline_stage_seconds'


class BatchRequestError(Exception):
    """The ingest route refused or failed a batch as a whole."""


class BatchSender(Protocol):
    """Sends one batch of paths through the ingest route and returns its
    ``BatchIngestResponse`` body."""

    async def post_batch(self, paths: list[str], *, source: str, run_id: str) -> dict[str, Any]: ...

    async def stage_seconds(self) -> dict[str, float]:
        """Cumulative pipeline seconds per stage over all API workers (empty when
        unreadable)."""
        ...


class HttpBatchSender:
    """``POST {base}/projects/{project}/ingest/batch`` over HTTP. In production
    ``base`` is the API's own listening address, so batches land on every worker
    process; the route enforces the project binding, the source roots and the
    per-request size cap exactly as for any other client."""

    def __init__(self, client: httpx.AsyncClient, *, api_prefix: str, project: str) -> None:
        self._client = client
        self._batch_url = f'{api_prefix}/projects/{project}/ingest/batch'

    async def post_batch(self, paths: list[str], *, source: str, run_id: str) -> dict[str, Any]:
        try:
            resp = await self._client.post(
                self._batch_url,
                json={
                    'items': [{'path': p, 'source': source} for p in paths],
                    'ingest_run_id': run_id,
                },
                timeout=BATCH_REQUEST_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise BatchRequestError(f'{type(exc).__name__}: {exc}'[:300]) from exc
        if resp.status_code != 200:
            raise BatchRequestError(f'HTTP {resp.status_code}: {resp.text[:250]}')
        body: dict[str, Any] = resp.json()
        return body

    async def stage_seconds(self) -> dict[str, float]:
        try:
            resp = await self._client.get('/metrics', timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        totals: dict[str, float] = {}
        for family in text_string_to_metric_families(resp.text):
            for sample in family.samples:
                if sample.name == f'{_STAGE_METRIC}_sum':
                    totals[sample.labels['stage']] = sample.value
        return totals

    async def aclose(self) -> None:
        await self._client.aclose()


def loopback_sender(server: tuple[str, int], *, api_prefix: str, project: str) -> HttpBatchSender:
    """A sender aimed at the API process's own listening socket (``server`` is the
    ASGI scope's ``(host, port)``)."""
    host, port = server
    netloc = f'[{host}]' if ':' in host else host
    client = httpx.AsyncClient(
        base_url=f'http://{netloc}:{port}',
        limits=httpx.Limits(max_connections=MAX_SENDER_CONNECTIONS),
    )
    return HttpBatchSender(client, api_prefix=api_prefix, project=project)
