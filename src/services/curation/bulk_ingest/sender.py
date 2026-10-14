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
from src.services.curation.bulk_ingest.object_store import ObjectRef


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

    def _request_body(self, paths: list[str], source: str, run_id: str) -> dict[str, Any]:
        return {
            'items': [{'path': p, 'source': source} for p in paths],
            'ingest_run_id': run_id,
        }

    def _answer(self, body: dict[str, Any], paths: list[str]) -> dict[str, Any]:  # noqa: ARG002  # interface method; this implementation does not need the argument
        return body

    async def post_batch(self, paths: list[str], *, source: str, run_id: str) -> dict[str, Any]:
        try:
            resp = await self._client.post(
                self._batch_url,
                json=self._request_body(paths, source, run_id),
                timeout=BATCH_REQUEST_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise BatchRequestError(f'{type(exc).__name__}: {exc}'[:300]) from exc
        if resp.status_code != 200:
            raise BatchRequestError(f'HTTP {resp.status_code}: {resp.text[:250]}')
        body: dict[str, Any] = resp.json()
        return self._answer(body, paths)

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


class ObjectBatchSender(HttpBatchSender):
    """The same sender for ``s3://`` references: ``POST .../ingest/objects``. The
    route may store an object under the etag it actually fetched, so each result is
    matched back to the reference the run sent by bucket and key."""

    def __init__(self, client: httpx.AsyncClient, *, api_prefix: str, project: str) -> None:
        super().__init__(client, api_prefix=api_prefix, project=project)
        self._batch_url = f'{api_prefix}/projects/{project}/ingest/objects'

    def _request_body(self, paths: list[str], source: str, run_id: str) -> dict[str, Any]:
        items = []
        for uri in paths:
            ref = ObjectRef.parse(uri)
            item: dict[str, Any] = {'bucket': ref.bucket, 'key': ref.key, 'etag': ref.etag}
            if ref.version_id:
                item['version_id'] = ref.version_id
            items.append(item)
        return {'items': items, 'source': source, 'ingest_run_id': run_id}

    def _answer(self, body: dict[str, Any], paths: list[str]) -> dict[str, Any]:
        sent: dict[tuple[str, str], list[ObjectRef]] = {}
        for uri in paths:
            ref = ObjectRef.parse(uri)
            sent.setdefault((ref.bucket, ref.key), []).append(ref)
        answered: set[str] = set()
        extra: list[dict[str, Any]] = []
        for result in body.get('results', []):
            identifier = result.get('source_identifier') or result.get('image_path') or ''
            try:
                got = ObjectRef.parse(identifier)
            except ValueError:
                continue
            rows = sent.get((got.bucket, got.key), [])
            same = [r for r in rows if r.version_id == got.version_id]
            match = next((r for r in same if r.etag == got.etag), None) or (
                same[0] if same else None
            )
            if match is None:
                continue
            result['image_path'] = match.to_uri()
            answered.add(match.to_uri())
            # Another row of the same object (a stale etag of a replaced key) was
            # fetched as this same current object, so it settles the same way.
            extra.extend(
                {**result, 'image_path': r.to_uri()}
                for r in same
                if r is not match and r.to_uri() not in answered
            )
            answered.update(r.to_uri() for r in same)
        body['results'] = [*body.get('results', []), *extra]
        return body


def make_client(base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url, limits=httpx.Limits(max_connections=MAX_SENDER_CONNECTIONS)
    )


def url_sender(
    base_url: str, *, api_prefix: str, project: str, objects: bool = False
) -> HttpBatchSender:
    """A sender aimed at an explicit API address (the dispatcher has no request to take
    the listening socket from)."""
    cls = ObjectBatchSender if objects else HttpBatchSender
    return cls(make_client(base_url), api_prefix=api_prefix, project=project)


def loopback_sender(
    server: tuple[str, int], *, api_prefix: str, project: str, objects: bool = False
) -> HttpBatchSender:
    """A sender aimed at the API process's own listening socket (``server`` is the
    ASGI scope's ``(host, port)``)."""
    host, port = server
    netloc = f'[{host}]' if ':' in host else host
    return url_sender(
        f'http://{netloc}:{port}', api_prefix=api_prefix, project=project, objects=objects
    )
