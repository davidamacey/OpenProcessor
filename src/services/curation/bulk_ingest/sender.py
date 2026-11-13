"""How a bulk run sends one batch of paths through the ingest route.

The runner talks to a :class:`BatchSender`; in production that is
:class:`HttpBatchSender` aimed at the API's own listening socket, so batches land
on every ``api`` worker process rather than only the one that owns the run.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import httpx
from prometheus_client.parser import text_string_to_metric_families
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.services.curation.bulk_ingest.limits import BATCH_REQUEST_TIMEOUT_S, MAX_SENDER_CONNECTIONS
from src.services.curation.bulk_ingest.object_store import ObjectRef


_STAGE_METRIC = 'op_pipeline_stage_seconds'


#: Attempts at one batch the route answers with a retryable 503 (``database_busy``, Triton
#: outage); the last answer then fails the batch as a whole.
BUSY_MAX_ATTEMPTS = 6
RETRY_SLEEP = asyncio.sleep
_BACKOFF = wait_exponential_jitter(initial=1.0, max=30.0)


class BatchRequestError(Exception):
    """The ingest route refused or failed a batch as a whole."""


class BatchBusyError(BatchRequestError):
    """The route answered 503 with ``Retry-After``: the batch was not processed, so sending it
    again is safe."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _wait_for_busy(state: RetryCallState) -> float:
    """The longer of the server's ``Retry-After`` and a jittered exponential backoff (tenacity
    docs: https://tenacity.readthedocs.io/en/stable/#waiting-before-retrying)."""
    outcome = state.outcome
    exc = outcome.exception() if outcome is not None else None
    hinted = exc.retry_after if isinstance(exc, BatchBusyError) else 0.0
    return max(hinted, _BACKOFF(state))


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get('Retry-After')
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


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
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(BatchBusyError),
            wait=_wait_for_busy,
            stop=stop_after_attempt(BUSY_MAX_ATTEMPTS),
            sleep=RETRY_SLEEP,
            reraise=True,
        ):
            with attempt:
                return await self._post_once(paths, source, run_id)
        raise AssertionError('unreachable')  # pragma: no cover - AsyncRetrying always yields

    async def _post_once(self, paths: list[str], source: str, run_id: str) -> dict[str, Any]:
        try:
            resp = await self._client.post(
                self._batch_url,
                json=self._request_body(paths, source, run_id),
                timeout=BATCH_REQUEST_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise BatchRequestError(f'{type(exc).__name__}: {exc}'[:300]) from exc
        if resp.status_code != 200:
            message = f'HTTP {resp.status_code}: {resp.text[:250]}'
            retry_after = _retry_after(resp) if resp.status_code == 503 else None
            if retry_after is not None:
                raise BatchBusyError(message, retry_after)
            raise BatchRequestError(message)
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
    """A sender aimed at an explicit API address (a job has no request to take the
    listening socket from)."""
    cls = ObjectBatchSender if objects else HttpBatchSender
    return cls(make_client(base_url), api_prefix=api_prefix, project=project)
