"""OpenSearch task helpers: long-running ``_by_query`` calls submitted as tasks and polled.

Docs: https://docs.opensearch.org/latest/api-reference/document-apis/update-by-query/
(``wait_for_completion=false`` returns a task id) and the tasks API.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


# Default poll cadence + ceiling for run_update_by_query_polled. 2s is
# frequent enough to feel responsive on a job dashboard without hammering
# the cluster; 1800s (30 min) matches the timeout envelope this module's
# blocking predecessor used for the same full-index operation.
_UBQ_POLL_INTERVAL_S = 2.0
_UBQ_POLL_TIMEOUT_S = 1800.0


async def run_update_by_query_polled(
    client: AsyncOpenSearch,
    *,
    index: str,
    body: dict[str, Any],
    conflicts: str = 'proceed',
    refresh: bool = True,
    poll_interval_s: float = _UBQ_POLL_INTERVAL_S,
    timeout_s: float = _UBQ_POLL_TIMEOUT_S,
    slices: int | None = None,
    requests_per_second: int | None = None,
) -> dict[str, Any]:
    """Run ``update_by_query`` without blocking on one giant response.

    ``wait_for_completion=True`` holds the HTTP connection open until
    OpenSearch finishes the *entire* operation, then returns one huge
    response. On a large index (the items index can be hundreds of
    thousands of docs) this has a real, observed failure mode distinct
    from a timeout: OpenSearch completes the operation successfully
    server-side (confirmed via ``GET _tasks`` mid-run), but the client
    fails to *parse* the final response — ``aiohttp.http_exceptions.
    BadHttpMessage: 400, Too many headers received`` — and
    opensearch-py's transport-level retry then re-runs the entire
    multi-minute operation again from scratch, hitting the identical
    parse failure every time. The operation itself was never broken;
    only the client's blocking-response path was.

    Fix: submit with ``wait_for_completion=False`` (returns a task id
    immediately, no giant response to fail on) and poll
    ``GET _tasks/{task_id}`` — the same task-status endpoint used to watch
    an in-flight ``update_by_query`` from the CLI — until OpenSearch
    reports it done. Raises on the task's own error (a real per-shard
    failure) or on exceeding ``timeout_s`` (mirrors the old client-side
    timeout, not a server-side one now).
    """
    throttle: dict[str, Any] = {}
    if slices is not None:
        throttle['slices'] = slices
    if requests_per_second is not None:
        throttle['requests_per_second'] = requests_per_second
    started = await client.update_by_query(
        index=index,
        body=body,
        conflicts=conflicts,
        refresh=refresh,
        wait_for_completion=False,
        **throttle,
    )
    task_id = started['task']
    deadline = time.monotonic() + timeout_s
    while True:
        task = await client.tasks.get(task_id=task_id)
        if task.get('completed'):
            error = task.get('error')
            if error:
                raise RuntimeError(f'update_by_query task {task_id} failed: {error}')
            return task.get('response') or {}
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f'update_by_query task {task_id} did not complete within {timeout_s}s'
            )
        await asyncio.sleep(poll_interval_s)
