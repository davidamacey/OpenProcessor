"""The API refuses to start unless every required service answers and the schema is current.

Run once from the lifespan, before any other startup step. The probes of services that are
not the database live here because they use their clients; the waiting and the schema check
are shared with the workers (``src.db.schema_check``).
"""

from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING

import httpx

from src.db.schema_check import (
    Clock,
    ProbeFailed,
    check_control_plane,
    database_probe,
    wait_for_dependencies,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.config import Settings

PROBE_TIMEOUT_S = 2.0


def _queue_probe() -> None:
    """The task queue answers. The producer client belongs to the queue module."""
    client = importlib.import_module('src.clients.task_queue')
    try:
        client.ping()
    except client.QueueUnavailable as exc:
        raise ProbeFailed(type(exc).__name__) from exc


def _opensearch_probe(url: str) -> Callable[[], None]:
    def probe() -> None:
        try:
            response = httpx.get(url, timeout=PROBE_TIMEOUT_S)
        except httpx.HTTPError as exc:
            raise ProbeFailed(type(exc).__name__) from exc
        if response.status_code >= 500:
            raise ProbeFailed(f'HTTP {response.status_code}')

    return probe


def _triton_probe(url: str) -> Callable[[], None]:
    def probe() -> None:
        from tritonclient import grpc as triton_grpc
        from tritonclient.utils import InferenceServerException

        try:
            client = triton_grpc.InferenceServerClient(url=url)
            live = client.is_server_live()
            client.close()
        except (InferenceServerException, OSError) as exc:
            raise ProbeFailed(type(exc).__name__) from exc
        if not live:
            raise ProbeFailed('not live')

    return probe


def startup_probes(settings: Settings) -> dict[str, Callable[[], None]]:
    return {
        'database': database_probe,
        'queue': _queue_probe,
        'opensearch': _opensearch_probe(settings.opensearch_url),
        'triton': _triton_probe(settings.triton_url),
    }


async def require_control_plane(settings: Settings, clock: Clock | None = None) -> None:
    """Wait for the dependencies, then check the schema head. Raises to refuse the start."""
    await asyncio.to_thread(wait_for_dependencies, startup_probes(settings), clock=clock)
    await asyncio.to_thread(check_control_plane)
