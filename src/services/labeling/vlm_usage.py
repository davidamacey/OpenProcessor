"""Record what each answered VLM call cost, under the caller named by ``usage_source``.

A labeler is built with a ``usage_source`` (``assist_loop``, ``autolabel``, ...) or ``None`` for
a reachability or test probe that is not a workload. After every answered call it adds the call
to ``vlm_usage_hourly`` for the bound project. A job handler routes the write through its own
session with :func:`usage_sessions`; the API path opens a project transaction on the process
engine.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from src.config.project_context import current_project_id, try_current_project
from src.core.logging import get_logger
from src.db.repos.vlm_usage import record_vlm_usage
from src.db.session import async_session_factory, project_tx
from src.domain.vlm_usage import NO_USAGE, VlmUsage, crops_in_payload, usage_from_response


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

_PROVIDER: ContextVar[Callable[[], AbstractAsyncContextManager[AsyncSession]] | None] = ContextVar(
    'vlm_usage_sessions', default=None
)


_OBSERVER: ContextVar[list[VlmUsage] | None] = ContextVar('vlm_usage_observer', default=None)


@contextmanager
def observe_usage() -> Iterator[list[VlmUsage]]:
    """Collect the usage of every answered call made inside this block (the calls of the task
    and of the tasks it starts), so a caller can settle a budget at what the endpoint reported."""
    seen: list[VlmUsage] = []
    token = _OBSERVER.set(seen)
    try:
        yield seen
    finally:
        _OBSERVER.reset(token)


def total_usage(seen: list[VlmUsage]) -> VlmUsage:
    """The sum of observed calls; without usage on every call (or with no call) it is
    ``NO_USAGE``, which a budget charges at the reserved upper bound."""
    if not seen or not all(u.present for u in seen):
        return NO_USAGE
    return VlmUsage(
        prompt_tokens=sum(u.prompt_tokens for u in seen),
        completion_tokens=sum(u.completion_tokens for u in seen),
        present=True,
    )


@contextmanager
def usage_sessions(
    provider: Callable[[], AbstractAsyncContextManager[AsyncSession]],
) -> Iterator[None]:
    """Write usage through ``provider`` (a job's ``ctx.session``) while this block runs."""
    token = _PROVIDER.set(provider)
    try:
        yield
    finally:
        _PROVIDER.reset(token)


@asynccontextmanager
async def _process_session() -> AsyncIterator[AsyncSession]:
    """A project transaction on the process engine, for the API path."""
    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        yield session


def _open_session() -> AbstractAsyncContextManager[AsyncSession]:
    provider = _PROVIDER.get()
    return provider() if provider is not None else _process_session()


async def record_call(
    *, endpoint_name: str, source: str | None, payload: dict[str, Any], response: dict[str, Any]
) -> None:
    """Add one answered call. ``source`` ``None`` records nothing. An unbound project or an
    endpoint with no stored document is logged and skipped: the answer is already paid for,
    and losing it would be worse than a missing usage row."""
    observed = _OBSERVER.get()
    if observed is not None:
        observed.append(usage_from_response(response))
    if source is None:
        return
    bound = try_current_project()
    if bound is None or bound.record.id is None:
        logger.warning('vlm_usage.no_project', source=source)
        return
    project_id = bound.record.id
    usage = usage_from_response(response)
    # Imported here: the config store imports the labeling package.
    from src.services.config_store.vlm_endpoints import endpoint_config_id

    config_id = await endpoint_config_id(endpoint_name)
    if config_id is None:
        logger.warning('vlm_usage.endpoint_unregistered', endpoint=endpoint_name)
        return
    crops = crops_in_payload(payload)
    async with _open_session() as session:
        await session.run_sync(
            lambda s: record_vlm_usage(s, project_id, config_id, source, crops, usage)
        )
