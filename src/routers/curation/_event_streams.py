"""Subscription dependencies and the frame encoder shared by the two event routes.

A stream holds no pooled database connection between queries, so these routes never declare
``DbSessionDep``. Subscribing in a dependency makes an unavailable listener or database a
proper 503 before the response starts, and the dependency's exit unsubscribes when the stream
ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException
from fastapi.sse import ServerSentEvent

from src.config import get_curation_config
from src.services.events.listener import ListenerNotRunningError, Subscription
from src.services.events.sse import Frame, open_subscription


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def _guarded(slug: str | None) -> AsyncIterator[Subscription]:
    try:
        async for sub in open_subscription(slug):
            yield sub
    except ListenerNotRunningError:
        raise HTTPException(status_code=503, detail='the event stream is not running') from None


async def project_subscription() -> AsyncIterator[Subscription]:
    async for sub in _guarded(get_curation_config().project_slug):
        yield sub


async def global_subscription() -> AsyncIterator[Subscription]:
    async for sub in _guarded(None):
        yield sub


ProjectSubscriptionDep = Annotated[Subscription, Depends(project_subscription)]
GlobalSubscriptionDep = Annotated[Subscription, Depends(global_subscription)]


def to_sse(frame: Frame) -> ServerSentEvent:
    return ServerSentEvent(id=frame.id, event=frame.event, data=frame.data)
