"""Publishing events: the row is written in the caller's transaction, the wake-up follows the commit.

``publish`` keeps the event hub's name and the (type, fields) shape of its call sites. With a
session the row joins the caller's transaction and the Valkey wake-up is sent only after that
transaction commits (a rolled-back change emits nothing). Without a session the call opens its
own short transaction. Sync ``publish`` is for threads and Celery children; async code awaits
``apublish`` so a failed write reaches the caller instead of being lost.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import event
from sqlalchemy.orm import Session

from src.config.project_context import current_project
from src.db.repos.events import insert_event, insert_event_async
from src.db.session import (
    async_session_factory,
    deployment_tx,
    sync_deployment_tx,
    sync_session_factory,
)
from src.tasks.runtime.event_wakeup import wake


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


# The only event families allowed on the global stream: project lifecycle, the combine job that
# builds a new project (its ``target``) and the deployment-wide VLM registry.
GLOBAL_EVENT_PREFIXES: Final[tuple[str, ...]] = ('project.', 'combine.', 'vlm.')
GLOBAL_EVENT_TOPIC: Final = 'project'
_RESERVED_KEYS = frozenset({'type', 'topic', 'project'})
_WAKES = 'op_event_wakes'


class _Global:
    def __repr__(self) -> str:
        return 'GLOBAL'


GLOBAL: Final = _Global()
"""Pass as ``project`` to publish onto the global (project-less) stream."""

type Target = uuid.UUID | str | _Global | None


def _pend_wake(session: Session, project_id: uuid.UUID | None) -> None:
    session.info.setdefault(_WAKES, set()).add(project_id)


@event.listens_for(Session, 'after_commit')
def _send_wakes(session: Session) -> None:
    # A short blocking call (1 s socket timeouts) after the commit; its failure is logged inside
    # ``wake`` because the listener's 2 s tick delivers the committed row anyway.
    for project_id in session.info.pop(_WAKES, ()):
        wake(project_id)


@event.listens_for(Session, 'after_rollback')
def _drop_wakes(session: Session) -> None:
    session.info.pop(_WAKES, None)


def _stamp(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.setdefault('ts', time.time())
    return body


def _resolve(project: Target) -> uuid.UUID | str | None:
    """The row's project: an explicit id or slug, the global stream, or the bound project."""
    if isinstance(project, _Global):
        return None
    if project is None:
        record = current_project().record
        return record.id or record.slug
    return project


def _write_sync(target: uuid.UUID | str | None, event_type: str, body: dict[str, Any]) -> None:
    with sync_session_factory()() as session, sync_deployment_tx(session, 'event publish'):
        inserted = insert_event(session, target, event_type, body)
        _pend_wake(session, inserted.project_id)


async def _write_async(
    target: uuid.UUID | str | None, events: Sequence[tuple[str, dict[str, Any]]]
) -> None:
    async with async_session_factory()() as session, deployment_tx(session, 'event publish'):
        for event_type, body in events:
            inserted = await insert_event_async(session, target, event_type, body)
            _pend_wake(session.sync_session, inserted.project_id)


class PublishInEventLoopError(RuntimeError):
    """``publish`` was called where a loop is running: it would block the loop on the database."""


def publish(
    event_type: str,
    payload: Mapping[str, Any],
    *,
    project: Target = None,
    session: Session | None = None,
) -> None:
    """Record an event for ``project`` (default: the bound project; ``GLOBAL``: the global stream).

    With ``session`` the row joins that transaction. Without one it opens its own and commits.
    Async code calls :func:`apublish`; calling this inside a running event loop raises
    :class:`PublishInEventLoopError` rather than blocking the loop or dropping the write.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise PublishInEventLoopError('publish() blocks on the database; use await apublish()')
    target = _resolve(project)
    body = _stamp(payload)
    if session is not None:
        inserted = insert_event(session, target, event_type, body)
        _pend_wake(session, inserted.project_id)
        return
    _write_sync(target, event_type, body)


async def apublish(
    event_type: str,
    payload: Mapping[str, Any],
    *,
    project: Target = None,
    session: AsyncSession | None = None,
) -> None:
    """:func:`publish` for async code that wants the write done (and raised) before it continues."""
    target = _resolve(project)
    body = _stamp(payload)
    if session is None:
        await _write_async(target, [(event_type, body)])
        return
    inserted = await insert_event_async(session, target, event_type, body)
    _pend_wake(session.sync_session, inserted.project_id)


async def apublish_many(
    events: Sequence[tuple[str, Mapping[str, Any]]], *, project: Target = None
) -> None:
    """Several events in one transaction (one wake-up), for a batch of ingest or label writes."""
    if not events:
        return
    target = _resolve(project)
    await _write_async(target, [(event_type, _stamp(payload)) for event_type, payload in events])


def _global_body(event_type: str, target: str | None, fields: dict[str, Any]) -> dict[str, Any]:
    if not event_type.startswith(GLOBAL_EVENT_PREFIXES):
        raise ValueError(f'{event_type!r} is not a global event type')
    if reserved := _RESERVED_KEYS & fields.keys():
        raise ValueError(f'fields may not set the event envelope: {sorted(reserved)}')
    topic = 'vlm' if event_type.startswith('vlm.') else GLOBAL_EVENT_TOPIC
    return {**fields, 'topic': topic, 'target': target}


async def apublish_global_event(
    event_type: str, *, target: str | None = None, **fields: Any
) -> None:
    """Publish onto the global stream, whatever project (if any) is bound.

    Only the families in :data:`GLOBAL_EVENT_PREFIXES` may go there, and ``fields`` can never
    override the envelope. ``target`` names the project a ``combine.*`` / ``project.*`` event is
    about; it is on the wire as ``null`` when not given.
    """
    await apublish(event_type, _global_body(event_type, target, fields), project=GLOBAL)


async def apublish_crops_created(crops: Sequence[tuple[str, str]]) -> None:
    """``crop.created`` for each ``(crop_id, image_path)``."""
    await apublish_many(
        [
            ('crop.created', {'topic': 'crop', 'crop_id': crop_id, 'image_path': image_path})
            for crop_id, image_path in crops
        ]
    )


async def apublish_crops_classified(classified: Sequence[Mapping[str, Any]]) -> None:
    """``crop.classified`` for each mapping with ``crop_id``, ``class_id``, ``class_name``, ``class_source``."""
    await apublish_many([('crop.classified', {'topic': 'crop', **c}) for c in classified])
