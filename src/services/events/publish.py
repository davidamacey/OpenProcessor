"""Publishing events: the row is written in the caller's transaction, the wake-up follows the commit.

``publish`` keeps the event hub's name and the (type, fields) shape of its call sites. With a
session the row joins the caller's transaction and the Valkey wake-up is sent only after that
transaction commits (a rolled-back change emits nothing). Without a session the call opens its
own short transaction. Inside a running event loop that write is scheduled as a task, because
the hub's ``publish`` never blocked its caller either; outside a loop it runs inline.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import event
from sqlalchemy.orm import Session

from src.config.project_context import current_project
from src.core.logging import get_logger
from src.db.repos.events import insert_event, insert_event_async
from src.db.session import (
    async_session_factory,
    deployment_tx,
    sync_deployment_tx,
    sync_session_factory,
)
from src.tasks.runtime.event_wakeup import wake


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession


logger = get_logger(__name__)

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


_tasks: set[asyncio.Task[None]] = set()


def _run_wake(project_id: uuid.UUID | None) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        wake(project_id)
        return
    # The wake-up is a blocking socket call (1 s timeout); keep it off the event loop.
    task = loop.create_task(asyncio.to_thread(wake, project_id))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def _pend_wake(session: Session, project_id: uuid.UUID | None) -> None:
    session.info.setdefault(_WAKES, set()).add(project_id)


@event.listens_for(Session, 'after_commit')
def _send_wakes(session: Session) -> None:
    for project_id in session.info.pop(_WAKES, ()):
        _run_wake(project_id)


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
        return current_project().record.slug
    return project


def _write_sync(target: uuid.UUID | str | None, event_type: str, body: dict[str, Any]) -> None:
    with sync_session_factory()() as session, sync_deployment_tx(session, 'event publish'):
        inserted = insert_event(session, target, event_type, body)
        _pend_wake(session, inserted.project_id)


async def _write_async(
    target: uuid.UUID | str | None, event_type: str, body: dict[str, Any]
) -> None:
    async with async_session_factory()() as session, deployment_tx(session, 'event publish'):
        inserted = await insert_event_async(session, target, event_type, body)
        _pend_wake(session.sync_session, inserted.project_id)


def _log_failure(task: asyncio.Task[None]) -> None:
    _tasks.discard(task)
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning('event_publish_failed', error=str(exc))


def publish(
    event_type: str,
    payload: Mapping[str, Any],
    *,
    project: Target = None,
    session: Session | None = None,
) -> None:
    """Record an event for ``project`` (default: the bound project; ``GLOBAL``: the global stream).

    With ``session`` the row joins that transaction. Without one: inside a running loop the
    write is scheduled (failures are logged, never raised into the caller); outside a loop it
    runs inline and raises on failure.
    """
    target = _resolve(project)
    body = _stamp(payload)
    if session is not None:
        inserted = insert_event(session, target, event_type, body)
        _pend_wake(session, inserted.project_id)
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _write_sync(target, event_type, body)
        return
    task = loop.create_task(_write_async(target, event_type, body))
    _tasks.add(task)
    task.add_done_callback(_log_failure)


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
        await _write_async(target, event_type, body)
        return
    inserted = await insert_event_async(session, target, event_type, body)
    _pend_wake(session.sync_session, inserted.project_id)


def publish_global_event(
    event_type: str,
    *,
    target: str | None = None,
    **fields: Any,
) -> None:
    """Publish onto the global stream, whatever project (if any) is bound.

    Only the families in :data:`GLOBAL_EVENT_PREFIXES` may go there, and ``fields`` can never
    override the envelope. ``target`` names the project a ``combine.*`` / ``project.*`` event is
    about; it is on the wire as ``null`` when not given.
    """
    if not event_type.startswith(GLOBAL_EVENT_PREFIXES):
        raise ValueError(f'{event_type!r} is not a global event type')
    if reserved := _RESERVED_KEYS & fields.keys():
        raise ValueError(f'fields may not set the event envelope: {sorted(reserved)}')
    topic = 'vlm' if event_type.startswith('vlm.') else GLOBAL_EVENT_TOPIC
    publish(event_type, {**fields, 'topic': topic, 'target': target}, project=GLOBAL)


def publish_crop_created(crop_id: str, image_path: str = '') -> None:
    publish('crop.created', {'topic': 'crop', 'crop_id': crop_id, 'image_path': image_path})


def publish_crop_classified(
    crop_id: str,
    *,
    class_id: int | None,
    class_name: str | None = None,
    class_source: str = '',
) -> None:
    publish(
        'crop.classified',
        {
            'topic': 'crop',
            'crop_id': crop_id,
            'class_id': class_id,
            'class_name': class_name,
            'class_source': class_source,
        },
    )
