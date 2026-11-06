"""Scoped transactions: the only way application code reaches a row-level-secured table.

Every project-scoped table has row-level security keyed on two transaction-local settings,
``op.scope`` and ``op.project_id``. The helpers here set them. The scope lives in a context
variable and is re-applied by a session ``after_begin`` event, so a scoped session that
commits and auto-begins again inside the same block keeps its scope instead of silently
losing it. ``set_config(..., true)`` is transaction-local, so a pooled connection never carries
a binding to the next request. A session that opens a transaction outside any helper has no
scope and PostgreSQL raises ``scope_not_set`` on the first protected statement.

``deployment_tx`` sees every project and ``core_tx`` sees the rows that belong to no project
(core faces, core near-duplicates, core jobs); both are restricted to the modules in
``src.db.scope_allowlist``.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker

from src.db.engine import current_async_engine, get_sync_engine
from src.db.scope_allowlist import CORE_TX_MODULES, DEPLOYMENT_TX_MODULES


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from types import FrameType

    from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

_TESTS_ROOT = Path(__file__).resolve().parents[2] / 'tests'
_APPLY_SCOPE = text(
    "SELECT set_config('op.scope', :scope, true), set_config('op.project_id', :pid, true)"
)


class ScopeNotAllowedError(PermissionError):
    """A module opened a transaction scope it is not listed for."""


class ScopeError(RuntimeError):
    """A scoped transaction was opened where it cannot be."""


@dataclass(frozen=True)
class Scope:
    kind: str
    project_id: str = ''


_scope: contextvars.ContextVar[Scope | None] = contextvars.ContextVar('op_db_scope', default=None)


def _reapply_scope(_session: Session, _transaction: object, connection: Connection) -> None:
    scope = _scope.get()
    if scope is not None:
        connection.execute(_APPLY_SCOPE, {'scope': scope.kind, 'pid': scope.project_id})


event.listen(Session, 'after_begin', _reapply_scope)


def _caller_is_allowed(frame: FrameType, allowed: tuple[str, ...]) -> bool:
    module = frame.f_globals.get('__name__', '')
    if any(fnmatchcase(module, p) or module.startswith(p + '.') for p in allowed):
        return True
    try:
        return Path(frame.f_code.co_filename).resolve().is_relative_to(_TESTS_ROOT)
    except OSError:
        return False


def _require(allowed: tuple[str, ...], helper: str) -> None:
    """Refuse unless the module that called ``helper`` (two frames up) is allowed."""
    caller = sys._getframe(2)  # the importer of the helper is the subject
    if not _caller_is_allowed(caller, allowed):
        msg = (
            f'{caller.f_globals.get("__name__", "?")} may not use {helper}; '
            'add it to src/db/scope_allowlist.py with its reason'
        )
        raise ScopeNotAllowedError(msg)


def _project_id(project: uuid.UUID | str) -> str:
    try:
        return str(uuid.UUID(str(project)))
    except ValueError:
        msg = f'not a project id: {project!r}'
        raise ScopeError(msg) from None


@asynccontextmanager
async def _scoped(session: AsyncSession, scope: Scope) -> AsyncIterator[AsyncSession]:
    """Bind ``scope`` for the block and commit on a clean exit, roll back on any error.

    The block may commit itself; the next statement auto-begins and ``after_begin`` re-applies
    the scope. A failure of the final commit propagates, so a write is never reported as done
    when it was not stored.
    """
    if session.in_transaction():
        msg = 'a scoped transaction needs a session with no transaction in progress'
        raise ScopeError(msg)
    token = _scope.set(scope)
    try:
        yield session
        if session.in_transaction():
            await session.commit()
    except BaseException:
        if session.in_transaction():
            await session.rollback()
        raise
    finally:
        _scope.reset(token)


@contextmanager
def _scoped_sync(session: Session, scope: Scope) -> Iterator[Session]:
    if session.in_transaction():
        msg = 'a scoped transaction needs a session with no transaction in progress'
        raise ScopeError(msg)
    token = _scope.set(scope)
    try:
        yield session
        if session.in_transaction():
            session.commit()
    except BaseException:
        if session.in_transaction():
            session.rollback()
        raise
    finally:
        _scope.reset(token)


def project_tx(
    session: AsyncSession, project: uuid.UUID | str
) -> AbstractAsyncContextManager[AsyncSession]:
    """A transaction that sees only ``project``'s rows; commits on exit, rolls back on error."""
    return _scoped(session, Scope('project', _project_id(project)))


def sync_project_tx(session: Session, project: uuid.UUID | str) -> AbstractContextManager[Session]:
    return _scoped_sync(session, Scope('project', _project_id(project)))


def deployment_tx(session: AsyncSession, reason: str) -> AbstractAsyncContextManager[AsyncSession]:
    """A transaction over every project's rows. Allowed only to the modules in the allowlist."""
    _require(DEPLOYMENT_TX_MODULES, 'deployment_tx')
    if not reason.strip():
        raise ScopeError('deployment_tx needs a reason')
    logger.debug('deployment_scope reason=%s', reason)
    return _scoped(session, Scope('deployment'))


def sync_deployment_tx(session: Session, reason: str) -> AbstractContextManager[Session]:
    _require(DEPLOYMENT_TX_MODULES, 'sync_deployment_tx')
    if not reason.strip():
        raise ScopeError('deployment_tx needs a reason')
    logger.debug('deployment_scope reason=%s', reason)
    return _scoped_sync(session, Scope('deployment'))


def core_tx(session: AsyncSession, reason: str) -> AbstractAsyncContextManager[AsyncSession]:
    """A transaction over the rows that belong to no project. Allowed only to listed modules."""
    _require(CORE_TX_MODULES, 'core_tx')
    if not reason.strip():
        raise ScopeError('core_tx needs a reason')
    return _scoped(session, Scope('core'))


def sync_core_tx(session: Session, reason: str) -> AbstractContextManager[Session]:
    _require(CORE_TX_MODULES, 'sync_core_tx')
    if not reason.strip():
        raise ScopeError('core_tx needs a reason')
    return _scoped_sync(session, Scope('core'))


def async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Sessions over the engine bound to this context, else the API engine.
    ``expire_on_commit=False`` so an attribute read after commit never does implicit I/O."""
    return async_sessionmaker(current_async_engine(), expire_on_commit=False)


def sync_session_factory() -> sessionmaker[Session]:
    """Sessions over the child's sync engine."""
    return sessionmaker(get_sync_engine(), expire_on_commit=False)


__all__ = [
    'Scope',
    'ScopeError',
    'ScopeNotAllowedError',
    'async_session_factory',
    'core_tx',
    'deployment_tx',
    'project_tx',
    'sync_core_tx',
    'sync_deployment_tx',
    'sync_project_tx',
    'sync_session_factory',
]
