"""HTTP face of the database layer: the request session and how a database failure reaches the client.

``DbSessionDep`` opens one session per request and closes it. It is declared with
``scope="function"``: FastAPI otherwise runs a ``yield`` dependency's exit code after the
response has been sent, when a failure can no longer change the status
(https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/#early-exit-and-scope).
The dependency never commits. Every write commits inside ``project_tx`` or ``deployment_tx``
in the service, before the handler returns, so a failed commit is a 5xx and never a 2xx. SSE and
other streaming routes never declare it: a session held for the life of a stream would pin a
pooled connection.

A request that cannot get a connection in time is told to retry (503 ``database_busy`` with
``Retry-After``); a request that cannot reach the database at all gets 503
``database_unavailable``. Neither is a 500, and neither is retried in-process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import psycopg
from fastapi import Depends
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, TimeoutError as PoolTimeoutError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger, get_request_id
from src.db.session import async_session_factory


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI, Request

logger = get_logger(__name__)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with async_session_factory()() as session:
        yield session


DbSessionDep = Annotated[AsyncSession, Depends(get_db, scope='function')]

DATABASE_BUSY_RETRY_AFTER_SECONDS = 2
DATABASE_UNAVAILABLE_RETRY_AFTER_SECONDS = 5


def _unavailable(code: str, detail: str, retry_after: int) -> JSONResponse:
    req_id = get_request_id()
    return JSONResponse(
        status_code=503,
        content={'detail': detail, 'code': code, 'request_id': req_id},
        headers={'Retry-After': str(retry_after), 'X-Request-ID': req_id},
    )


async def database_busy_response(request: Request, exc: PoolTimeoutError) -> JSONResponse:
    """The request pool had no free connection within its wait."""
    logger.warning('database_busy', path=request.url.path, error=str(exc))
    return _unavailable(
        'database_busy',
        'The database is busy; retry shortly',
        DATABASE_BUSY_RETRY_AFTER_SECONDS,
    )


async def database_unavailable_response(request: Request, exc: OperationalError) -> JSONResponse:
    """The connection itself failed (refused, reset, host down).

    A server-side error such as a cancelled statement is a subclass of the driver's
    ``OperationalError`` and stays a 500: only the bare class means no usable connection.
    """
    if type(exc.orig) is not psycopg.OperationalError:
        raise exc
    logger.error('database_unavailable', path=request.url.path, error=str(exc.orig))
    return _unavailable(
        'database_unavailable',
        'The database is unavailable; retry shortly',
        DATABASE_UNAVAILABLE_RETRY_AFTER_SECONDS,
    )


def install_database_error_handlers(application: FastAPI) -> None:
    application.exception_handler(PoolTimeoutError)(database_busy_response)
    application.exception_handler(OperationalError)(database_unavailable_response)


__all__ = ['DbSessionDep', 'get_db', 'install_database_error_handlers']
