"""HTTP face of the database layer: how a database failure reaches the client.

A request that cannot get a connection in time is told to retry (503 ``database_busy`` with
``Retry-After``); a request that cannot reach the database at all gets 503
``database_unavailable``. Neither is a 500, and neither is retried in-process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import psycopg
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, TimeoutError as PoolTimeoutError

from src.core.logging import get_logger, get_request_id


if TYPE_CHECKING:
    from fastapi import FastAPI, Request

logger = get_logger(__name__)

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
