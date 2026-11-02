"""5xx responses for the routers: a fixed message to the caller, the cause in the log."""

from __future__ import annotations

from fastapi import HTTPException

from src.core.logging import get_logger


logger = get_logger(__name__)


def server_error(status: int, message: str, exc: BaseException) -> HTTPException:
    """Build the ``HTTPException`` for a failed request and log ``exc`` server-side.

    ``message`` is a fixed string (what failed, never why); the exception text stays out of the
    response body. Call it inside the ``except`` block so the traceback is logged.
    """
    logger.error('router_5xx', message=message, status=status, exc_info=exc)
    return HTTPException(status_code=status, detail=message)
