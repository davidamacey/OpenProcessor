"""Shared FastAPI exception-handler bodies, kept out of ``src/main.py`` so
the app-factory module stays under the repo's LOC ratchet.

Each function here is the *body* of an ``@application.exception_handler``
registration in ``src/main.py`` -- registration stays in ``main.py`` (it
needs the live ``application`` instance), but the response-construction
logic lives in exactly one place per exception type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from src.core.logging import get_logger, get_request_id


if TYPE_CHECKING:
    from fastapi import Request

    from src.utils.retry import RetryExhaustedError


logger = get_logger(__name__)

TRITON_UNAVAILABLE_RETRY_AFTER_SECONDS = 5


async def triton_unavailable_response(request: Request, exc: RetryExhaustedError) -> JSONResponse:
    """Triton unreachable after retries (gRPC UNAVAILABLE / connection
    refused / etc.) -> 503, not a bare 500.

    Registered ahead of the generic ``Exception`` handler in
    ``src/main.py``, and every core route (``/detect``, ``/faces/*``,
    ``/embed/*``, ``/ocr/*``, ``/analyze``) re-raises
    :class:`RetryExhaustedError` unchanged instead of wrapping it in a
    generic ``HTTPException``, so this is the single place that decides
    the status code and ``Retry-After`` value for a Triton outage.
    """
    req_id = get_request_id()
    logger.warning(
        'triton_unavailable',
        request_id=req_id,
        method=request.method,
        path=request.url.path,
        error=str(exc),
    )
    return JSONResponse(
        status_code=503,
        content={
            'detail': 'Inference backend (Triton) is temporarily unavailable; retry shortly',
            'request_id': req_id,
            'error_type': type(exc).__name__,
        },
        headers={
            'X-Request-ID': req_id,
            'Retry-After': str(TRITON_UNAVAILABLE_RETRY_AFTER_SECONDS),
        },
    )
