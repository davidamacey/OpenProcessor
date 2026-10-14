"""HTTP error builders for the curation routers.

Every project or config route raises through :func:`api_error`, never a bare
string ``detail``: one stable shape (``{"detail": ConfigErrorDetail}``) for the
frontend to parse everywhere. Services do not build ``HTTPException``; they
raise :class:`~src.core.exceptions.RequestRejectedError`, which the app answers with
the same body.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.schemas.curation.config_common import ActiveRef, ErrorCode, error_body


def api_error(status: int, code: ErrorCode, message: str, **fields: Any) -> HTTPException:
    """Build ``HTTPException(status, {"detail": ConfigErrorDetail})`` with only the
    top-level fields the code carries."""
    return HTTPException(status_code=status, detail=error_body(code, message, **fields))


def active_conflict_error(what: str, current: dict[str, Any] | None) -> HTTPException:
    """The 409 for every activation compare-and-set miss. The message spells out
    the ``expected_active`` shape so a caller that omitted it (or sent the wrong
    shape) can fix the request without reading the schema; ``current`` is what
    to echo back."""
    return api_error(
        409,
        'active_conflict',
        f'{what} does not match expected_active; send expected_active as '
        '{"name": <string or null>, "revision": <integer or null>} equal to `current` '
        '(omit it, or send null name, only when nothing is active)',
        current=ActiveRef(**current) if current else None,
    )
