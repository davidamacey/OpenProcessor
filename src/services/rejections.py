"""The structured refusal services raise for a project or config request.

One shape (``{"detail": ConfigErrorDetail}``) for the frontend to parse
everywhere; the app answers a :class:`~src.core.exceptions.RequestRejectedError` with
the status and body given here, exactly as the routers' ``api_error`` does.
"""

from __future__ import annotations

from typing import Any

from src.core.exceptions import RequestRejectedError
from src.schemas.curation.config_common import ErrorCode, error_body


def rejection(status: int, code: ErrorCode, message: str, **fields: Any) -> RequestRejectedError:
    """Build the refusal for ``code`` with only the top-level fields it carries."""
    return RequestRejectedError(status, error_body(code, message, **fields))
