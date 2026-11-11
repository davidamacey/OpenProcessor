"""The ``cursor`` query parameter and its refusal, shared by every cursored list route."""

from __future__ import annotations

from typing import Annotated

from fastapi import HTTPException, Query

from src.routers.curation._api_error import api_error


CursorQ = Annotated[
    str | None,
    Query(
        description=(
            "The previous page's `next_cursor`; omit it for the first page. A cursor belongs "
            'to the sort that served it and is refused (400 `cursor_invalid`) under another.'
        )
    ),
]


def cursor_refused(exc: Exception) -> HTTPException:
    """The 400 for a cursor that is garbage or belongs to a different sort."""
    return api_error(400, 'cursor_invalid', str(exc))
