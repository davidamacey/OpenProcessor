"""A registry refusal answered with its own status and stable code."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from src.core.exceptions import RequestRejectedError
from src.domain.embedding_spaces import EmbeddingSpaceError


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def space_refusals() -> AsyncIterator[None]:
    """Turn an :class:`EmbeddingSpaceError` into the response of its status and code."""
    try:
        yield
    except EmbeddingSpaceError as exc:
        raise RequestRejectedError(exc.status, {'error': exc.code, 'message': exc.detail}) from exc
