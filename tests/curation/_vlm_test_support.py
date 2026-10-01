"""Helpers for tests whose OpenSearch stand-in is a bare ``AsyncMock``: the
VLM routes read the endpoint registry through it, so it must answer those
reads the way an empty registry would (nothing stored, nothing probed)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from opensearchpy.exceptions import NotFoundError


def empty_registry_reads(client: Any) -> Any:
    client.get = AsyncMock(side_effect=NotFoundError(404, 'not_found', {}))
    client.search = AsyncMock(return_value={'hits': {'hits': []}})
    return client
