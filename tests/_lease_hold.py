"""Hold a job lease for the body of a ``with`` block, from a sync test.

The lease store is the in-memory one of ``tests/conftest.py``; the client is unused.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from src.clients import job_lease
from src.domain.job_lease import Acquired


if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def held(kind: str, scope: str | None = None, job_id: str = 'held-by-test') -> Iterator[None]:
    result = asyncio.run(job_lease.acquire(object(), kind, scope, job_id))  # type: ignore[arg-type]
    assert isinstance(result, Acquired), result
    try:
        yield
    finally:
        asyncio.run(job_lease.release(object(), result.lease))  # type: ignore[arg-type]
