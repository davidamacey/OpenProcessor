"""Write the counts-only record of a face or text purge to PostgreSQL."""

from __future__ import annotations

import getpass
import socket
from typing import TYPE_CHECKING

from src.db.repos import privacy_purges
from src.db.session import async_session_factory


if TYPE_CHECKING:
    from collections.abc import Mapping


def _actor() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):  # no passwd entry for the uid in a container
        return 'unknown'


async def record_purge(kind: str, counts: Mapping[str, object]) -> None:
    """One row, committed before it returns. Counts only: never an id, name, path or text."""
    async with async_session_factory()() as session, session.begin():
        await privacy_purges.record(
            session, kind, counts, actor=_actor(), host=socket.gethostname()
        )
