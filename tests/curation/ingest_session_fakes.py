"""Count the PostgreSQL sessions an ingest opens, with the repositories behind them stubbed.

A session here is one pool checkout and one transaction, whatever the code runs inside it. The
stubs make no network call, so the same test runs against the per-photo code too. ``count_sessions``
patches every module that imported ``async_session_factory`` and records, for each placement read
and pending-rows write, which session it ran in.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import src.db.session as db_session
from src.db.repos import image_contents as contents_repo, near_dup_pending as pending_repo
from src.services.curation import ingest_index
from src.services.events import publish as publish_module
from src.services.near_duplicates import content_key, ingest_claim, pending


if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

# The offline conftest swaps these for in-memory fakes per test; keep the real ones.
_REAL = {
    (content_key, 'resolve_many_bound'): content_key.resolve_many_bound,
    (content_key, 'claim_bound'): content_key.claim_bound,
    (content_key, 'release_bound'): content_key.release_bound,
    (ingest_claim, 'claim_and_mark_in_session'): ingest_claim.claim_and_mark_in_session,
    (pending, 'enqueue_pending_bound'): pending.enqueue_pending_bound,
    (publish_module, 'apublish_many'): publish_module.apublish_many,
}
_REAL_FACTORY = db_session.async_session_factory


@dataclasses.dataclass
class Sessions:
    """Sessions opened (pool checkouts; each is one transaction and one COMMIT)."""

    opened: int = 0
    peak: int = 0
    open_now: int = 0
    made: list[Any] = dataclasses.field(default_factory=list)
    """The sessions in the order they were opened."""
    placed_in: list[Any] = dataclasses.field(default_factory=list)
    """The session of each placement read."""
    pending_in: list[Any] = dataclasses.field(default_factory=list)
    """The session of each pending-rows write."""


class _Session:
    def __init__(self, counts: Sessions) -> None:
        self._counts = counts
        self.sync_session = SimpleNamespace(info={})

    async def __aenter__(self) -> _Session:
        self._counts.opened += 1
        self._counts.made.append(self)
        self._counts.open_now += 1
        self._counts.peak = max(self._counts.peak, self._counts.open_now)
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._counts.open_now -= 1

    def in_transaction(self) -> bool:
        return False


def count_sessions(monkeypatch: pytest.MonkeyPatch) -> Sessions:
    counts = Sessions()
    project = uuid.uuid4()

    def factory() -> Callable[[], _Session]:
        return lambda: _Session(counts)

    for module in list(sys.modules.values()):
        if getattr(module, 'async_session_factory', None) is _REAL_FACTORY:
            monkeypatch.setattr(module, 'async_session_factory', factory)
        if hasattr(module, 'bound_project_id') and module.__name__.startswith('src.'):

            async def bound() -> uuid.UUID:
                return project

            monkeypatch.setattr(module, 'bound_project_id', bound)

    @asynccontextmanager
    async def tx(session: Any, _project: Any) -> Any:
        yield session

    for module in (content_key, pending, ingest_claim, ingest_index):
        monkeypatch.setattr(module, 'project_tx', lambda s, p: tx(s, p), raising=False)
    monkeypatch.setattr(db_session, 'project_tx', lambda s, p: tx(s, p))

    async def rows(_session: Any, _project: Any, hashes: Any) -> Any:
        hashes = sorted(set(hashes))
        return {h: [] for h in hashes}

    async def none(*_a: Any, **_k: Any) -> None:
        return None

    async def known(*_a: Any, **_k: Any) -> set[str]:
        return set()

    async def enqueue(session: Any, *_a: Any, **_k: Any) -> int:
        counts.pending_in.append(session)
        return 1

    async def insert_contents(_s: Any, _p: Any, new: Any) -> set[str]:
        return {r.image_id for r in new}

    async def place(session: Any, **_k: Any) -> list[Any]:
        counts.placed_in.append(session)
        return []

    async def no_rows(*_a: Any, **_k: Any) -> list[Any]:
        return []

    async def deleted(*_a: Any, **_k: Any) -> int:
        return 1

    # Names of both the per-photo code and the batch code; whichever exists is stubbed.
    for name, stub in {
        'lock_imohash': none,
        'lock_imohashes': none,
        'rows_for_imohash': no_rows,
        'rows_for_imohashes': rows,
        'imohashes_with_rows': known,
        'insert_content': none,
        'insert_contents': insert_contents,
        'release_contents': deleted,
    }.items():
        monkeypatch.setattr(contents_repo, name, stub, raising=False)
    monkeypatch.setattr(pending_repo, 'enqueue_pending', enqueue)
    monkeypatch.setattr(ingest_index, 'place_new', place)
    monkeypatch.setattr(ingest_index, 'async_session_factory', factory)

    async def inserted(_session: Any, _target: Any, events: Any) -> Any:
        return [SimpleNamespace(project_id=None) for _ in events]

    monkeypatch.setattr(publish_module, 'insert_events_async', inserted)
    monkeypatch.setattr(publish_module, '_pend_wake', lambda *_a, **_k: None)
    monkeypatch.setattr(publish_module, 'deployment_tx', lambda s, _r: tx(s, None))
    for (module, name), real in _REAL.items():
        monkeypatch.setattr(module, name, real, raising=False)
    return counts
