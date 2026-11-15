"""``apublish_many`` sends one INSERT per call, not one per event (offline, no database).

The session is a recorder: it counts ``execute`` calls and answers a RETURNING of ids, so the test
proves the statement count and the id-to-event mapping without a server.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from src.db.models.jobs import EVENT_PAYLOAD_MAX_BYTES
from src.db.repos import events as events_repo
from src.db.repos.events import EventPayloadTooLargeError
from src.services.events import publish as publish_module


PROJECT = uuid.uuid4()
# The offline fixtures replace ``apublish_many`` per test; keep the real one.
REAL_APUBLISH_MANY = publish_module.apublish_many


class _Result:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def scalars(self) -> _Result:
        return self

    def scalar_one(self) -> int:
        return self._ids[0]

    def all(self) -> list[int]:
        return self._ids


class RecordingSession:
    """Both the async and the sync (``run_sync``) side record into one list."""

    def __init__(self) -> None:
        self.statements: list[tuple[Any, Any]] = []
        self.sync_session = SimpleNamespace(info={}, execute=self._execute)
        self._next_id = 1000

    def _execute(self, statement: Any, params: Any = None) -> _Result:
        self.statements.append((statement, params))
        count = len(params) if isinstance(params, list) else 1
        ids = list(range(self._next_id, self._next_id + count))
        self._next_id += count
        return _Result(ids)

    async def execute(self, statement: Any, params: Any = None) -> _Result:
        return self._execute(statement, params)

    async def run_sync(self, fn: Any) -> Any:
        return fn(self.sync_session)


def _rows(n: int) -> list[tuple[str, dict[str, Any]]]:
    return [('crop.created', {'topic': 'crop', 'crop_id': f'c{i}'}) for i in range(n)]


@pytest.mark.asyncio
async def test_200_events_are_one_statement_with_ids_in_order() -> None:
    session = RecordingSession()
    inserted = await events_repo.insert_events_async(session, PROJECT, _rows(200))  # type: ignore[arg-type]
    assert len(session.statements) == 1
    assert [e.id for e in inserted] == list(range(1000, 1200))
    assert {e.project_id for e in inserted} == {PROJECT}
    assert [p['payload']['crop_id'] for p in session.statements[0][1]] == [
        f'c{i}' for i in range(200)
    ]


@pytest.mark.asyncio
async def test_zero_events_run_no_statement() -> None:
    session = RecordingSession()
    assert await events_repo.insert_events_async(session, PROJECT, []) == []  # type: ignore[arg-type]
    assert session.statements == []


@pytest.mark.asyncio
async def test_one_event_is_one_statement() -> None:
    session = RecordingSession()
    inserted = await events_repo.insert_events_async(session, PROJECT, _rows(1))  # type: ignore[arg-type]
    assert len(session.statements) == 1
    assert [e.id for e in inserted] == [1000]


@pytest.mark.asyncio
async def test_more_events_than_one_insertmanyvalues_batch_is_still_one_execute() -> None:
    session = RecordingSession()
    inserted = await events_repo.insert_events_async(session, PROJECT, _rows(2500))  # type: ignore[arg-type]
    assert len(session.statements) == 1
    assert len(inserted) == 2500


@pytest.mark.asyncio
async def test_an_oversized_payload_inserts_nothing() -> None:
    session = RecordingSession()
    rows = [*_rows(3), ('crop.created', {'blob': 'x' * (EVENT_PAYLOAD_MAX_BYTES + 1)})]
    with pytest.raises(EventPayloadTooLargeError):
        await events_repo.insert_events_async(session, PROJECT, rows)  # type: ignore[arg-type]
    assert session.statements == []


@pytest.mark.asyncio
async def test_apublish_many_is_one_statement_and_one_wakeup(monkeypatch) -> None:
    session = RecordingSession()

    @asynccontextmanager
    async def tx(_session: Any, _reason: str):
        yield

    @asynccontextmanager
    async def open_session():
        yield session

    monkeypatch.setattr(publish_module, 'async_session_factory', lambda: open_session)
    monkeypatch.setattr(publish_module, 'deployment_tx', tx)
    await REAL_APUBLISH_MANY(_rows(200), project=PROJECT)
    assert len(session.statements) == 1
    assert session.sync_session.info[publish_module._WAKES] == {PROJECT}
