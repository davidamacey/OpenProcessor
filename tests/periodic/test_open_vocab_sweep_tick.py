"""``op.periodic.open_vocab_sweep``: start the project's ``open_vocab_drain`` only when the active
set opts in on ingest and an image is left ``pending``. The exclusive group of the job kind
coalesces repeat starts, so a second tick while one drain runs adds nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
from fixtures.job_rows import make_project, sql

from src.services.periodic import open_vocab_sweep
from src.services.periodic.runner import run_periodic


pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures('pg_env', 'job_env', 'producer', 'wakes'),
]


class FakeOpenSearch:
    def __init__(self, pending: int) -> None:
        self.pending = pending
        self.queries: list[dict[str, Any]] = []
        self.closed = False

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, int]:
        self.queries.append({'index': index, **body})
        return {'count': self.pending}

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch):
    state = {'client': FakeOpenSearch(pending=3), 'active': (SimpleSet(run_on_ingest=True), 2)}

    async def client() -> FakeOpenSearch:
        return state['client']

    async def active() -> Any:
        return state['active']

    monkeypatch.setattr(open_vocab_sweep, 'opensearch_client', client)
    monkeypatch.setattr(open_vocab_sweep, 'current_active_set', active)
    monkeypatch.setattr(open_vocab_sweep, 'images_index', lambda: 'images')
    return state


class SimpleSet:
    def __init__(self, *, run_on_ingest: bool) -> None:
        self.run_on_ingest = run_on_ingest


def drains(pg, pid) -> int:
    rows = sql(pg, "SELECT 1 FROM jobs WHERE kind = 'open_vocab_drain' AND project_id = %s", pid)
    return len(rows)


def tick() -> Any:
    return run_periodic('open_vocab_sweep', open_vocab_sweep.tick())


def test_pending_images_of_an_opted_in_project_start_one_drain(pg, world) -> None:
    pid = make_project(pg)
    tick()
    tick()  # the unique index of the exclusive group coalesces the second start
    assert drains(pg, pid) == 1
    assert world['client'].closed
    assert world['client'].queries[0]['index'] == 'images'
    assert world['client'].queries[0]['query'] == {'term': {'open_vocab_status': 'pending'}}


def test_nothing_pending_starts_nothing(pg, world) -> None:
    pid = make_project(pg)
    world['client'] = FakeOpenSearch(pending=0)
    tick()
    assert drains(pg, pid) == 0


def test_a_set_that_does_not_run_on_ingest_starts_nothing_and_asks_nothing(pg, world) -> None:
    pid = make_project(pg)
    world['active'] = (SimpleSet(run_on_ingest=False), 1)
    tick()
    assert drains(pg, pid) == 0
    assert world['client'].queries == []


def test_no_active_set_starts_nothing(pg, world) -> None:
    pid = make_project(pg)
    world['active'] = None
    tick()
    assert drains(pg, pid) == 0
