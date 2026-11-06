"""Two ingest requests make one pending drain job: the exclusive group coalesces the starts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from fixtures.job_rows import sql
from projects.conftest import FakeLifecycleOpenSearch, fake_ensure_indexes

from src.config.project_context import bind_project
from src.core.exceptions import RequestRejectedError
from src.services.curation import open_vocab_ingest
from src.services.curation.open_vocab_ingest import schedule_open_vocab_after_ingest, start_drain
from src.services.projects.registry import ProjectRegistry, set_project_registry


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real project row, bound the way a request binds it."""
    import src.config.curation_accessors as curation_mod
    from src.services.projects import capacity as capacity_mod, lifecycle

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(ProjectRegistry())
    with patch(
        'src.services.curation.index_bootstrap.ensure_indexes',
        new=AsyncMock(side_effect=fake_ensure_indexes),
    ):
        record, _ = asyncio.run(
            lifecycle.create_project(FakeLifecycleOpenSearch(), slug='alpha', display_name='Alpha')
        )
    with bind_project(record):
        yield record
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)


def _pending_drains(pg: Any, project: Any) -> list[tuple[Any, ...]]:
    return sql(
        pg,
        "SELECT status FROM jobs WHERE kind = 'open_vocab_drain' AND project_id = %s "
        "AND status IN ('queued', 'running')",
        project.id,
    )


def test_two_starts_make_one_pending_job_and_one_message(pg, project, producer) -> None:
    assert asyncio.run(start_drain()) is True
    assert asyncio.run(start_drain()) is False  # already pending
    assert _pending_drains(pg, project) == [('queued',)]
    assert len(producer.sent()) == 1


def test_two_ingest_requests_report_scheduled_and_queue_one_job(
    pg, project, producer, monkeypatch
) -> None:
    stamped: list[str] = []

    async def active() -> Any:
        return SimpleNamespace(run_on_ingest=True), 1

    async def stamp(_client: Any, image_id: str, _status: str) -> None:
        stamped.append(image_id)

    monkeypatch.setattr(open_vocab_ingest, 'current_active_set', active)
    monkeypatch.setattr(open_vocab_ingest, 'stamp_open_vocab_status', stamp)

    first = asyncio.run(schedule_open_vocab_after_ingest(object(), ['a']))  # type: ignore[arg-type]
    second = asyncio.run(schedule_open_vocab_after_ingest(object(), ['b']))  # type: ignore[arg-type]

    assert (first, second) == (True, True)
    assert stamped == ['a', 'b']
    assert _pending_drains(pg, project) == [('queued',)]
    assert len(producer.sent()) == 1


def test_a_broker_that_is_down_fails_the_start_loudly_and_the_ingest_still_succeeds(
    pg, project, producer, monkeypatch
) -> None:
    producer.fail_send = True
    with pytest.raises(RequestRejectedError) as down:
        asyncio.run(start_drain())
    assert down.value.detail['error'] == 'queue_unavailable'

    async def active() -> Any:
        return SimpleNamespace(run_on_ingest=True), 1

    async def stamp(*_a: Any) -> None:
        return None

    monkeypatch.setattr(open_vocab_ingest, 'current_active_set', active)
    monkeypatch.setattr(open_vocab_ingest, 'stamp_open_vocab_status', stamp)
    assert asyncio.run(schedule_open_vocab_after_ingest(object(), ['a'])) is False  # type: ignore[arg-type]
