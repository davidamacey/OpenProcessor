"""The start gate (R-9a, 2E-7): 0.7.0 processes refuse to run while the 0.6.0 upgrade is pending."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from src.db import schema_check
from src.db.schema_check import (
    DependencyUnavailableError,
    SchemaNotCurrentError,
    UpgradeIncompleteError,
    check_control_plane,
)
from src.db.session import async_session_factory, deployment_tx
from src.services.upgrade import gate
from src.services.upgrade.from_060.reset import reset
from src.services.upgrade.from_060.runner import UpgradeResult, VerifyFailedError


pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def _no_legacy_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schema_check, 'legacy_projects_index_exists', lambda: False)


@pytest.fixture(autouse=True)
def _clean_bookkeeping(pg) -> None:
    with pg.engine('super').begin() as conn:
        conn.execute(text('DELETE FROM upgrade_runs'))
        conn.execute(text("DELETE FROM deployment_settings WHERE key = 'upgrade_state'"))


async def _set(state: str | None) -> None:
    async with async_session_factory()() as session, deployment_tx(session, 'test gate'):
        if state == gate.IN_PROGRESS:
            await gate.mark_in_progress(session)
        elif state == gate.COMPLETED:
            await gate.mark_in_progress(session)
            await gate.mark_completed(session)


@pytest.mark.asyncio
async def test_a_fresh_database_starts(pg_env) -> None:
    check_control_plane()


@pytest.mark.asyncio
async def test_an_upgrade_in_progress_refuses_the_start(pg_env) -> None:
    await _set(gate.IN_PROGRESS)
    with pytest.raises(UpgradeIncompleteError) as caught:
        check_control_plane()
    assert str(caught.value).startswith('upgrade_incomplete')
    assert isinstance(
        caught.value, SchemaNotCurrentError
    )  # every caller that refuses on a stale schema refuses on this


@pytest.mark.asyncio
async def test_a_completed_upgrade_starts(pg_env) -> None:
    await _set(gate.COMPLETED)
    check_control_plane()


@pytest.mark.asyncio
async def test_the_legacy_projects_index_refuses_the_start_even_with_no_state(
    pg_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(schema_check, 'legacy_projects_index_exists', lambda: True)
    with pytest.raises(UpgradeIncompleteError, match='op_projects'):
        check_control_plane()


@pytest.mark.asyncio
async def test_an_unreachable_opensearch_is_a_refusal_naming_it(
    pg_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreachable() -> bool:
        raise DependencyUnavailableError(['opensearch'], 'ConnectError')

    monkeypatch.setattr(schema_check, 'legacy_projects_index_exists', unreachable)
    with pytest.raises(DependencyUnavailableError, match='opensearch'):
        check_control_plane()


@pytest.mark.asyncio
async def test_a_completed_upgrade_cannot_be_started_again(pg_env) -> None:
    await _set(gate.COMPLETED)
    async with async_session_factory()() as session:
        with pytest.raises(gate.UpgradeStateError):
            async with deployment_tx(session, 'test gate'):
                await gate.mark_in_progress(session)


@pytest.mark.asyncio
async def test_a_verify_mismatch_deletes_nothing_and_keeps_the_gate_closed(
    make_runner, clean_db, v060, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trap T-25: the sources are deleted only after verify passed."""
    runner = make_runner()
    await runner._begin()
    result = UpgradeResult(bundle=await runner._export())
    await runner._load(result)
    with clean_db.engine('super').begin() as conn:  # a row lost between load and verify
        conn.execute(text("DELETE FROM project_classes WHERE name = 'Person'"))
    before = {
        p: p.read_bytes() for p in v060.root.rglob('*') if p.is_file() and 'storage' not in p.parts
    }
    indexes = set(v060.os.indexes)

    with pytest.raises(VerifyFailedError) as caught:
        await runner._verify(result)
    assert any('project_classes' in p for p in caught.value.problems)
    assert caught.value.report_path.is_file()

    assert set(v060.os.indexes) == indexes
    assert before == {p: p.read_bytes() for p in before}
    with clean_db.engine('super').connect() as conn:
        steps = conn.execute(
            text('SELECT step, status FROM upgrade_runs ORDER BY step, status')
        ).all()
    assert ('verify', 'failed') in steps
    assert ('delete_sources', 'completed') not in steps
    with pytest.raises(UpgradeIncompleteError):
        check_control_plane()


@pytest.mark.asyncio
async def test_reset_after_a_failed_verify_gives_the_same_rows_on_the_rerun(
    make_runner, clean_db, v060
) -> None:
    runner = make_runner()
    await runner._begin()
    first = UpgradeResult(bundle=await runner._export())
    await runner._load(first)
    with clean_db.engine('super').connect() as conn:
        before = conn.execute(text('SELECT slug, display_name FROM projects ORDER BY 1')).all()
        classes = conn.execute(
            text(
                'SELECT p.slug, c.class_id, c.name FROM project_classes c JOIN projects p ON p.id = c.project_id ORDER BY 1, 2'
            )
        ).all()
    with clean_db.engine('super').begin() as conn:
        conn.execute(text("DELETE FROM project_classes WHERE name = 'Person'"))
    with pytest.raises(VerifyFailedError):
        await runner._verify(first)

    with clean_db.engine('owner').connect() as conn:
        reset(conn)
    with clean_db.engine('super').connect() as conn:
        assert conn.execute(text('SELECT count(*) FROM projects')).scalar_one() == 0
        assert conn.execute(text('SELECT count(*) FROM upgrade_runs')).scalar_one() == 0
        assert (
            conn.execute(
                text(
                    "SELECT value ->> 'state' FROM deployment_settings WHERE key = 'upgrade_state'"
                )
            ).scalar_one()
            == 'in_progress'
        )

    await make_runner().run()  # the sources are still there: export, load, verify, delete
    with clean_db.engine('super').connect() as conn:
        assert (
            conn.execute(text('SELECT slug, display_name FROM projects ORDER BY 1')).all() == before
        )
        assert (
            conn.execute(
                text(
                    'SELECT p.slug, c.class_id, c.name FROM project_classes c JOIN projects p ON p.id = c.project_id ORDER BY 1, 2'
                )
            ).all()
            == classes
        )


@pytest.mark.asyncio
async def test_reset_after_delete_sources_is_refused(make_runner, clean_db) -> None:
    await make_runner().run()
    with (
        clean_db.engine('owner').connect() as conn,
        pytest.raises(gate.UpgradeStateError, match='delete_sources'),
    ):
        reset(conn)
    with clean_db.engine('super').connect() as conn:
        assert (
            conn.execute(text('SELECT count(*) FROM projects')).scalar_one() == 2
        )  # nothing was truncated
