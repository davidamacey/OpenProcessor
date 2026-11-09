"""``scripts.db.vlm_pause``: what ``openprocessor vlm use`` runs inside the API container."""

from __future__ import annotations

import time

import pytest
from fixtures.job_rows import sql
from psycopg.types.json import Jsonb

from scripts.db import vlm_pause
from src.domain import gpu_arbiter_state as state
from src.services.jobs import blockers
from src.services.jobs.gpu_pause import vlm_paused


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.fixture
def clean(pg):
    sql(pg, 'DELETE FROM deployment_settings WHERE key = %s', state.KEY)
    return pg


def document(pg):
    rows = sql(pg, 'SELECT value FROM deployment_settings WHERE key = %s', state.KEY)
    return rows[0][0] if rows else None


def test_pause_create_marks_a_swap_and_a_second_swap_finds_it_present(clean, capsys) -> None:
    assert vlm_pause.run('pause-create') == 0
    assert capsys.readouterr().out.strip() == 'created'
    assert state.switch_active(document(clean))

    assert vlm_pause.run('pause-create') == 0
    assert capsys.readouterr().out.strip() == 'present'


def test_pause_remove_clears_only_the_swap_section(clean) -> None:
    sql(
        clean,
        'INSERT INTO deployment_settings (key, value, updated_by) VALUES (%s, %s, %s)',
        state.KEY,
        Jsonb(
            {
                state.ARBITER: state.ArbiterRecord(claims=('job-1',)).to_json(),
                state.VLM_SWITCH: {'at': time.time()},
            }
        ),
        'test',
    )

    assert vlm_pause.run('pause-remove') == 0

    value = document(clean)
    assert state.VLM_SWITCH not in value
    assert state.arbiter_record(value).claims == ('job-1',)


def test_a_stale_swap_can_be_taken_over(clean, capsys) -> None:
    sql(
        clean,
        'INSERT INTO deployment_settings (key, value, updated_by) VALUES (%s, %s, %s)',
        state.KEY,
        Jsonb({state.VLM_SWITCH: {'at': time.time() - state.SWITCH_TTL_S - 5}}),
        'test',
    )

    assert vlm_pause.run('pause-create') == 0

    assert capsys.readouterr().out.strip() == 'created'


def test_lock_check_exits_zero_only_while_a_run_holds_cards(clean) -> None:
    assert vlm_pause.run('lock-check') == 1
    sql(
        clean,
        'INSERT INTO deployment_settings (key, value, updated_by) VALUES (%s, %s, %s)',
        state.KEY,
        Jsonb({state.ARBITER: state.ArbiterRecord(claims=('job-1',)).to_json()}),
        'test',
    )
    assert vlm_pause.run('lock-check') == 0


def test_an_unknown_action_is_an_error(clean) -> None:
    assert vlm_pause.run('nope') == 2


@pytest.mark.asyncio
async def test_the_workers_and_the_reaper_read_the_same_document(clean, pg) -> None:
    assert await vlm_paused() is False
    vlm_pause.run('pause-create')
    assert await vlm_paused() is True

    from fixtures.job_rows import deployment_session

    with deployment_session(pg.engine()) as session:
        assert blockers.vlm_available(session) is False  # taken, whatever the probes say
