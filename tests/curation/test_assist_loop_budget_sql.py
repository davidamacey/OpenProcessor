"""The loop's budget reservation, against the SQL: a randomized crash test and the fences."""

from __future__ import annotations

import random
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from sqlalchemy import text


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'db'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _research_seed import make_loop_run, make_project
from fixtures.job_rows import project_session

from src.db.repos import assist_loop as repo
from src.domain.assist_loop_budget import call_units, settled_amounts
from src.domain.vlm_usage import NO_USAGE, VlmUsage


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

SEEDS = 200
UNITS = ('crops', 'requests', 'prompt_tokens', 'completion_tokens')


@pytest.fixture(scope='module')
def project(pg: PgDatabase) -> uuid.UUID:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        return make_project(conn, f'budget-{uuid.uuid4().hex[:6]}')


def _run(pg: PgDatabase, project: uuid.UUID, token: int = 1) -> tuple[uuid.UUID, uuid.UUID]:
    """A running job and its run."""
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        run = make_loop_run(conn, project)
        job = conn.execute('SELECT job_id FROM assist_loop_runs WHERE id = %s', (run,)).fetchone()[
            0
        ]
        conn.execute(
            "UPDATE jobs SET status = 'running', fencing_token = %s, attempt = 1, "
            "owner = 'test', started_at = now(), heartbeat_at = now(), "
            "lease_expires_at = now() + interval '1 hour' WHERE id = %s",
            (token, job),
        )
    return run, job


def _invariants(session: Any, run: uuid.UUID) -> None:
    rows = repo.budgets(session, run)
    open_reserved = {
        (pool, unit): amount
        for pool, unit, amount in session.execute(
            text(
                'SELECT r.pool, r.unit, sum(r.amount) FROM assist_loop_budget_ledger r '
                "WHERE r.run_id = :run AND r.entry = 'reserve' AND NOT EXISTS ("
                'SELECT 1 FROM assist_loop_budget_ledger c WHERE c.run_id = r.run_id '
                'AND c.request_id = r.request_id AND c.pool = r.pool AND c.unit = r.unit '
                "AND c.entry <> 'reserve') GROUP BY r.pool, r.unit"
            ),
            {'run': run},
        )
    }
    for row in rows:
        assert row.reserved == open_reserved.get((row.pool, row.unit), 0)
        if row.limit_value is not None and row.unit != 'prompt_tokens':
            assert row.spent + row.reserved <= row.limit_value, row


@pytest.mark.parametrize('block', range(SEEDS // 20))
def test_no_limit_is_ever_exceeded_whatever_crashes(pg: PgDatabase, project, block: int) -> None:
    engine = pg.engine('app')
    for seed in range(block * 20, block * 20 + 20):
        rng = random.Random(seed)
        run, job = _run(pg, project)
        limits = {
            ('vlm', unit): rng.randint(5, 120) if unit != 'wall_s' else None for unit in UNITS
        }
        with project_session(engine, project) as session:
            repo.seed_budgets(session, run, limits)
        open_calls: list[tuple[uuid.UUID, dict[str, int]]] = []
        for _step in range(40):
            action = rng.choice(('reserve', 'reserve', 'settle', 'crash'))
            with project_session(engine, project) as session:
                if action == 'reserve':
                    units = call_units(
                        images=rng.randint(1, 4),
                        max_tokens=rng.randint(1, 20),
                        prompt_length_tokens=rng.randint(0, 10),
                    )
                    request = uuid.uuid4()
                    try:
                        repo.reserve(
                            session,
                            run_id=run,
                            job_id=job,
                            token=1,
                            request_id=request,
                            pool='vlm',
                            amounts=units.as_amounts(),
                        )
                    except repo.BudgetExhaustedError:
                        pass
                    else:
                        open_calls.append((request, units.as_amounts()))
                elif action == 'settle' and open_calls:
                    request, reserved = open_calls.pop(rng.randrange(len(open_calls)))
                    usage = (
                        NO_USAGE
                        if rng.random() < 0.3
                        else VlmUsage(rng.randint(0, 300), rng.randint(0, 30), True)
                    )
                    repo.settle(
                        session,
                        run_id=run,
                        job_id=job,
                        token=1,
                        request_id=request,
                        actual=settled_amounts(reserved, usage),
                    )
                elif action == 'crash':
                    open_calls.clear()  # the in-flight calls are forgotten, as a killed slice does
                    repo.forfeit_open(session, run_id=run, job_id=job, token=1)
                _invariants(session, run)
        with project_session(engine, project) as session:
            repo.forfeit_open(session, run_id=run, job_id=job, token=1)
            _invariants(session, run)
            assert all(row.reserved == 0 for row in repo.budgets(session, run))


def test_a_refused_unit_leaves_no_unit_reserved(pg: PgDatabase, project) -> None:
    run, job = _run(pg, project)
    engine = pg.engine('app')
    with project_session(engine, project) as session:
        repo.seed_budgets(
            session,
            run,
            {('vlm', 'crops'): 10, ('vlm', 'requests'): 10, ('vlm', 'completion_tokens'): 5},
        )
        with pytest.raises(repo.BudgetExhaustedError) as refused:
            repo.reserve(
                session,
                run_id=run,
                job_id=job,
                token=1,
                request_id=uuid.uuid4(),
                pool='vlm',
                amounts={'crops': 3, 'requests': 1, 'completion_tokens': 6},
            )
        assert refused.value.unit == 'completion_tokens'
        assert all(row.reserved == 0 for row in repo.budgets(session, run))


def test_a_settle_replayed_changes_nothing(pg: PgDatabase, project) -> None:
    run, job = _run(pg, project)
    request = uuid.uuid4()
    with project_session(pg.engine('app'), project) as session:
        repo.seed_budgets(session, run, {('vlm', 'crops'): 10})
        repo.reserve(
            session, run_id=run, job_id=job, token=1, request_id=request, pool='vlm',
            amounts={'crops': 4},
        )  # fmt: skip
        assert repo.settle(
            session, run_id=run, job_id=job, token=1, request_id=request, actual={'crops': 4}
        )
        assert not repo.settle(
            session, run_id=run, job_id=job, token=1, request_id=request, actual={'crops': 4}
        )
        (row,) = repo.budgets(session, run)
        assert (row.reserved, row.spent) == (0, 4)


def test_a_slice_that_lost_the_lease_writes_nothing(pg: PgDatabase, project) -> None:
    run, job = _run(pg, project, token=2)
    with project_session(pg.engine('app'), project) as session:
        repo.seed_budgets(session, run, {('vlm', 'crops'): 10})
        with pytest.raises(repo.LostFenceError):
            repo.reserve(
                session, run_id=run, job_id=job, token=1, request_id=uuid.uuid4(), pool='vlm',
                amounts={'crops': 1},
            )  # fmt: skip
        (row,) = repo.budgets(session, run)
        assert row.reserved == 0


def test_a_limit_below_what_is_spent_is_refused(pg: PgDatabase, project) -> None:
    run, job = _run(pg, project)
    with project_session(pg.engine('app'), project) as session:
        repo.seed_budgets(session, run, {('vlm', 'crops'): 10})
        repo.add_spent(session, run_id=run, job_id=job, token=1, pool='vlm', unit='crops', amount=6)
        with pytest.raises(repo.BudgetBelowSpentError) as refused:
            repo.edit_limit(session, run_id=run, pool='vlm', unit='crops', limit_value=5)
        assert refused.value.floor == 6
        repo.edit_limit(session, run_id=run, pool='vlm', unit='crops', limit_value=6)
        with pytest.raises(ValueError, match='required'):
            repo.edit_limit(session, run_id=run, pool='vlm', unit='crops', limit_value=None)
