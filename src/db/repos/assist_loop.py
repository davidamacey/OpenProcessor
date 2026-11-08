"""The assist loop's budget tables: reserve, settle, forfeit and edit, every statement fenced.

Every write carries the job's fencing token: a slice that lost its lease writes nothing and
raises :class:`LostFenceError`. One reservation is one upstream call: it reserves all of the
call's units in a savepoint, so a refusal on any unit leaves no unit reserved. The ledger is
the record of every movement (``reserve``, ``settle``, ``release``, ``forfeit``); a request is
open while it has ``reserve`` rows and no closing row for the same unit, and closing is
idempotent, so a replayed settle changes nothing.

Functions take the caller's sync session inside its project transaction and never commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import text


if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping

    from sqlalchemy.orm import Session


class LostFenceError(RuntimeError):
    """The job's fencing token moved on: this slice is no longer the owner."""


class BudgetExhaustedError(RuntimeError):
    """A reservation would take ``spent + reserved`` past a limit."""

    def __init__(self, pool: str, unit: str) -> None:
        super().__init__(f'budget_exhausted: {pool}.{unit}')
        self.pool = pool
        self.unit = unit


class BudgetBelowSpentError(RuntimeError):
    """A limit edit below what is already spent plus reserved (409 ``budget_below_spent``)."""

    def __init__(self, pool: str, unit: str, floor: int) -> None:
        super().__init__(f'budget_below_spent: {pool}.{unit} floor {floor}')
        self.pool = pool
        self.unit = unit
        self.floor = floor


@dataclass(frozen=True)
class BudgetRow:
    pool: str
    unit: str
    limit_value: int | None
    reserved: int
    spent: int


_FENCE = text(
    'SELECT 1 FROM jobs WHERE id = :job AND fencing_token = :token '
    "AND status IN ('running', 'cancelling') FOR SHARE"
)
_SEED = text(
    'INSERT INTO assist_loop_budgets (run_id, pool, unit, limit_value) '
    'VALUES (:run, :pool, :unit, :limit) ON CONFLICT DO NOTHING'
)
_RESERVE = text(
    'UPDATE assist_loop_budgets SET reserved = reserved + :n, updated_at = now() '
    'WHERE run_id = :run AND pool = :pool AND unit = :unit '
    'AND (limit_value IS NULL OR spent + reserved + :n <= limit_value) RETURNING 1'
)
_LEDGER = text(
    'INSERT INTO assist_loop_budget_ledger (run_id, request_id, pool, unit, entry, amount, '
    'fencing_token) VALUES (:run, :request, :pool, :unit, :entry, :amount, :token)'
)
_OPEN_OF_REQUEST = text(
    'SELECT r.pool, r.unit, r.amount FROM assist_loop_budget_ledger r '
    "WHERE r.run_id = :run AND r.request_id = :request AND r.entry = 'reserve' "
    'AND NOT EXISTS (SELECT 1 FROM assist_loop_budget_ledger c WHERE c.run_id = r.run_id '
    'AND c.request_id = r.request_id AND c.pool = r.pool AND c.unit = r.unit '
    "AND c.entry IN ('settle', 'release', 'forfeit'))"
)
_OPEN_OF_RUN = text(
    'SELECT r.request_id, r.pool, r.unit, r.amount FROM assist_loop_budget_ledger r '
    "WHERE r.run_id = :run AND r.entry = 'reserve' "
    'AND NOT EXISTS (SELECT 1 FROM assist_loop_budget_ledger c WHERE c.run_id = r.run_id '
    'AND c.request_id = r.request_id AND c.pool = r.pool AND c.unit = r.unit '
    "AND c.entry IN ('settle', 'release', 'forfeit'))"
)
_MOVE = text(
    'UPDATE assist_loop_budgets SET reserved = reserved - :reserved, spent = spent + :spent, '
    'updated_at = now() WHERE run_id = :run AND pool = :pool AND unit = :unit'
)
_ADD_SPENT = text(
    'UPDATE assist_loop_budgets SET spent = spent + :n, updated_at = now() '
    'WHERE run_id = :run AND pool = :pool AND unit = :unit'
)
_ROWS = text(
    'SELECT pool, unit, limit_value, reserved, spent FROM assist_loop_budgets '
    'WHERE run_id = :run ORDER BY pool, unit'
)
_EDIT = text(
    'UPDATE assist_loop_budgets SET limit_value = :limit, updated_at = now() '
    'WHERE run_id = :run AND pool = :pool AND unit = :unit '
    'AND (CAST(:limit AS bigint) IS NULL OR CAST(:limit AS bigint) >= spent + reserved) '
    'RETURNING 1'
)
_FLOOR = text(
    'SELECT spent + reserved FROM assist_loop_budgets '
    'WHERE run_id = :run AND pool = :pool AND unit = :unit'
)


def assert_fenced(session: Session, job_id: uuid.UUID, token: int) -> None:
    if session.execute(_FENCE, {'job': job_id, 'token': token}).first() is None:
        raise LostFenceError(f'job {job_id} no longer holds fencing token {token}')


def seed_budgets(
    session: Session, run_id: uuid.UUID, limits: Mapping[tuple[str, str], int | None]
) -> None:
    """One row per ``(pool, unit)`` of a new run; an existing row is left alone."""
    for (pool, unit), limit in limits.items():
        session.execute(_SEED, {'run': run_id, 'pool': pool, 'unit': unit, 'limit': limit})


def reserve(
    session: Session,
    *,
    run_id: uuid.UUID,
    job_id: uuid.UUID,
    token: int,
    request_id: uuid.UUID,
    pool: str,
    amounts: Mapping[str, int],
) -> None:
    """Reserve every unit of one call or none. Raises :class:`BudgetExhaustedError` naming the
    unit that refused."""
    assert_fenced(session, job_id, token)
    with session.begin_nested():
        for unit, amount in amounts.items():
            params = {'run': run_id, 'pool': pool, 'unit': unit, 'n': amount}
            if session.execute(_RESERVE, params).first() is None:
                raise BudgetExhaustedError(pool, unit)
            session.execute(
                _LEDGER,
                {
                    'run': run_id,
                    'request': request_id,
                    'pool': pool,
                    'unit': unit,
                    'entry': 'reserve',
                    'amount': amount,
                    'token': token,
                },
            )


def _close(
    session: Session,
    *,
    run_id: uuid.UUID,
    token: int,
    request_id: uuid.UUID,
    pool: str,
    unit: str,
    reserved: int,
    spent: int,
    entry: str,
) -> None:
    session.execute(
        _MOVE, {'run': run_id, 'pool': pool, 'unit': unit, 'reserved': reserved, 'spent': spent}
    )
    session.execute(
        _LEDGER,
        {
            'run': run_id,
            'request': request_id,
            'pool': pool,
            'unit': unit,
            'entry': entry,
            'amount': spent if entry in ('settle', 'forfeit') else reserved,
            'token': token,
        },
    )


def settle(
    session: Session,
    *,
    run_id: uuid.UUID,
    job_id: uuid.UUID,
    token: int,
    request_id: uuid.UUID,
    actual: Mapping[str, int],
) -> bool:
    """Move a call's open reservation to spent at the amounts in ``actual``. A unit not in
    ``actual`` is released unspent. ``False`` when the request has nothing open (a replay)."""
    assert_fenced(session, job_id, token)
    open_rows = session.execute(_OPEN_OF_REQUEST, {'run': run_id, 'request': request_id}).all()
    for pool, unit, reserved in open_rows:
        spent = actual.get(unit)
        _close(
            session,
            run_id=run_id,
            token=token,
            request_id=request_id,
            pool=pool,
            unit=unit,
            reserved=reserved,
            spent=0 if spent is None else spent,
            entry='release' if spent is None else 'settle',
        )
    return bool(open_rows)


def forfeit_open(session: Session, *, run_id: uuid.UUID, job_id: uuid.UUID, token: int) -> int:
    """At a resume or a slice start: every reservation with no closing row is counted spent,
    because its call may have reached the endpoint. Returns how many reservations were moved."""
    assert_fenced(session, job_id, token)
    rows = session.execute(_OPEN_OF_RUN, {'run': run_id}).all()
    for request_id, pool, unit, reserved in rows:
        _close(
            session,
            run_id=run_id,
            token=token,
            request_id=request_id,
            pool=pool,
            unit=unit,
            reserved=reserved,
            spent=reserved,
            entry='forfeit',
        )
    return len(rows)


def add_spent(
    session: Session,
    *,
    run_id: uuid.UUID,
    job_id: uuid.UUID,
    token: int,
    pool: str,
    unit: str,
    amount: int,
) -> None:
    """Charge ``amount`` straight to spent (wall time of a slice)."""
    assert_fenced(session, job_id, token)
    session.execute(_ADD_SPENT, {'run': run_id, 'pool': pool, 'unit': unit, 'n': amount})


def budgets(session: Session, run_id: uuid.UUID) -> list[BudgetRow]:
    return [BudgetRow(*row) for row in session.execute(_ROWS, {'run': run_id}).all()]


def edit_limit(
    session: Session, *, run_id: uuid.UUID, pool: str, unit: str, limit_value: int | None
) -> None:
    """Set one limit (``None``: uncapped). Raises :class:`BudgetBelowSpentError` when it is
    below ``spent + reserved``."""
    if limit_value is None and (pool, unit) == ('vlm', 'crops'):
        raise ValueError('the vlm crop limit is required and cannot be removed')
    params = {'run': run_id, 'pool': pool, 'unit': unit, 'limit': limit_value}
    if session.execute(_EDIT, params).first() is None:
        floor = session.execute(_FLOOR, params).scalar_one()
        raise BudgetBelowSpentError(pool, unit, int(floor))
