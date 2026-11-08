"""The assist loop's record: runs, iterations, nodes, selections and proposals.

Every write is fenced by the job that owns the run (:func:`~src.db.repos.assist_loop.assert_fenced`);
a read needs no fence. A node's ``step`` only moves forward (open, admitted, gated, propagating,
done) and its decision is written together with ``done``. Functions take the caller's sync
session inside its project transaction and never commit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.db.repos.assist_loop import assert_fenced, seed_budgets


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Mapping, Sequence

    from sqlalchemy.orm import Session


NODE_COLUMNS = frozenset(
    {
        'step',
        'decision',
        'class_name',
        'answered',
        'agree',
        'bound',
        'distribution',
        'admitted_crops',
        'admitted_requests',
        'label_op_id',
        'items_propagated',
        'skipped',
    }
)
_JSON_COLUMNS = frozenset({'distribution', 'skipped'})


@dataclass(frozen=True)
class RunRow:
    id: uuid.UUID
    project_id: uuid.UUID
    job_id: uuid.UUID
    request: dict[str, Any]
    strategy: str
    seed: int
    pinned: dict[str, Any]
    stop_reason: str | None
    created_at: dt.datetime
    created_by: str


@dataclass(frozen=True)
class NodeRow:
    path: str
    iteration: int
    parent_path: str | None
    level: int
    member_count: int
    eligible_count: int
    membership_digest: str
    snapshot_storage_key: str
    step: str
    decision: str | None
    class_name: str | None
    answered: int | None
    agree: int | None
    bound: float | None
    distribution: dict[str, Any] | None
    admitted_crops: int | None
    admitted_requests: int | None
    label_op_id: uuid.UUID | None
    items_propagated: int
    skipped: dict[str, Any]


@dataclass(frozen=True)
class NewNode:
    path: str
    parent_path: str | None
    level: int
    member_count: int
    eligible_count: int
    membership_digest: str
    snapshot_storage_key: str


@dataclass(frozen=True)
class SelectionRow:
    item_id: str
    purpose: str
    node_path: str | None
    iteration: int
    draw_rank: int


@dataclass(frozen=True)
class ProposalRow:
    item_id: str
    role: str
    request_id: uuid.UUID
    outcome: str
    class_name: str | None
    new_class_name: str | None
    proposer_ref: str = ''
    item_write: str | None = None


_RUN_BY_ID = text(
    'SELECT id, project_id, job_id, request, strategy, seed, pinned, stop_reason, created_at, created_by FROM assist_loop_runs WHERE id = :id'
)
_RUN_BY_JOB = text(
    'SELECT id, project_id, job_id, request, strategy, seed, pinned, stop_reason, created_at, created_by FROM assist_loop_runs WHERE job_id = :id'
)
_RUN_LIST = text(
    'SELECT id, project_id, job_id, request, strategy, seed, pinned, stop_reason, created_at, created_by FROM assist_loop_runs WHERE project_id = :project '
    'ORDER BY created_at DESC, id DESC LIMIT :limit'
)
_NODE_SELECT = (
    'SELECT path, iteration, parent_path, level, member_count, eligible_count, '
    'membership_digest, snapshot_storage_key, step, decision, class_name, answered, agree, '
    'bound, distribution, admitted_crops, admitted_requests, label_op_id, items_propagated, '
    'skipped FROM assist_loop_nodes WHERE run_id = :run ORDER BY level, path'
)


def create_run(
    session: Session,
    *,
    project_id: uuid.UUID,
    job_id: uuid.UUID,
    request: Mapping[str, Any],
    pinned: Mapping[str, Any],
    seed: int,
    created_by: str,
    limits: Mapping[tuple[str, str], int | None],
) -> uuid.UUID:
    """The run of a freshly queued job: its budgets and its first iteration, in the caller's
    transaction (``start_job``'s ``with_rows``)."""
    run_id = session.execute(
        text(
            'INSERT INTO assist_loop_runs (project_id, job_id, request, strategy, seed, pinned, '
            'created_by) VALUES (:project, :job, CAST(:request AS jsonb), :strategy, :seed, '
            'CAST(:pinned AS jsonb), :by) RETURNING id'
        ),
        {
            'project': project_id,
            'job': job_id,
            'request': json.dumps(request),
            'strategy': 'cluster_sampled',
            'seed': seed,
            'pinned': json.dumps(pinned),
            'by': created_by,
        },
    ).scalar_one()
    seed_budgets(session, run_id, limits)
    session.execute(
        text(
            "INSERT INTO assist_loop_iterations (run_id, ordinal, status) VALUES (:run, 1, 'open')"
        ),
        {'run': run_id},
    )
    return run_id  # type: ignore[no-any-return]


def _run(row: Any) -> RunRow:
    return RunRow(*row)


def run_by_id(session: Session, run_id: uuid.UUID) -> RunRow | None:
    row = session.execute(_RUN_BY_ID, {'id': run_id}).first()
    return None if row is None else _run(row)


def run_by_job(session: Session, job_id: uuid.UUID) -> RunRow | None:
    row = session.execute(_RUN_BY_JOB, {'id': job_id}).first()
    return None if row is None else _run(row)


def list_runs(session: Session, project_id: uuid.UUID, *, limit: int) -> list[RunRow]:
    rows = session.execute(
        _RUN_LIST,
        {'project': project_id, 'limit': limit},
    ).all()
    return [_run(row) for row in rows]


def set_pinned(
    session: Session, run_id: uuid.UUID, *, job_id: uuid.UUID, token: int, pinned: Mapping[str, Any]
) -> None:
    assert_fenced(session, job_id, token)
    session.execute(
        text('UPDATE assist_loop_runs SET pinned = CAST(:pinned AS jsonb) WHERE id = :run'),
        {'run': run_id, 'pinned': json.dumps(pinned)},
    )


def set_stop_reason(
    session: Session, run_id: uuid.UUID, *, job_id: uuid.UUID, token: int, reason: str
) -> None:
    assert_fenced(session, job_id, token)
    session.execute(
        text('UPDATE assist_loop_runs SET stop_reason = :reason WHERE id = :run'),
        {'run': run_id, 'reason': reason},
    )


def iteration_counts(session: Session, run_id: uuid.UUID) -> dict[str, Any]:
    row = session.execute(
        text(
            'SELECT counts FROM assist_loop_iterations WHERE run_id = :run '
            'ORDER BY ordinal DESC LIMIT 1'
        ),
        {'run': run_id},
    ).scalar_one_or_none()
    return dict(row or {})


def set_iteration_counts(
    session: Session,
    run_id: uuid.UUID,
    *,
    job_id: uuid.UUID,
    token: int,
    counts: Mapping[str, Any],
    close: bool = False,
) -> None:
    assert_fenced(session, job_id, token)
    session.execute(
        text(
            'UPDATE assist_loop_iterations SET counts = CAST(:counts AS jsonb), '
            "status = CASE WHEN :close THEN 'closed' ELSE status END, "
            'closed_at = CASE WHEN :close THEN now() ELSE closed_at END '
            'WHERE run_id = :run AND ordinal = (SELECT max(ordinal) FROM assist_loop_iterations '
            'WHERE run_id = :run)'
        ),
        {'run': run_id, 'counts': json.dumps(counts), 'close': close},
    )


def insert_nodes(
    session: Session,
    run_id: uuid.UUID,
    *,
    job_id: uuid.UUID,
    token: int,
    iteration: int,
    nodes: Sequence[NewNode],
) -> None:
    """New nodes at step ``open``; an existing path is left alone (a replayed insert)."""
    assert_fenced(session, job_id, token)
    for node in nodes:
        session.execute(
            text(
                'INSERT INTO assist_loop_nodes (run_id, path, iteration, parent_path, level, '
                'member_count, eligible_count, membership_digest, snapshot_storage_key, step) '
                'VALUES (:run, :path, :iteration, :parent, :level, :members, :eligible, :digest, '
                ":key, 'open') ON CONFLICT DO NOTHING"
            ),
            {
                'run': run_id,
                'path': node.path,
                'iteration': iteration,
                'parent': node.parent_path,
                'level': node.level,
                'members': node.member_count,
                'eligible': node.eligible_count,
                'digest': node.membership_digest,
                'key': node.snapshot_storage_key,
            },
        )


def update_node(
    session: Session,
    run_id: uuid.UUID,
    path: str,
    *,
    job_id: uuid.UUID,
    token: int,
    **columns: Any,
) -> None:
    """Set columns of one node (only :data:`NODE_COLUMNS`)."""
    unknown = set(columns) - NODE_COLUMNS
    if unknown:
        raise ValueError(f'not node columns: {sorted(unknown)}')
    assert_fenced(session, job_id, token)
    sets = ', '.join(
        f'{col} = CAST(:{col} AS jsonb)' if col in _JSON_COLUMNS else f'{col} = :{col}'
        for col in columns
    )
    values = {
        col: json.dumps(val) if col in _JSON_COLUMNS and val is not None else val
        for col, val in columns.items()
    }
    session.execute(
        text(
            f'UPDATE assist_loop_nodes SET {sets}, updated_at = now() '  # nosec B608 - columns are the NODE_COLUMNS whitelist
            'WHERE run_id = :run AND path = :path'
        ),
        {'run': run_id, 'path': path, **values},
    )


def split_node(
    session: Session,
    run_id: uuid.UUID,
    path: str,
    *,
    job_id: uuid.UUID,
    token: int,
    iteration: int,
    decision_columns: Mapping[str, Any],
    children: Sequence[NewNode],
) -> None:
    """The children and the parent's ``split`` decision in one savepoint: a crash leaves the
    node undecided or fully split, never half."""
    with session.begin_nested():
        insert_nodes(
            session, run_id, job_id=job_id, token=token, iteration=iteration, nodes=children
        )
        update_node(
            session,
            run_id,
            path,
            job_id=job_id,
            token=token,
            step='done',
            decision='split',
            **decision_columns,
        )


def nodes(session: Session, run_id: uuid.UUID) -> list[NodeRow]:
    return [NodeRow(*row) for row in session.execute(text(_NODE_SELECT), {'run': run_id}).all()]


def insert_selections(
    session: Session,
    run_id: uuid.UUID,
    *,
    job_id: uuid.UUID,
    token: int,
    rows: Sequence[SelectionRow],
) -> None:
    """What the loop is about to ask; written before the request is sent, so a crash leaves the
    ids on record and a resume asks only what has no answer."""
    assert_fenced(session, job_id, token)
    for row in rows:
        session.execute(
            text(
                'INSERT INTO assist_loop_selections (run_id, item_id, purpose, node_path, '
                'iteration, draw_rank) VALUES (:run, :item, :purpose, :node, :iteration, :rank) '
                'ON CONFLICT DO NOTHING'
            ),
            {
                'run': run_id,
                'item': row.item_id,
                'purpose': row.purpose,
                'node': row.node_path,
                'iteration': row.iteration,
                'rank': row.draw_rank,
            },
        )


def selections(session: Session, run_id: uuid.UUID) -> list[SelectionRow]:
    rows = session.execute(
        text(
            'SELECT item_id, purpose, node_path, iteration, draw_rank '
            'FROM assist_loop_selections WHERE run_id = :run ORDER BY node_path, draw_rank'
        ),
        {'run': run_id},
    ).all()
    return [SelectionRow(*row) for row in rows]


def insert_proposals(
    session: Session,
    run_id: uuid.UUID,
    *,
    job_id: uuid.UUID,
    token: int,
    rows: Sequence[ProposalRow],
) -> None:
    """The labeler's answers; an item already answered by this role keeps its first answer."""
    assert_fenced(session, job_id, token)
    for row in rows:
        session.execute(
            text(
                'INSERT INTO assist_loop_proposals (run_id, item_id, role, proposer_kind, '
                'proposer_ref, request_id, outcome, class_name, new_class_name, item_write) '
                "VALUES (:run, :item, :role, 'vlm', :ref, :request, :outcome, :class_name, "
                ':new_class, :write) ON CONFLICT DO NOTHING'
            ),
            {
                'run': run_id,
                'item': row.item_id,
                'role': row.role,
                'ref': row.proposer_ref,
                'request': row.request_id,
                'outcome': row.outcome,
                'class_name': row.class_name,
                'new_class': row.new_class_name,
                'write': row.item_write,
            },
        )


def delete_failed_proposals(
    session: Session, run_id: uuid.UUID, *, job_id: uuid.UUID, token: int, ids: Sequence[str]
) -> None:
    """Forget ``failed`` answers so the items can be asked again (a VLM outage, not a verdict)."""
    assert_fenced(session, job_id, token)
    session.execute(
        text(
            "DELETE FROM assist_loop_proposals WHERE run_id = :run AND outcome = 'failed' "
            "AND role = 'primary' AND item_id = ANY(:ids)"
        ),
        {'run': run_id, 'ids': list(ids)},
    )


def proposals(session: Session, run_id: uuid.UUID, *, role: str | None = None) -> list[ProposalRow]:
    rows = session.execute(
        text(
            'SELECT item_id, role, request_id, outcome, class_name, new_class_name, proposer_ref, '
            'item_write FROM assist_loop_proposals WHERE run_id = :run '
            'AND (CAST(:role AS text) IS NULL OR role = :role) ORDER BY item_id'
        ),
        {'run': run_id, 'role': role},
    ).all()
    return [ProposalRow(*row) for row in rows]


def prompt_history(session: Session, run_id: uuid.UUID, *, calls: int) -> list[float]:
    """Prompt tokens per image of the last ``calls`` settled calls, oldest first, read from the
    ledger: a settled call has a ``settle`` row for crops and one for prompt tokens."""
    rows = session.execute(
        text(
            'SELECT p.amount::float8 / NULLIF(c.amount, 0) FROM assist_loop_budget_ledger p '
            'JOIN assist_loop_budget_ledger c ON c.run_id = p.run_id AND c.request_id = p.request_id '
            "AND c.pool = p.pool AND c.unit = 'crops' AND c.entry = 'settle' "
            "WHERE p.run_id = :run AND p.unit = 'prompt_tokens' AND p.entry = 'settle' "
            'AND c.amount > 0 ORDER BY p.id DESC LIMIT :calls'
        ),
        {'run': run_id, 'calls': calls},
    ).all()
    return [float(r[0]) for r in reversed(rows) if r[0] is not None]


def node_ops(session: Session, run_id: uuid.UUID) -> dict[str, str]:
    """``{node path: label operation id}`` of the nodes whose propagation was recorded."""
    found = session.execute(
        text(
            'SELECT path, label_op_id FROM assist_loop_nodes '
            'WHERE run_id = :run AND label_op_id IS NOT NULL'
        ),
        {'run': run_id},
    ).all()
    return {path: str(op) for path, op in found}


def slices(session: Session, job_id: uuid.UUID) -> int:
    """Deliveries of the loop's job so far (attempts, yields included)."""
    return int(
        session.execute(
            text('SELECT count(*) FROM job_attempts WHERE job_id = :job'), {'job': job_id}
        ).scalar_one()
    )
