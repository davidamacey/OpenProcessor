"""The cluster-sampled auto-label loop record: runs, iterations, nodes, selections, proposals,
reviews, budgets and the budget ledger, plus the hourly VLM usage every caller writes.

Item ids are OpenSearch document ids; class identity is the normalized name, never an id.
"""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    PrimaryKeyConstraint,
    SmallInteger,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.types import EMPTY_OBJECT, NOW, UUIDV7, check, in_list


STRATEGIES = ('cluster_sampled',)
STOP_REASONS = (
    'done',
    'budget_crops',
    'budget_requests',
    'budget_tokens',
    'budget_time',
    'vlm_failed',
)
ITERATION_STATUSES = ('open', 'closed')
NODE_STEPS = ('open', 'admitted', 'asked', 'gated', 'propagating', 'done')
NODE_DECISIONS = ('pass', 'split', 'review_leaf', 'review_small', 'stale', 'unvisited')
SELECTION_PURPOSES = ('node_draw', 'node_topup', 'redraw', 'audit', 'blind_audit')
PROPOSAL_ROLES = ('primary', 'audit')
PROPOSER_KINDS = ('vlm',)
PROPOSAL_OUTCOMES = ('answered', 'unreadable', 'failed')
ITEM_WRITES = ('written', 'locked', 'token_changed', 'not_attempted')
REVIEW_QUEUES = ('verify', 'blind_audit')
MACHINE_SOURCES = ('propagated', 'direct', 'proposal_only')
REVIEW_DECISIONS = ('accept', 'correct', 'reject', 'skip')
BUDGET_POOLS = ('vlm', 'audit')
BUDGET_UNITS = ('crops', 'requests', 'prompt_tokens', 'completion_tokens', 'wall_s')
LEDGER_ENTRIES = ('reserve', 'settle', 'release', 'forfeit')
VLM_USAGE_SOURCES = (
    'assist_loop',
    'assist_run',
    'autolabel',
    'vlm_worker',
    'region_verify',
    'label_batch',
    'open_vocab_gate',
)

_RUN_FK = 'assist_loop_runs.id'


class AssistLoopRun(Base):
    __tablename__ = 'assist_loop_runs'
    __table_args__ = (
        ForeignKeyConstraint(
            ['preset_config_id', 'preset_revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_assist_loop_runs_preset',
        ),
        check('strategy', in_list('strategy', STRATEGIES)),
        check('stop_reason', in_list('stop_reason', STOP_REASONS)),
        check('preset_pair', '(preset_config_id IS NULL) = (preset_revision IS NULL)'),
        Index('ix_assist_loop_runs_project', 'project_id', text('created_at DESC')),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    # No ON DELETE rule: a job that a run still references is never pruned.
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('jobs.id'), unique=True)
    preset_config_id: Mapped[uuid.UUID | None]
    preset_revision: Mapped[int | None] = mapped_column(Integer)
    request: Mapped[Any] = mapped_column(JSONB)
    strategy: Mapped[str]
    seed: Mapped[int] = mapped_column(BigInteger)
    pinned: Mapped[Any] = mapped_column(JSONB)
    stop_reason: Mapped[str | None]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    created_by: Mapped[str]
    created_by_user_id: Mapped[uuid.UUID | None]


class AssistLoopIteration(Base):
    __tablename__ = 'assist_loop_iterations'
    __table_args__ = (
        check('ordinal', 'ordinal >= 1'),
        check('status', in_list('status', ITERATION_STATUSES)),
        check('closed_iff_closed_at', "(status = 'closed') = (closed_at IS NOT NULL)"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(_RUN_FK, ondelete='CASCADE'), primary_key=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str]
    counts: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    started_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    closed_at: Mapped[dt.datetime | None]


class AssistLoopNode(Base):
    __tablename__ = 'assist_loop_nodes'
    __table_args__ = (
        ForeignKeyConstraint(
            ['run_id', 'iteration'],
            ['assist_loop_iterations.run_id', 'assist_loop_iterations.ordinal'],
            ondelete='CASCADE',
            name='fk_assist_loop_nodes_iteration',
        ),
        check('path', "path ~ '^r[0-9]+(\\.[0-9]+)*$'"),
        check('level', 'level >= 0'),
        check('member_count', 'member_count >= 0'),
        check('eligible_count', 'eligible_count >= 0'),
        check('step', in_list('step', NODE_STEPS)),
        check('decision', in_list('decision', NODE_DECISIONS)),
        check('done_iff_decided', "(step = 'done') = (decision IS NOT NULL)"),
        Index('ix_assist_loop_nodes_open', 'run_id', postgresql_where=text("step <> 'done'")),
        Index('ix_assist_loop_nodes_decision', 'run_id', 'decision', 'level'),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(_RUN_FK, ondelete='CASCADE'), primary_key=True
    )
    path: Mapped[str] = mapped_column(primary_key=True)
    iteration: Mapped[int] = mapped_column(Integer)
    parent_path: Mapped[str | None]
    level: Mapped[int] = mapped_column(SmallInteger)
    member_count: Mapped[int] = mapped_column(Integer)
    eligible_count: Mapped[int] = mapped_column(Integer)
    membership_digest: Mapped[str]
    snapshot_storage_key: Mapped[str]
    step: Mapped[str]
    decision: Mapped[str | None]
    class_name: Mapped[str | None]
    answered: Mapped[int | None] = mapped_column(Integer)
    agree: Mapped[int | None] = mapped_column(Integer)
    bound: Mapped[float | None] = mapped_column(Float(53))
    distribution: Mapped[Any | None] = mapped_column(JSONB)
    admitted_crops: Mapped[int | None] = mapped_column(Integer)
    admitted_requests: Mapped[int | None] = mapped_column(Integer)
    label_op_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('label_operations.id', ondelete='SET NULL')
    )
    items_propagated: Mapped[int] = mapped_column(Integer, server_default='0')
    skipped: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class AssistLoopSelection(Base):
    __tablename__ = 'assist_loop_selections'
    __table_args__ = (check('purpose', in_list('purpose', SELECTION_PURPOSES)),)

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(_RUN_FK, ondelete='CASCADE'), primary_key=True
    )
    item_id: Mapped[str] = mapped_column(primary_key=True)
    purpose: Mapped[str] = mapped_column(primary_key=True)
    node_path: Mapped[str | None]
    iteration: Mapped[int] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float(53))
    draw_rank: Mapped[int] = mapped_column(Integer)
    selected_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class AssistLoopProposal(Base):
    __tablename__ = 'assist_loop_proposals'
    __table_args__ = (
        check('role', in_list('role', PROPOSAL_ROLES)),
        check('proposer_kind', in_list('proposer_kind', PROPOSER_KINDS)),
        check('outcome', in_list('outcome', PROPOSAL_OUTCOMES)),
        check('item_write', in_list('item_write', ITEM_WRITES)),
        check(
            'answered_has_class',
            "(outcome = 'answered') = (class_name IS NOT NULL OR new_class_name IS NOT NULL)",
        ),
        Index('ix_assist_loop_proposals_request', 'run_id', 'request_id'),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(_RUN_FK, ondelete='CASCADE'), primary_key=True
    )
    item_id: Mapped[str] = mapped_column(primary_key=True)
    role: Mapped[str] = mapped_column(primary_key=True)
    proposer_kind: Mapped[str]
    proposer_ref: Mapped[str]
    request_id: Mapped[uuid.UUID]
    outcome: Mapped[str]
    class_name: Mapped[str | None]
    new_class_name: Mapped[str | None]
    attributes: Mapped[Any | None] = mapped_column(JSONB)
    item_write: Mapped[str | None]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class AssistLoopReview(Base):
    __tablename__ = 'assist_loop_reviews'
    __table_args__ = (
        UniqueConstraint('run_id', 'item_id', 'queue', name='uq_assist_loop_reviews_item_queue'),
        check('queue', in_list('queue', REVIEW_QUEUES)),
        check('machine_source', in_list('machine_source', MACHINE_SOURCES)),
        check('decision', in_list('decision', REVIEW_DECISIONS)),
        check('active_ms', 'active_ms BETWEEN 0 AND 600000'),
        check('batch_size', 'batch_size BETWEEN 1 AND 200'),
        check('blind_audit_hides_class', "queue <> 'blind_audit' OR shown_class_name IS NULL"),
        check(
            'human_class_iff_accepted',
            "(decision IN ('accept', 'correct')) = (human_class_name IS NOT NULL)",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(_RUN_FK, ondelete='CASCADE'))
    item_id: Mapped[str]
    queue: Mapped[str]
    machine_source: Mapped[str]
    machine_class_name: Mapped[str | None]
    shown_class_name: Mapped[str | None]
    decision: Mapped[str]
    human_class_name: Mapped[str | None]
    served_at: Mapped[dt.datetime]
    decided_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    active_ms: Mapped[int | None] = mapped_column(Integer)
    batch_size: Mapped[int] = mapped_column(SmallInteger, server_default='1')
    label_op_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('label_operations.id', ondelete='SET NULL')
    )
    reviewer: Mapped[str]
    reviewer_user_id: Mapped[uuid.UUID | None]


class AssistLoopBudget(Base):
    __tablename__ = 'assist_loop_budgets'
    __table_args__ = (
        PrimaryKeyConstraint('run_id', 'pool', 'unit', name='pk_assist_loop_budgets'),
        check('pool', in_list('pool', BUDGET_POOLS)),
        check('unit', in_list('unit', BUDGET_UNITS)),
        check('limit_value', 'limit_value IS NULL OR limit_value >= 0'),
        check('reserved', 'reserved >= 0'),
        check('spent', 'spent >= 0'),
        check(
            'no_over_reserve',
            "unit = 'prompt_tokens' OR limit_value IS NULL OR spent + reserved <= limit_value",
        ),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(_RUN_FK, ondelete='CASCADE'))
    pool: Mapped[str]
    unit: Mapped[str]
    limit_value: Mapped[int | None] = mapped_column(BigInteger)
    reserved: Mapped[int] = mapped_column(BigInteger, server_default='0')
    spent: Mapped[int] = mapped_column(BigInteger, server_default='0')
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class AssistLoopBudgetLedger(Base):
    __tablename__ = 'assist_loop_budget_ledger'
    __table_args__ = (
        ForeignKeyConstraint(
            ['run_id', 'pool', 'unit'],
            ['assist_loop_budgets.run_id', 'assist_loop_budgets.pool', 'assist_loop_budgets.unit'],
            ondelete='CASCADE',
            name='fk_assist_loop_ledger_budget',
        ),
        check('entry', in_list('entry', LEDGER_ENTRIES)),
        Index('ix_assist_loop_ledger_request', 'run_id', 'request_id'),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(_RUN_FK, ondelete='CASCADE'))
    request_id: Mapped[uuid.UUID]
    pool: Mapped[str]
    unit: Mapped[str]
    entry: Mapped[str]
    amount: Mapped[int] = mapped_column(BigInteger)
    fencing_token: Mapped[int] = mapped_column(BigInteger)
    at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class VlmUsageHourly(Base):
    __tablename__ = 'vlm_usage_hourly'
    __table_args__ = (
        check('source', in_list('source', VLM_USAGE_SOURCES)),
        check('hour', "hour = date_trunc('hour', hour)"),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True
    )
    endpoint_config_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('config_documents.id', ondelete='CASCADE'), primary_key=True
    )
    source: Mapped[str] = mapped_column(primary_key=True)
    hour: Mapped[dt.datetime] = mapped_column(primary_key=True)
    requests: Mapped[int] = mapped_column(BigInteger, server_default='0')
    crops: Mapped[int] = mapped_column(BigInteger, server_default='0')
    prompt_tokens: Mapped[int] = mapped_column(BigInteger, server_default='0')
    completion_tokens: Mapped[int] = mapped_column(BigInteger, server_default='0')
    usage_missing: Mapped[int] = mapped_column(BigInteger, server_default='0')
