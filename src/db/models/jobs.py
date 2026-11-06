"""Jobs, attempts, explicit item lists, events and the worker and GPU registries."""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Interval,
    PrimaryKeyConstraint,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base
from src.db.types import EMPTY_OBJECT, NOW, UUIDV7, Xid8, check, in_list
from src.domain import job_states


PAUSE_REASONS = ('user', 'audit', 'budget', 'project_paused', 'vlm_unavailable')
ATTEMPT_OUTCOMES = (
    'completed',
    'completed_with_errors',
    'failed',
    'cancelled',
    'paused',
    'interrupted',
    'lost',
    'time_limit',
    'yielded',
)
WORKER_KINDS = ('detection', 'vlm')
EVENT_PAYLOAD_MAX_BYTES = 7000

_ACTIVE_WHERE = in_list('status', tuple(sorted(job_states.ACTIVE)))
_FINISHED = in_list('status', tuple(sorted(job_states.TERMINAL)))


class Job(Base):
    __tablename__ = 'jobs'
    __table_args__ = (
        check('status', in_list('status', job_states.STATUSES)),
        check('max_attempts', 'max_attempts BETWEEN 1 AND 10'),
        check('pause_reason', in_list('pause_reason', PAUSE_REASONS)),
        check('idempotency_key', 'length(idempotency_key) BETWEEN 8 AND 200'),
        check('finished_iff_terminal', f'({_FINISHED}) = (finished_at IS NOT NULL)'),
        check(
            'live_has_lease',
            "status NOT IN ('running', 'cancelling') "
            'OR (owner IS NOT NULL AND lease_expires_at IS NOT NULL)',
        ),
        check('paused_has_reason', "status <> 'paused' OR pause_reason IS NOT NULL"),
        Index(
            'uq_jobs_one_active',
            'project_id',
            'exclusive_group',
            'scope',
            unique=True,
            postgresql_nulls_not_distinct=True,
            postgresql_where=text(f'exclusive_group IS NOT NULL AND {_ACTIVE_WHERE}'),
        ),
        Index(
            'uq_jobs_idempotency',
            'project_id',
            'kind',
            'idempotency_key',
            unique=True,
            postgresql_nulls_not_distinct=True,
            postgresql_where=text('idempotency_key IS NOT NULL'),
        ),
        Index('ix_jobs_project_recent', 'project_id', 'kind', text('created_at DESC')),
        Index(
            'ix_jobs_live',
            'status',
            postgresql_where=text("status IN ('running', 'cancelling')"),
        ),
        Index('ix_jobs_queued', 'queued_at', postgresql_where=text("status = 'queued'")),
        Index(
            'ix_jobs_interrupted',
            'updated_at',
            postgresql_where=text("status = 'interrupted' AND auto_resume"),
        ),
        {
            'postgresql_with': {
                'fillfactor': 70,
                'autovacuum_vacuum_scale_factor': 0.02,
                'autovacuum_vacuum_threshold': 50,
                'autovacuum_vacuum_cost_delay': 2,
            }
        },
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE')
    )
    kind: Mapped[str]
    scope: Mapped[str] = mapped_column(server_default='')
    exclusive_group: Mapped[str | None]
    queue: Mapped[str]
    status: Mapped[str]
    params: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    checkpoint: Mapped[Any | None] = mapped_column(JSONB)
    progress: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    progress_updated_at: Mapped[dt.datetime | None]
    result: Mapped[Any | None] = mapped_column(JSONB)
    error_code: Mapped[str | None]
    error_message: Mapped[str | None]
    attempt: Mapped[int] = mapped_column(Integer, server_default='0')
    retry_count: Mapped[int] = mapped_column(Integer, server_default='0')
    max_attempts: Mapped[int] = mapped_column(Integer, server_default='3')
    auto_resume: Mapped[bool] = mapped_column(server_default=text('false'))
    fencing_token: Mapped[int] = mapped_column(BigInteger, server_default='0')
    owner: Mapped[str | None]
    heartbeat_at: Mapped[dt.datetime | None]
    lease_expires_at: Mapped[dt.datetime | None]
    max_queue_wait: Mapped[dt.timedelta] = mapped_column(Interval)
    dispatch_count: Mapped[int] = mapped_column(Integer, server_default='0')
    last_dispatched_at: Mapped[dt.datetime | None]
    cancel_requested_at: Mapped[dt.datetime | None]
    cancel_requested_by: Mapped[str | None]
    pause_requested_at: Mapped[dt.datetime | None]
    pause_reason: Mapped[str | None]
    idempotency_key: Mapped[str | None]
    parent_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('jobs.id', ondelete='SET NULL')
    )
    requested_by: Mapped[str]
    requested_by_user_id: Mapped[uuid.UUID | None]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    queued_at: Mapped[dt.datetime | None]
    started_at: Mapped[dt.datetime | None]
    finished_at: Mapped[dt.datetime | None]
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)

    attempts: Mapped[list['JobAttempt']] = relationship(lazy='raise', back_populates='job')
    items: Mapped[list['JobItem']] = relationship(lazy='raise', back_populates='job')


class JobAttempt(Base):
    __tablename__ = 'job_attempts'
    __table_args__ = (check('outcome', in_list('outcome', ATTEMPT_OUTCOMES)),)

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('jobs.id', ondelete='CASCADE'), primary_key=True
    )
    attempt: Mapped[int] = mapped_column(Integer, primary_key=True)
    fencing_token: Mapped[int] = mapped_column(BigInteger)
    owner: Mapped[str]
    started_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    ended_at: Mapped[dt.datetime | None]
    outcome: Mapped[str | None]
    error_code: Mapped[str | None]
    error_message: Mapped[str | None]

    job: Mapped['Job'] = relationship(lazy='raise', back_populates='attempts')


class JobItem(Base):
    __tablename__ = 'job_items'
    __table_args__ = (UniqueConstraint('job_id', 'item_id', name='uq_job_items_job_id_item_id'),)

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('jobs.id', ondelete='CASCADE'), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[str]

    job: Mapped['Job'] = relationship(lazy='raise', back_populates='items')


class Event(Base):
    __tablename__ = 'events'
    __table_args__ = (
        check('payload_size', f'octet_length(payload::text) <= {EVENT_PAYLOAD_MAX_BYTES}'),
        Index('ix_events_cursor', 'project_id', 'xid', 'id'),
        Index(
            'ix_events_cursor_global',
            'xid',
            'id',
            postgresql_where=text('project_id IS NULL'),
        ),
        Index('ix_events_created', 'created_at'),
        {
            'postgresql_with': {
                'autovacuum_vacuum_scale_factor': 0.05,
                'autovacuum_vacuum_insert_scale_factor': 0.05,
            }
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    xid: Mapped[int] = mapped_column(Xid8(), server_default=text('pg_current_xact_id()'))
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE')
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'))
    type: Mapped[str]
    payload: Mapped[Any] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class StreamWorkerClaim(Base):
    __tablename__ = 'stream_worker_claims'
    __table_args__ = (
        PrimaryKeyConstraint('worker_kind', 'project_id', name='pk_stream_worker_claims'),
        check('worker_kind', in_list('worker_kind', WORKER_KINDS)),
    )

    worker_kind: Mapped[str]
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    worker_id: Mapped[str]
    fencing_token: Mapped[int] = mapped_column(BigInteger, server_default='1')
    lease_expires_at: Mapped[dt.datetime]


class WorkerCapability(Base):
    __tablename__ = 'worker_capabilities'

    worker_id: Mapped[str] = mapped_column(primary_key=True)
    queue: Mapped[str]
    host: Mapped[str]
    capabilities: Mapped[Any] = mapped_column(JSONB)
    seen_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class GpuDevice(Base):
    __tablename__ = 'gpu_devices'

    gpu_uuid: Mapped[str] = mapped_column(primary_key=True)
    host: Mapped[str]
    nvml_index: Mapped[int] = mapped_column(Integer)
    cuda_index: Mapped[int | None] = mapped_column(Integer)
    name: Mapped[str]
    compute_capability: Mapped[str]
    memory_total_mib: Mapped[int] = mapped_column(Integer)
    numa_node: Mapped[int | None] = mapped_column(Integer)
    pcie_gen: Mapped[int | None] = mapped_column(Integer)
    pcie_width: Mapped[int | None] = mapped_column(Integer)
    hw_jpeg_decoder: Mapped[bool]
    driver_version: Mapped[str]
    cuda_version: Mapped[str]
    seen_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class GpuReservation(Base):
    __tablename__ = 'gpu_reservations'

    gpu_uuid: Mapped[str] = mapped_column(
        ForeignKey('gpu_devices.gpu_uuid', ondelete='RESTRICT'), primary_key=True
    )
    holder_job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'))
    reserved_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
