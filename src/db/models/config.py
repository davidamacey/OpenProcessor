"""Config store, project settings, keymap, internal markers and projection state."""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    PrimaryKeyConstraint,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base
from src.db.types import EMPTY_OBJECT, NOW, UUIDV7, check, in_list


CONFIG_KINDS = (
    'prompt_pack',
    'region_profile',
    'open_vocab_set',
    'ingest_profile',
    'vlm_endpoint',
    'serving_profile',
    'cluster_profile',
    'assist_loop_preset',
    'near_dup_profile',
)
# Documents of these kinds are deployment-wide; every other kind belongs to a project.
DEPLOYMENT_CONFIG_KINDS = ('vlm_endpoint', 'serving_profile')
# (axis, kind of the document it activates)
ACTIVATION_AXES: tuple[tuple[str, str], ...] = (
    ('prompt_pack', 'prompt_pack'),
    ('detection_profile', 'region_profile'),
    ('vlm', 'vlm_endpoint'),
    ('open_vocab', 'open_vocab_set'),
    ('ingest_profile', 'ingest_profile'),
    ('clustering_item', 'cluster_profile'),
    ('clustering_region_box', 'cluster_profile'),
    ('assist_loop_preset', 'assist_loop_preset'),
    ('near_dup', 'near_dup_profile'),
)
ACTIVATION_REASONS = ('activate', 'rollback', 'clone', 'upgrade')
LOCATION_POLICIES = ('drop', 'store_coarse', 'store')
PROJECT_STATE_KEYS = (
    'cluster_refresh',
    'region_drain',
    'region_dependency_health',
    'vlm_external_ack',
    'open_vocab_gate',
)


def _axis_kind_pairs() -> str:
    pairs = ', '.join(f"('{axis}', '{kind}')" for axis, kind in ACTIVATION_AXES)
    return f'(axis, kind) IN ({pairs})'


class ConfigDocument(Base):
    __tablename__ = 'config_documents'
    __table_args__ = (
        UniqueConstraint(
            'project_id',
            'kind',
            'name',
            name='uq_config_documents_project_id_kind_name',
            postgresql_nulls_not_distinct=True,
        ),
        UniqueConstraint('id', 'kind', name='uq_config_documents_id_kind'),
        check('kind', in_list('kind', CONFIG_KINDS)),
        check('name', "name ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'"),
        check('current_revision', 'current_revision >= 1'),
        check(
            'scope',
            f'({in_list("kind", DEPLOYMENT_CONFIG_KINDS)}) = (project_id IS NULL)',
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE')
    )
    kind: Mapped[str]
    name: Mapped[str]
    current_revision: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(server_default='')
    cloned_from_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('config_documents.id', ondelete='SET NULL')
    )
    cloned_from_revision: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    updated_by: Mapped[str]
    updated_by_user_id: Mapped[uuid.UUID | None]

    revisions: Mapped[list['ConfigRevision']] = relationship(
        lazy='raise', back_populates='document'
    )


class ConfigRevision(Base):
    __tablename__ = 'config_revisions'
    __table_args__ = (check('revision', 'revision >= 1'),)

    config_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('config_documents.id', ondelete='CASCADE'), primary_key=True
    )
    revision: Mapped[int] = mapped_column(Integer, primary_key=True)
    body: Mapped[Any] = mapped_column(JSONB)
    body_sha256: Mapped[str]
    description: Mapped[str] = mapped_column(server_default='')
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    created_by: Mapped[str]

    document: Mapped['ConfigDocument'] = relationship(lazy='raise', back_populates='revisions')


class ConfigActivation(Base):
    __tablename__ = 'config_activations'
    __table_args__ = (
        PrimaryKeyConstraint('project_id', 'axis', name='pk_config_activations'),
        ForeignKeyConstraint(
            ['config_id', 'kind'],
            ['config_documents.id', 'config_documents.kind'],
            name='fk_config_activations_config_id_kind_config_documents',
        ),
        ForeignKeyConstraint(
            ['config_id', 'revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_config_activations_config_id_revision_config_revisions',
        ),
        check('axis', in_list('axis', tuple(a for a, _ in ACTIVATION_AXES))),
        check('axis_kind', _axis_kind_pairs()),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    axis: Mapped[str]
    kind: Mapped[str]
    config_id: Mapped[uuid.UUID]
    revision: Mapped[int] = mapped_column(Integer)
    activated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    activated_by: Mapped[str]


class ConfigActivationEvent(Base):
    __tablename__ = 'config_activation_events'
    __table_args__ = (
        check('reason', in_list('reason', ACTIVATION_REASONS)),
        # An axis turned off is an event with no target: both NULL, never one of the two.
        check('to_pair', '(to_config_id IS NULL) = (to_revision IS NULL)'),
        Index('ix_config_activation_events_axis', 'project_id', 'axis', text('id DESC')),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    axis: Mapped[str]
    from_config_id: Mapped[uuid.UUID | None]
    from_revision: Mapped[int | None] = mapped_column(Integer)
    to_config_id: Mapped[uuid.UUID | None]
    to_revision: Mapped[int | None] = mapped_column(Integer)
    reason: Mapped[str]
    at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    by: Mapped[str]


class VlmEndpointProbe(Base):
    __tablename__ = 'vlm_endpoint_probes'
    __table_args__ = (
        ForeignKeyConstraint(
            ['config_id', 'revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            ondelete='CASCADE',
            name='fk_vlm_endpoint_probes_config_id_revision_config_revisions',
        ),
    )

    config_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('config_documents.id', ondelete='CASCADE'), primary_key=True
    )
    revision: Mapped[int] = mapped_column(Integer, primary_key=True)
    probed_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    ok: Mapped[bool]
    result: Mapped[Any] = mapped_column(JSONB)


class ConfigRuntimeReport(Base):
    __tablename__ = 'config_runtime_reports'

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True
    )
    process: Mapped[str] = mapped_column(primary_key=True)
    host: Mapped[str] = mapped_column(primary_key=True)
    config_epoch: Mapped[int] = mapped_column(BigInteger)
    applied_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class ProjectSettings(Base):
    __tablename__ = 'project_settings'
    __table_args__ = (check('location_policy', in_list('location_policy', LOCATION_POLICIES)),)

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True
    )
    defaults: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    ingest_policy: Mapped[Any | None] = mapped_column(JSONB)
    vlm_policy: Mapped[Any | None] = mapped_column(JSONB)
    vlm_scope_reps: Mapped[Any | None] = mapped_column(JSONB)
    search: Mapped[Any | None] = mapped_column(JSONB)
    location_policy: Mapped[str] = mapped_column(server_default='store')
    revision: Mapped[int] = mapped_column(BigInteger, server_default='1')
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    updated_by: Mapped[str]


class ProjectKeymap(Base):
    __tablename__ = 'project_keymaps'

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True
    )
    overrides: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    revision: Mapped[int] = mapped_column(BigInteger, server_default='1')
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    updated_by: Mapped[str]


class ProjectState(Base):
    __tablename__ = 'project_state'
    __table_args__ = (check('key', in_list('key', PROJECT_STATE_KEYS)),)

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True
    )
    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB)
    revision: Mapped[int] = mapped_column(BigInteger, server_default='1')
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)


class ProjectionState(Base):
    __tablename__ = 'projection_states'

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True
    )
    projection_version: Mapped[int] = mapped_column(Integer)
    scope: Mapped[str]
    cluster_uid: Mapped[uuid.UUID | None]
    n_points: Mapped[int] = mapped_column(Integer)
    n_components: Mapped[int] = mapped_column(Integer)
    metric: Mapped[str]
    fitted_at: Mapped[dt.datetime]
    reducer_storage_key: Mapped[str | None]
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
