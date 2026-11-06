"""Clustering partitions, stable cluster identities, memberships, calibrations and activations.

A NULL ``project_id`` is a core row (faces) or, for calibrations, the deployment default. The
immutability and project-copy triggers are in revision ``0002_research_seams``.
"""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import REAL

from src.db.base import Base
from src.db.types import EMPTY_ARRAY, EMPTY_OBJECT, NOW, UUIDV7, check, in_list
from src.domain import clustering_vocab as vocab


def _project_fk() -> ForeignKey:
    return ForeignKey('projects.id', ondelete='CASCADE')


class ClusterCalibration(Base):
    __tablename__ = 'cluster_calibrations'
    __table_args__ = (
        check('target', in_list('target', vocab.TARGETS)),
        check('level', in_list('level', vocab.LEVELS)),
        check('kind', in_list('kind', vocab.CALIBRATION_KINDS)),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID | None] = mapped_column(_project_fk())
    target: Mapped[str]
    level: Mapped[str]
    embedding_field: Mapped[str]
    embedding_model: Mapped[str]
    preprocess_id: Mapped[str]
    kind: Mapped[str]
    fitted_on: Mapped[str]
    body: Mapped[Any] = mapped_column(JSONB)
    dev_metrics: Mapped[Any] = mapped_column(JSONB)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    created_by: Mapped[str]


class ClusterPartition(Base):
    __tablename__ = 'cluster_partitions'
    __table_args__ = (
        UniqueConstraint('id', 'project_id', name='uq_cluster_partitions_id_project_id'),
        ForeignKeyConstraint(
            ['profile_config_id', 'profile_revision'],
            ['config_revisions.config_id', 'config_revisions.revision'],
            name='fk_cluster_partitions_profile',
        ),
        check('target', in_list('target', vocab.TARGETS)),
        check('level', in_list('level', vocab.LEVELS)),
        check('method', in_list('method', vocab.METHODS)),
        check('backend', in_list('backend', vocab.BACKENDS)),
        check('status', in_list('status', vocab.PARTITION_STATUSES)),
        check('profile_pair', '(profile_config_id IS NULL) = (profile_revision IS NULL)'),
        Index(
            'ix_cluster_partitions_recent',
            'project_id',
            'target',
            'level',
            text('created_at DESC'),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID | None] = mapped_column(_project_fk())
    target: Mapped[str]
    level: Mapped[str]
    embedding_field: Mapped[str]
    embedding_model: Mapped[str]
    preprocess_id: Mapped[str]
    method: Mapped[str]
    method_version: Mapped[str]
    profile_config_id: Mapped[uuid.UUID | None]
    profile_revision: Mapped[int | None] = mapped_column(Integer)
    params: Mapped[Any] = mapped_column(JSONB)
    calibration_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('cluster_calibrations.id'))
    pool_filter: Mapped[Any] = mapped_column(JSONB)
    input_digest: Mapped[str]
    seed: Mapped[int] = mapped_column(BigInteger)
    backend: Mapped[str]
    backend_detail: Mapped[Any] = mapped_column(JSONB)
    parent_partition_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            'cluster_partitions.id',
            ondelete='SET NULL',
            name='fk_cluster_partitions_parent_partition',
        )
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
    status: Mapped[str]
    n_items: Mapped[int | None] = mapped_column(Integer)
    n_clusters: Mapped[int | None] = mapped_column(Integer)
    n_review: Mapped[int | None] = mapped_column(Integer)
    n_unassigned: Mapped[int | None] = mapped_column(Integer)
    metrics: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    diff: Mapped[Any | None] = mapped_column(JSONB)
    stage_durations: Mapped[Any] = mapped_column(JSONB, server_default=EMPTY_OBJECT)
    peak_rss_mb: Mapped[int | None] = mapped_column(Integer)
    peak_vram_mb: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    finished_at: Mapped[dt.datetime | None]
    created_by: Mapped[str]


class ClusterIdentity(Base):
    __tablename__ = 'cluster_identities'
    __table_args__ = (
        ForeignKeyConstraint(
            ['project_id', 'promoted_class_id'],
            ['project_classes.project_id', 'project_classes.class_id'],
            name='fk_cluster_identities_promoted_class',
        ),
        check('target', in_list('target', vocab.TARGETS)),
        check('level', in_list('level', vocab.IDENTITY_LEVELS)),
        check('name_source', in_list('name_source', vocab.NAME_SOURCES)),
        check('anchored_pair', '(anchored_at IS NULL) = (anchored_by IS NULL)'),
    )

    cluster_uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID | None] = mapped_column(_project_fk())
    target: Mapped[str]
    level: Mapped[str]
    name: Mapped[str | None]
    name_source: Mapped[str | None]
    anchored_at: Mapped[dt.datetime | None]
    anchored_by: Mapped[str | None]
    promoted_class_id: Mapped[int | None] = mapped_column(Integer)
    first_partition_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            'cluster_partitions.id',
            ondelete='SET NULL',
            name='fk_cluster_identities_first_partition',
        )
    )
    last_partition_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            'cluster_partitions.id',
            ondelete='SET NULL',
            name='fk_cluster_identities_last_partition',
        )
    )
    revision: Mapped[int] = mapped_column(BigInteger, server_default='1')


class PartitionCluster(Base):
    __tablename__ = 'partition_clusters'
    __table_args__ = (
        UniqueConstraint('partition_id', 'display_no', name='uq_partition_clusters_display_no'),
        ForeignKeyConstraint(
            ['partition_id', 'project_id'],
            ['cluster_partitions.id', 'cluster_partitions.project_id'],
            ondelete='CASCADE',
            name='fk_partition_clusters_partition_project',
        ),
        check('transition', in_list('transition', vocab.TRANSITIONS)),
        check('size', 'size > 0'),
    )

    partition_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('cluster_partitions.id', ondelete='CASCADE'), primary_key=True
    )
    cluster_uid: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('cluster_identities.cluster_uid', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID | None]
    display_no: Mapped[int] = mapped_column(Integer)
    parent_uid: Mapped[uuid.UUID | None]
    transition: Mapped[str]
    predecessor_uids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(Uuid()), server_default=EMPTY_ARRAY
    )
    size: Mapped[int] = mapped_column(Integer)
    dispersion: Mapped[float | None] = mapped_column(REAL)
    prototypes: Mapped[Any] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))


class ClusterMembership(Base):
    __tablename__ = 'cluster_memberships'
    __table_args__ = (
        ForeignKeyConstraint(
            ['partition_id', 'project_id'],
            ['cluster_partitions.id', 'cluster_partitions.project_id'],
            ondelete='CASCADE',
            name='fk_cluster_memberships_partition_project',
        ),
        check('state', in_list('state', vocab.MEMBERSHIP_STATES)),
        check('unassigned_iff_no_cluster', "(state = 'unassigned') = (cluster_uid IS NULL)"),
        Index('ix_cluster_memberships_cluster', 'partition_id', 'cluster_uid'),
    )

    partition_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('cluster_partitions.id', ondelete='CASCADE'), primary_key=True
    )
    project_id: Mapped[uuid.UUID | None]
    item_key: Mapped[str] = mapped_column(primary_key=True)
    cluster_uid: Mapped[uuid.UUID | None]
    state: Mapped[str]
    score: Mapped[float | None] = mapped_column(REAL)
    margin: Mapped[float | None] = mapped_column(REAL)


class ClusterActivation(Base):
    __tablename__ = 'cluster_activations'
    __table_args__ = (
        UniqueConstraint(
            'project_id',
            'target',
            'level',
            'embedding_field',
            name='uq_cluster_activations_served_key',
            postgresql_nulls_not_distinct=True,
        ),
        check('target', in_list('target', vocab.TARGETS)),
        check('level', in_list('level', vocab.LEVELS)),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID | None] = mapped_column(_project_fk())
    target: Mapped[str]
    level: Mapped[str]
    embedding_field: Mapped[str]
    serving_partition_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('cluster_partitions.id', name='fk_cluster_activations_serving_partition')
    )
    pending_partition_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('cluster_partitions.id', name='fk_cluster_activations_pending_partition')
    )
    revision: Mapped[int] = mapped_column(BigInteger, server_default='1')
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    updated_by: Mapped[str]


class ClusterActivationEvent(Base):
    __tablename__ = 'cluster_activation_events'
    __table_args__ = (check('reason', in_list('reason', vocab.ACTIVATION_REASONS)),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    project_id: Mapped[uuid.UUID | None] = mapped_column(_project_fk())
    target: Mapped[str]
    level: Mapped[str]
    embedding_field: Mapped[str]
    from_partition_id: Mapped[uuid.UUID | None]
    to_partition_id: Mapped[uuid.UUID]
    reason: Mapped[str]
    at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    by: Mapped[str]


class ClusterBackendValidation(Base):
    __tablename__ = 'cluster_backend_validations'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    method: Mapped[str]
    backend_versions: Mapped[Any] = mapped_column(JSONB)
    embedding_field: Mapped[str]
    dataset_pin: Mapped[str]
    ari_vs_cpu: Mapped[float] = mapped_column(REAL)
    purity_delta: Mapped[float] = mapped_column(REAL)
    graph_recall: Mapped[float | None] = mapped_column(REAL)
    passed: Mapped[bool] = mapped_column(Boolean)
    validated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    validated_by: Mapped[str]
