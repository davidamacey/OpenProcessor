"""Projects and deployment-wide tables."""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Identity, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base
from src.db.types import NOW, UUIDV7, check, in_list


PROJECT_STATUSES = ('building', 'active', 'archived', 'deleting', 'failed')
PRE_DELETE_STATUSES = ('active', 'archived', 'failed')
RESERVED_SLUGS = (
    'combine',
    'new',
    'all',
    'none',
    'projects',
    'global',
    'settings',
    'vlm',
    'health',
)
PROJECT_INDEX_ROLES = ('images', 'items')
DEPLOYMENT_SETTING_KEYS = (
    'local_vlm_desired',
    'vlm_paused_for_training',
    'active_serving_profile',
    'upgrade_state',
)


class ProjectSlug(Base):
    __tablename__ = 'project_slugs'
    __table_args__ = (
        check(
            'slug',
            "slug ~ '^[a-z][a-z0-9-]{1,31}$' AND slug !~ '--' AND slug !~ '-$'",
        ),
    )

    slug: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    retired_at: Mapped[dt.datetime | None]


class Project(Base):
    __tablename__ = 'projects'
    __table_args__ = (
        check('status', in_list('status', PROJECT_STATUSES)),
        check('pre_delete_status', in_list('pre_delete_status', PRE_DELETE_STATUSES)),
        check('slug_reserved', f'NOT ({in_list("slug", RESERVED_SLUGS)})'),
        check(
            'deleting_has_prior_status', "(status = 'deleting') = (pre_delete_status IS NOT NULL)"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    lock_key: Mapped[int] = mapped_column(Integer, Identity(always=True), unique=True)
    slug: Mapped[str] = mapped_column(ForeignKey('project_slugs.slug'), unique=True)
    display_name: Mapped[str]
    description: Mapped[str] = mapped_column(server_default='')
    status: Mapped[str]
    pre_delete_status: Mapped[str | None]
    revision: Mapped[int] = mapped_column(BigInteger, server_default='1')
    config_epoch: Mapped[int] = mapped_column(BigInteger, server_default='0')
    paused_at: Mapped[dt.datetime | None]
    paused_by: Mapped[str | None]
    origin: Mapped[Any | None] = mapped_column(JSONB)
    index_prefix: Mapped[str]
    storage_prefix: Mapped[str]
    model_prefix: Mapped[str]
    mlflow_experiment: Mapped[str]
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    created_by: Mapped[str]
    created_by_user_id: Mapped[uuid.UUID | None]

    indexes: Mapped[list['ProjectIndex']] = relationship(lazy='raise', back_populates='project')


class ProjectIndex(Base):
    __tablename__ = 'project_indexes'
    __table_args__ = (check('role', in_list('role', PROJECT_INDEX_ROLES)),)

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True
    )
    role: Mapped[str] = mapped_column(primary_key=True)
    index_name: Mapped[str] = mapped_column(unique=True)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)

    project: Mapped['Project'] = relationship(lazy='raise', back_populates='indexes')


class DeploymentCounter(Base):
    __tablename__ = 'deployment_counters'

    name: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[int] = mapped_column(BigInteger, server_default='0')


class DeploymentSetting(Base):
    __tablename__ = 'deployment_settings'
    __table_args__ = (check('key', in_list('key', DEPLOYMENT_SETTING_KEYS)),)

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB)
    revision: Mapped[int] = mapped_column(BigInteger, server_default='1')
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    updated_by: Mapped[str]


__all__ = [
    'DEPLOYMENT_SETTING_KEYS',
    'PROJECT_INDEX_ROLES',
    'PROJECT_STATUSES',
    'DeploymentCounter',
    'DeploymentSetting',
    'Project',
    'ProjectIndex',
    'ProjectSlug',
]
