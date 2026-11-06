"""Class registry and the outbox for the search-index class-name projection."""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import BigInteger, ForeignKey, ForeignKeyConstraint, Index, Integer, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base
from src.db.types import NOW, UUIDV7, check, in_list


REGISTRY_MIGRATED_FROM = ('class_registry_file', 'opensearch_registry_doc')
PROJECTION_TASK_KINDS = ('class_rename', 'class_merge', 'cluster_partition', 'near_dup_run')


class ClassRegistry(Base):
    __tablename__ = 'class_registries'
    __table_args__ = (
        check('next_class_id', 'next_class_id >= 0'),
        check('migrated_from', in_list('migrated_from', REGISTRY_MIGRATED_FROM)),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True
    )
    next_class_id: Mapped[int] = mapped_column(Integer, server_default='0')
    revision: Mapped[int] = mapped_column(BigInteger, server_default='0')
    migrated_from: Mapped[str | None]
    migrated_at: Mapped[dt.datetime | None]
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)

    classes: Mapped[list['ProjectClass']] = relationship(
        lazy='raise', back_populates='class_registry'
    )


class ProjectClass(Base):
    __tablename__ = 'project_classes'
    __table_args__ = (
        ForeignKeyConstraint(
            ['project_id', 'merged_into'],
            ['project_classes.project_id', 'project_classes.class_id'],
            name='fk_project_classes_project_id_merged_into_project_classes',
        ),
        check('class_id', 'class_id >= 0'),
        check('name', 'length(name) BETWEEN 1 AND 200 AND name = btrim(name)'),
        check('hotkey_letter', "hotkey_letter ~ '^[a-z0-9]$'"),
        check('name_key', "name_key ~ '^[a-z0-9_]+$'"),
        check('merged_into_other', 'merged_into IS NULL OR merged_into <> class_id'),
        check('merged_into_deprecated', 'merged_into IS NULL OR deprecated'),
        Index(
            'uq_project_classes_hotkey',
            'project_id',
            'hotkey_letter',
            unique=True,
            postgresql_where=text('hotkey_letter IS NOT NULL'),
        ),
        Index(
            'uq_project_classes_active_name',
            'project_id',
            'name_key',
            unique=True,
            postgresql_where=text('NOT deprecated'),
        ),
        Index('ix_project_classes_name_key', 'project_id', 'name_key'),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('class_registries.project_id', ondelete='CASCADE'), primary_key=True
    )
    class_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str]
    class_group: Mapped[str | None]
    deprecated: Mapped[bool] = mapped_column(server_default=text('false'))
    merged_into: Mapped[int | None] = mapped_column(Integer)
    hotkey_letter: Mapped[str | None]
    notes: Mapped[str] = mapped_column(server_default='')
    added_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    updated_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    name_key: Mapped[str]

    class_registry: Mapped['ClassRegistry'] = relationship(lazy='raise', back_populates='classes')


class SearchProjectionTask(Base):
    __tablename__ = 'search_projection_tasks'
    __table_args__ = (
        check('kind', in_list('kind', PROJECTION_TASK_KINDS)),
        Index(
            'ix_search_projection_tasks_pending',
            'created_at',
            postgresql_where=text('done_at IS NULL'),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=UUIDV7)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'))
    kind: Mapped[str]
    payload: Mapped[Any] = mapped_column(JSONB)
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('jobs.id', ondelete='SET NULL'))
    created_at: Mapped[dt.datetime] = mapped_column(server_default=NOW)
    done_at: Mapped[dt.datetime | None]
