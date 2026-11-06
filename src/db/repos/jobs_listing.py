"""Read-only job listings for the job routes: filtered, newest first, keyset pages.

The cursor is ``(created_at, id)`` of the last row of the previous page, so a page is stable while
new jobs are inserted at the head (no offset to shift). Functions take the caller's session inside
its scoped transaction: a project transaction sees only that project's rows (row-level
security); a deployment transaction filters to the rows that belong to no project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, tuple_

from src.db.models.jobs import Job
from src.db.models.projects import Project
from src.db.repos.jobs import _ROW_COLUMNS, JobRow, _to_row


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Sequence

    from sqlalchemy import Select
    from sqlalchemy.engine import RowMapping
    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class JobFilter:
    """``project_id`` picks one project's jobs; ``None`` picks the jobs that belong to none."""

    project_id: uuid.UUID | None
    kinds: Sequence[str] = field(default_factory=tuple)
    statuses: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ListedJob:
    row: JobRow
    project_slug: str | None


def _select(flt: JobFilter) -> Select[Any]:
    query = select(*_ROW_COLUMNS, Project.slug.label('project_slug')).select_from(
        Job.__table__.outerjoin(Project.__table__, Project.id == Job.project_id)
    )
    query = query.where(Job.project_id.is_not_distinct_from(flt.project_id))
    if flt.kinds:
        query = query.where(Job.kind.in_(list(flt.kinds)))
    if flt.statuses:
        query = query.where(Job.status.in_(list(flt.statuses)))
    return query


def _listed(mapping: RowMapping) -> ListedJob:
    return ListedJob(_to_row(mapping), mapping['project_slug'])


def list_page(
    session: Session,
    flt: JobFilter,
    *,
    limit: int,
    after: tuple[dt.datetime, uuid.UUID] | None = None,
) -> list[ListedJob]:
    """Up to ``limit`` jobs after the cursor; ask for one more than a page to know there is a next."""
    query = _select(flt)
    if after is not None:
        query = query.where(tuple_(Job.created_at, Job.id) < tuple_(after[0], after[1]))
    query = query.order_by(Job.created_at.desc(), Job.id.desc()).limit(limit)
    return [_listed(m) for m in session.execute(query).mappings()]


def get_one(session: Session, flt: JobFilter, job_id: uuid.UUID) -> ListedJob | None:
    found = session.execute(_select(flt).where(Job.id == job_id)).mappings().first()
    return None if found is None else _listed(found)
