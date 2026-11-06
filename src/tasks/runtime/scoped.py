"""The transaction scope of a job's own writes: its project, or the deployment for a NULL one.

A job's rows are read first under deployment scope (the project is not known until the row is
read); every later write of the wrapper runs in the job's project scope, so a bug cannot touch
another project's rows. A deployment-wide job (no project) has no narrower scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.db.session import deployment_tx, project_tx, sync_deployment_tx, sync_project_tx


if TYPE_CHECKING:
    import uuid
    from contextlib import AbstractAsyncContextManager, AbstractContextManager

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session


def job_transaction(
    session: Session, project_id: uuid.UUID | None
) -> AbstractContextManager[Session]:
    """Commits on a clean exit and rolls back on any error."""
    if project_id is None:
        return sync_deployment_tx(session, 'job runtime write of a deployment-wide job')
    return sync_project_tx(session, project_id)


def head_transaction(session: Session) -> AbstractContextManager[Session]:
    """Deployment scope, for the one read that finds a job's project and kind."""
    return sync_deployment_tx(session, 'job runtime reads the project and kind of a job by id')


def async_job_transaction(
    session: AsyncSession, project_id: uuid.UUID | None
) -> AbstractAsyncContextManager[AsyncSession]:
    """The async twin, for the handler's own writes on the per-task engine."""
    if project_id is None:
        return deployment_tx(session, 'job runtime write of a deployment-wide job')
    return project_tx(session, project_id)
