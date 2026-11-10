"""Reasons a job may not be resumed, and when the reaper may resume a system pause for it.

A project pause stops its jobs until the project resumes (``projects.paused_at``); a job paused
because the VLM was taken for training waits until ``deployment_settings('vlm_paused_for_training')``
holds no claim and no model swap (:mod:`src.domain.gpu_arbiter_state`) and a VLM endpoint answers its latest probe (02b section 4, rule h).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from src.db.repos import gpu_arbiter as arbiter_repo
from src.db.session import async_session_factory, deployment_tx
from src.domain import gpu_arbiter_state as state


if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from src.db.repos.jobs import JobRow

PROJECT_PAUSED = 'project_paused'
VLM_UNAVAILABLE = 'vlm_unavailable'


def project_is_paused(session: Session, project_id: object) -> bool:
    row = session.execute(
        text('SELECT paused_at IS NOT NULL FROM projects WHERE id = :id'), {'id': project_id}
    ).scalar_one_or_none()
    return bool(row)


def vlm_available(session: Session) -> bool:
    """The VLM is free (not taken for training) and at least one endpoint's latest probe is ok."""
    if state.is_paused(arbiter_repo.read_value(session)):
        return False
    latest_ok = session.execute(
        text(
            'SELECT bool_or(ok) FROM (SELECT DISTINCT ON (config_id) ok FROM vlm_endpoint_probes '
            'ORDER BY config_id, revision DESC) latest'
        )
    ).scalar_one_or_none()
    return bool(latest_ok)


async def vlm_ready() -> bool:
    """:func:`vlm_available` for a caller inside a project transaction.

    The endpoint registry is deployment-wide (its documents have no project), and row-level
    security hides those rows, and so every probe, from a project-scope session: read there, the
    answer is always false. This reads them in a deployment-scope session of its own.
    """
    async with (
        async_session_factory()() as session,
        deployment_tx(session, 'latest probes of the deployment-wide VLM endpoints'),
    ):
        return await session.run_sync(vlm_available)


def resume_blocker(session: Session, job: JobRow, *, vlm_ok: bool) -> str | None:
    """The reason a person's resume is refused right now, or ``None``.

    ``vlm_ok`` is :func:`vlm_ready`, read by the caller (it needs a session of its own).
    """
    if job.project_id is not None and project_is_paused(session, job.project_id):
        return PROJECT_PAUSED
    if job.pause_reason == VLM_UNAVAILABLE and not vlm_ok:
        return VLM_UNAVAILABLE
    return None
