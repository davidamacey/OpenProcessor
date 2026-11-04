"""The lease of one combine job.

Combine jobs are global (their state lives outside any project), but a lease is a document
in a project's index, so the ``combine`` lease (scope: the job id) is taken in the
**target** project's index. Every lease operation binds the target for its own duration only;
the rest of a combine runs unbound, as it always has, because it reads the source projects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.domain.job_lease import LeaseKind
from src.services.curation.lease_heartbeat import acquire_heartbeat


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.config.projects import ProjectRecord
    from src.services.curation.lease_heartbeat import LeaseHeartbeat


async def take(
    client: Any,
    target: ProjectRecord,
    job_id: str,
    *,
    busy: Callable[[str | None], Exception],
) -> LeaseHeartbeat:
    """The job's lease, not yet renewing (see :func:`begin`); ``busy(holder)`` is raised
    while another worker holds it."""
    with bind_project(target):
        return await acquire_heartbeat(
            client, LeaseKind.COMBINE, job_id, job_id, job=None, busy=busy
        )


def begin(beat: LeaseHeartbeat, target: ProjectRecord) -> None:
    """Start renewing; the heartbeat thread keeps the target bound."""
    with bind_project(target):
        beat.start()


async def give_back(client: Any, target: ProjectRecord, beat: LeaseHeartbeat) -> None:
    with bind_project(target):
        await beat.release(client)
