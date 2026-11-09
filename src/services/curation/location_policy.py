"""Change a project's ``location_policy`` and start what the change requires.

A policy that keeps less than before, or the same again, starts the ``source_backfill`` job with
its ``location_purge`` step alone: ingest and every reader already follow the new policy once the
write is committed (``apply_location_policy``), the purge removes what was stored under the old
one. Loosening (``drop`` to ``store``) starts nothing and restores nothing: a coordinate that was
cleared is read again only by re-ingesting the file.

The write commits before the job is queued (``start_job`` owns its transaction). If a
``source_backfill`` job is already running the start is refused with 409 ``job_busy`` *after* the
policy was saved; the message names the call that starts the purge once that job ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from src.config.project_context import current_project_id
from src.core.exceptions import RequestRejectedError
from src.db.repos.config import RevisionConflictError
from src.schemas.curation.location_policy import LocationPolicyResponse
from src.services.curation.location_policy_store import read_location_policy, write_location_policy
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.curation.location_policy import LocationPolicyName

#: How much each policy keeps (lower keeps less).
KEEPS: Final = {'drop': 0, 'store_coarse': 1, 'store': 2}
REQUESTER: Final = 'curation-api'


async def get_policy() -> LocationPolicyResponse:
    policy, revision = await read_location_policy(fresh=True)
    return LocationPolicyResponse(
        location_policy=cast('LocationPolicyName', policy), revision=revision
    )


async def change_policy(
    session: AsyncSession, policy: str, *, expected_revision: int
) -> LocationPolicyResponse:
    """Store ``policy`` (``RevisionConflictError`` on a stale revision) and start the purge."""
    previous, _ = await read_location_policy(fresh=True)
    stored, revision = await write_location_policy(policy, expected_revision=expected_revision)
    job_id: str | None = None
    if KEEPS[stored] < KEEPS['store'] and KEEPS[stored] <= KEEPS[previous]:
        try:
            started = await start_job(
                session,
                'source_backfill',
                current_project_id(),
                params={'steps': ['location_purge']},
                requested_by=REQUESTER,
            )
        except RequestRejectedError as exc:
            if exc.status_code != 409:
                raise
            msg = (
                f'location_policy was saved as {stored!r} (revision {revision}); the purge of the '
                'coordinates already stored could not start because a source backfill is '
                'running. Start it when that job ends: POST /source_backfill/runs with '
                '{"steps": ["location_purge"]}'
            )
            raise RequestRejectedError(
                409, {'error': 'job_busy', 'message': msg, 'revision': revision}
            ) from exc
        job_id = str(started.job.id)
    return LocationPolicyResponse(
        location_policy=cast('LocationPolicyName', stored),
        revision=revision,
        purge_job_id=job_id,
    )


__all__ = ['KEEPS', 'RevisionConflictError', 'change_policy', 'get_policy']
