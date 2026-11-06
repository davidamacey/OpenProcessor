"""The fenced flip of a target's primary space (embeddings plan 5.5).

One transaction demotes the old primary and promotes the candidate, each guarded by the
``revision`` read at the start (a concurrent flip or status move makes the fence fail with 409),
then bumps ``projects.config_epoch`` so every worker re-resolves its spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.db.repos import embedding_spaces as repo
from src.db.session import project_tx
from src.domain.embedding_spaces import EmbeddingSpaceError, Role, Status, Target


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class FlipResult:
    primary: repo.SpaceRow
    previous: repo.SpaceRow | None
    config_epoch: int


async def promote(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    target: Target,
    key: str,
    force: bool = False,
) -> FlipResult:
    """Make ``key`` the primary space of ``target``.

    409 ``embedding_space_not_ready`` unless the candidate is ``ready``; 409
    ``embedding_evaluation_missing`` unless an evaluation run for (current primary, candidate)
    exists, which ``force=True`` waives (RD-35). Promoting the current primary is refused too:
    nothing would change, so no epoch is bumped.
    """
    async with project_tx(session, project_id):
        candidate = await repo.get_space(session, project_id, target, key)
        if candidate is None:
            raise EmbeddingSpaceError(
                'embedding_space_not_found', f'no {target.value} space {key}', status=404
            )
        if candidate.role is Role.PRIMARY:
            raise EmbeddingSpaceError(
                'embedding_space_already_primary', f'{key} is already the primary'
            )
        if candidate.status is not Status.READY:
            raise EmbeddingSpaceError(
                'embedding_space_not_ready', f'{key} is {candidate.status.value}, not ready'
            )
        old = await repo.get_primary_space(session, project_id, target)
        if (
            old is not None
            and not force
            and not await repo.has_evaluation(session, project_id, old.id, candidate.id)
        ):
            raise EmbeddingSpaceError(
                'embedding_evaluation_missing',
                f'no evaluation of {key} against {old.space_key}; run one or pass force',
            )
        demoted = None
        if old is not None:
            demoted = await repo.set_role(
                session, old.id, expected_revision=old.revision, role=Role.OPTIONAL
            )
            if demoted is None:
                raise EmbeddingSpaceError(
                    'embedding_space_changed', f'{old.space_key} changed during the flip'
                )
        promoted = await repo.set_role(
            session, candidate.id, expected_revision=candidate.revision, role=Role.PRIMARY
        )
        if promoted is None:
            raise EmbeddingSpaceError('embedding_space_changed', f'{key} changed during the flip')
        epoch = await repo.bump_config_epoch(session, project_id)
        return FlipResult(primary=promoted, previous=demoted, config_epoch=epoch)
