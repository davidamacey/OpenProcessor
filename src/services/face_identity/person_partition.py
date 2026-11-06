"""Person grouping as a job that records a partition (research clustering plan 5.2, smaller option).

``persons_cluster`` runs today's grouping (``auto_cluster_faces``: exact range search, connected
components, stable person ids) and then records the result as a ``ready`` face partition
(``target = 'face'``, ``method = 'union_find'``, no project), activated so ``GET /persons/cluster``
reads it. A person id maps to the stable cluster uid ``uuid5(PERSON_NAMESPACE, person_id)``, so a
person keeps its identity across runs without a lookup.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.services.clustering.activation import activate, complete_projection, get_activation
from src.services.clustering.partitions import (
    PartitionMember,
    PartitionRecord,
    PartitionSpace,
    record_partition,
)
from src.services.face_identity import get_face_identity_service


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.tasks.runtime.context import JobContext

PERSON_NAMESPACE = uuid.UUID('6f0f6a53-5d3c-5e8e-9c1e-0a4b7a2f9d11')
FACE_SPACE = PartitionSpace('embedding', 'arcface', 'arcface')
FACE_FIELD = FACE_SPACE.embedding_field
ACTOR = 'system:persons_cluster'


def person_uid(person_id: str) -> uuid.UUID:
    return uuid.uuid5(PERSON_NAMESPACE, person_id)


async def record_person_partition(
    session: AsyncSession,
    assignments: Mapping[str, str],
    params: Mapping[str, Any],
    *,
    job_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Write the face partition of ``assignments`` (face id to person id) and make it serve."""
    members = [
        PartitionMember(face_id, person_uid(person_id))
        for face_id, person_id in sorted(assignments.items())
    ]
    current = await get_activation(
        session, project=None, target='face', level='identity', embedding_field=FACE_FIELD
    )
    partition_id = await record_partition(
        session,
        project=None,
        target='face',
        level='identity',
        space=FACE_SPACE,
        method='union_find',
        params=params,
        members=members,
        record=PartitionRecord(
            parent_partition_id=current.serving_partition_id if current else None,
            job_id=job_id,
            created_by=ACTOR,
        ),
    )
    await activate(
        session,
        project=None,
        partition_id=partition_id,
        expected_serving=current.serving_partition_id if current else None,
        expected_revision=current.revision if current else None,
        by=ACTOR,
        reason='auto',
    )
    # Faces keep person_id in the faces index (no projection job), so the flip is immediate.
    await complete_projection(session, project=None, partition_id=partition_id)
    return partition_id


@dataclass(frozen=True)
class PersonsGroupingResult:
    partition_id: uuid.UUID
    n_faces: int
    n_persons: int
    params: dict[str, Any]
    created_at: str


async def persons_grouping_result(session: AsyncSession) -> PersonsGroupingResult | None:
    """The serving face partition's summary, or ``None`` before the first run."""
    row = (
        await session.execute(
            text(
                'SELECT p.id, p.n_items, p.n_clusters, p.params, p.created_at::text '
                'FROM cluster_activations a JOIN cluster_partitions p '
                'ON p.id = a.serving_partition_id WHERE a.project_id IS NULL '
                "AND a.target = 'face' AND a.level = 'identity' AND a.embedding_field = :f"
            ),
            {'f': FACE_FIELD},
        )
    ).first()
    if row is None:
        return None
    return PersonsGroupingResult(row[0], row[1], row[2], dict(row[3]), row[4])


async def persons_cluster(ctx: JobContext) -> dict[str, Any]:
    """Group every face (the existing grouping), then record the result as a face partition."""
    service = get_face_identity_service()
    threshold = float(ctx.params['similarity_threshold'])
    min_faces = int(ctx.params['min_faces_per_person'])
    params = {'similarity_threshold': threshold, 'min_faces_per_person': min_faces}
    await ctx.progress(0, None, 'group')
    result = await service.auto_cluster_faces(
        similarity_threshold=threshold, min_faces_per_person=min_faces
    )
    if result.get('status') == 'error':
        raise RuntimeError(str(result.get('error') or 'person grouping failed'))
    await ctx.progress(0, None, 'record')
    assignments = await service.person_assignments()
    async with ctx.session() as session:
        partition_id = await record_person_partition(
            session, assignments, params, job_id=ctx.job_id
        )
    return {**result, 'partition_id': str(partition_id)}
