"""Activation of a partition: one OCC transaction, then a fenced projection (plan 5.1, 7.1).

``activate`` moves ``cluster_activations.pending_partition_id`` to the new partition only if the
caller saw the current serving partition and revision (zero rows: 409 ``partition_conflict``),
records the event and the ``search_projection_tasks`` row so the projection cannot be forgotten.
``complete_projection`` flips ``serving_partition_id`` to the pending one and is itself fenced on
the pending id. ``auto_activation_allowed`` is the auto-activation guard (L1b, J3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.services.events.publish import apublish


if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

MIN_AUTO_STABILITY = 0.95
PROJECTION_KIND = 'cluster_partition'
EVENT_ACTIVATION = 'cluster.activation'


class PartitionConflictError(Exception):
    """409 ``partition_conflict``: the serving partition or its revision is not what the caller saw."""

    code = 'partition_conflict'
    status = 409


class PartitionNotReadyError(ValueError):
    """The partition is missing, not ``ready``, or belongs to another key."""


@dataclass(frozen=True)
class Activation:
    project_id: uuid.UUID | None
    target: str
    level: str
    embedding_field: str
    serving_partition_id: uuid.UUID
    pending_partition_id: uuid.UUID | None
    revision: int


def auto_activation_allowed(diff: Mapping[str, Any] | None, *, activation: str) -> bool:
    """Auto only when the profile allows it, no anchored or validated member moved and at least
    95 percent of the parent's cluster identities were inherited (a first run has no parent)."""
    if activation != 'auto_when_safe' or diff is None:
        return False
    if int(diff.get('anchored_moves', 0)) or int(diff.get('validated_moves', 0)):
        return False
    return float(diff.get('stability', 0.0)) >= MIN_AUTO_STABILITY


async def _ready_partition(
    session: AsyncSession, project: uuid.UUID | None, partition_id: uuid.UUID
) -> tuple[str, str, str]:
    row = (
        await session.execute(
            text(
                'SELECT target, level, embedding_field, status, project_id IS NOT DISTINCT FROM '
                ':p FROM cluster_partitions WHERE id = :id'
            ),
            {'id': partition_id, 'p': project},
        )
    ).first()
    if row is None or row[3] != 'ready' or not row[4]:
        raise PartitionNotReadyError(f'partition {partition_id} is not a ready partition')
    return row[0], row[1], row[2]


async def get_activation(
    session: AsyncSession,
    *,
    project: uuid.UUID | None,
    target: str,
    level: str,
    embedding_field: str,
) -> Activation | None:
    row = (
        await session.execute(
            text(
                'SELECT serving_partition_id, pending_partition_id, revision FROM '
                'cluster_activations WHERE project_id IS NOT DISTINCT FROM :p AND target = :t '
                'AND level = :l AND embedding_field = :f'
            ),
            {'p': project, 't': target, 'l': level, 'f': embedding_field},
        )
    ).first()
    if row is None:
        return None
    return Activation(project, target, level, embedding_field, row[0], row[1], int(row[2]))


async def activate(
    session: AsyncSession,
    *,
    project: uuid.UUID | None,
    partition_id: uuid.UUID,
    expected_serving: uuid.UUID | None,
    expected_revision: int | None,
    by: str,
    reason: str = 'human',
) -> Activation:
    """Make ``partition_id`` the pending (then serving) partition of its key, or raise 409."""
    target, level, field = await _ready_partition(session, project, partition_id)
    key = {'p': project, 't': target, 'l': level, 'f': field}
    current = await get_activation(
        session, project=project, target=target, level=level, embedding_field=field
    )
    if current is None:
        if expected_serving is not None:
            raise PartitionConflictError('nothing is serving for this key yet')
        inserted = await session.execute(
            text(
                'INSERT INTO cluster_activations (project_id, target, level, embedding_field, '
                'serving_partition_id, pending_partition_id, updated_by) VALUES (:p, :t, :l, :f, '
                ':new, :new, :by) ON CONFLICT ON CONSTRAINT uq_cluster_activations_served_key '
                'DO NOTHING RETURNING 1'
            ),
            {**key, 'new': partition_id, 'by': by},
        )
        if inserted.first() is None:
            raise PartitionConflictError('another activation won the first write')
        revision = 1
    else:
        if current.serving_partition_id == partition_id:
            raise PartitionConflictError('the partition is already serving')
        updated = await session.execute(
            text(
                'UPDATE cluster_activations SET pending_partition_id = :new, '
                'revision = revision + 1, updated_at = now(), updated_by = :by '
                'WHERE project_id IS NOT DISTINCT FROM :p AND target = :t AND level = :l '
                'AND embedding_field = :f AND serving_partition_id = :serving '
                'AND revision = :rev RETURNING revision'
            ),
            {
                **key,
                'new': partition_id,
                'by': by,
                'serving': expected_serving,
                'rev': expected_revision,
            },
        )
        found = updated.scalar_one_or_none()
        if found is None:
            raise PartitionConflictError('the serving partition or its revision moved')
        revision = int(found)
    await session.execute(
        text(
            'INSERT INTO cluster_activation_events (project_id, target, level, embedding_field, '
            'from_partition_id, to_partition_id, reason, by) VALUES (:p, :t, :l, :f, :old, :new, '
            ':reason, :by)'
        ),
        {
            **key,
            'old': current.serving_partition_id if current else None,
            'new': partition_id,
            'reason': reason,
            'by': by,
        },
    )
    if project is not None:
        await session.execute(
            text(
                'INSERT INTO search_projection_tasks (project_id, kind, payload) '
                'VALUES (:p, :kind, CAST(:payload AS jsonb))'
            ),
            {
                'p': project,
                'kind': PROJECTION_KIND,
                'payload': json.dumps(
                    {
                        'partition_id': str(partition_id),
                        'target': target,
                        'level': level,
                        'embedding_field': field,
                    }
                ),
            },
        )
        await apublish(
            EVENT_ACTIVATION,
            {
                'partition_id': str(partition_id),
                'target': target,
                'level': level,
                'reason': reason,
                'revision': revision,
            },
            project=project,
            session=session,
        )
    return Activation(
        project,
        target,
        level,
        field,
        current.serving_partition_id if current else partition_id,
        partition_id,
        revision,
    )


async def complete_projection(
    session: AsyncSession,
    *,
    project: uuid.UUID | None,
    partition_id: uuid.UUID,
) -> bool:
    """Flip serving to ``partition_id`` if it is still the pending one; ``False`` means a newer
    activation replaced it (the projection wrote nothing that outlives it)."""
    target, level, field = await _ready_partition(session, project, partition_id)
    flipped = await session.execute(
        text(
            'UPDATE cluster_activations SET serving_partition_id = pending_partition_id, '
            'pending_partition_id = NULL, updated_at = now() '
            'WHERE project_id IS NOT DISTINCT FROM :p AND target = :t AND level = :l '
            'AND embedding_field = :f AND pending_partition_id = :id RETURNING 1'
        ),
        {'p': project, 't': target, 'l': level, 'f': field, 'id': partition_id},
    )
    return flipped.first() is not None


__all__ = [
    'MIN_AUTO_STABILITY',
    'Activation',
    'PartitionConflictError',
    'PartitionNotReadyError',
    'activate',
    'auto_activation_allowed',
    'complete_projection',
    'get_activation',
]
