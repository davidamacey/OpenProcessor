"""``record_partition``: write one immutable partition in one transaction (plan 5.1).

The partition, its clusters, their identities and every membership are written in the caller's
scoped transaction and the partition is marked ``ready`` last, so a crash leaves nothing behind.
Memberships go in with ``COPY`` (https://www.psycopg.org/psycopg3/docs/basic/copy.html) through a
temporary table, the only way a 1M-row snapshot stays a matter of seconds under row-level security. Identities are supplied by the caller: a carried
identity keeps its uid (``carry_over``), a new one is minted by ``mint_uids`` (the database's ``uuidv7()``).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.domain import clustering_vocab as vocab


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.domain.embedding_spaces import ResolvedSpace

# PostgreSQL refuses COPY FROM into a table with row-level security ("COPY FROM not supported with
# row-level security", https://www.postgresql.org/docs/current/sql-copy.html), so the rows are
# copied into a transaction-local temporary table (no policy) and moved with one INSERT ... SELECT,
# which the policy checks like any insert.
CREATE_STAGING = (
    'CREATE TEMPORARY TABLE IF NOT EXISTS cluster_memberships_staging (item_key text, cluster_uid uuid, state text, '
    'score real, margin real) ON COMMIT DROP'
)
COPY_STAGING = (
    'COPY cluster_memberships_staging (item_key, cluster_uid, state, score, margin) FROM STDIN'
)
MOVE_STAGING = (
    'INSERT INTO cluster_memberships (partition_id, project_id, item_key, cluster_uid, state, '
    'score, margin) SELECT :partition, :project, item_key, cluster_uid, state, score, margin '
    'FROM cluster_memberships_staging'
)
UNASSIGNED = 'unassigned'


class PartitionError(ValueError):
    """The members or the record break a rule of the partition tables."""


@dataclass(frozen=True)
class PartitionSpace:
    """The vector space a partition was fitted in; a calibration is valid for exactly one."""

    embedding_field: str
    embedding_model: str
    preprocess_id: str

    @classmethod
    def of(cls, space: ResolvedSpace) -> PartitionSpace:
        """The registry's space: its key names the model version and the preprocessing."""
        return cls(
            embedding_field=space.vector_field,
            embedding_model=f'{space.model_key}@v{space.model_version}',
            preprocess_id=space.space_key,
        )


@dataclass(frozen=True)
class PartitionMember:
    item_key: str
    cluster_uid: uuid.UUID | None
    state: str = 'assigned'
    score: float | None = None
    margin: float | None = None


@dataclass(frozen=True)
class ClusterLineage:
    """How one cluster of the partition descends from the parent's (``carry_over``)."""

    transition: str = 'new'
    predecessor_uids: tuple[uuid.UUID, ...] = ()


@dataclass(frozen=True)
class PartitionRecord:
    """Provenance and measurements that are not part of the fixed ``record_partition`` call."""

    method_version: str = '1'
    profile: tuple[uuid.UUID, int] | None = None
    calibration_id: uuid.UUID | None = None
    pool_filter: Mapping[str, Any] = field(default_factory=dict)
    input_digest: str = ''
    seed: int = 0
    backend: str = 'cpu'
    backend_detail: Mapping[str, Any] = field(default_factory=dict)
    parent_partition_id: uuid.UUID | None = None
    job_id: uuid.UUID | None = None
    created_by: str = 'system'
    metrics: Mapping[str, Any] = field(default_factory=dict)
    diff: Mapping[str, Any] | None = None
    stage_durations: Mapping[str, Any] = field(default_factory=dict)
    lineage: Mapping[uuid.UUID, ClusterLineage] = field(default_factory=dict)
    prototypes: Mapping[uuid.UUID, Sequence[Mapping[str, Any]]] = field(default_factory=dict)
    dispersion: Mapping[uuid.UUID, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _Cluster:
    uid: uuid.UUID
    size: int
    first_key: str


def _validate(members: Sequence[PartitionMember]) -> None:
    seen: set[str] = set()
    for member in members:
        if member.state not in vocab.MEMBERSHIP_STATES:
            raise PartitionError(f'unknown membership state {member.state!r}')
        if (member.state == UNASSIGNED) != (member.cluster_uid is None):
            raise PartitionError(f'{member.item_key}: unassigned iff no cluster')
        if member.item_key in seen:
            raise PartitionError(f'{member.item_key} is a member twice')
        seen.add(member.item_key)


def _clusters(members: Sequence[PartitionMember]) -> list[_Cluster]:
    """Largest first, then by smallest member key: the order of the UI numbers."""
    sizes: Counter[uuid.UUID] = Counter()
    first: dict[uuid.UUID, str] = {}
    for member in members:
        if member.cluster_uid is None:
            continue
        sizes[member.cluster_uid] += 1
        if member.cluster_uid not in first or member.item_key < first[member.cluster_uid]:
            first[member.cluster_uid] = member.item_key
    return sorted(
        (_Cluster(uid, n, first[uid]) for uid, n in sizes.items()),
        key=lambda c: (-c.size, c.first_key),
    )


async def _insert_partition(
    session: AsyncSession,
    *,
    project: uuid.UUID | None,
    target: str,
    level: str,
    space: PartitionSpace,
    method: str,
    params: Mapping[str, Any],
    members: Sequence[PartitionMember],
    clusters: Sequence[_Cluster],
    record: PartitionRecord,
) -> uuid.UUID:
    config_id, revision = record.profile or (None, None)
    counts = Counter(m.state for m in members)
    row = await session.execute(
        text(
            'INSERT INTO cluster_partitions (project_id, target, level, embedding_field, '
            'embedding_model, preprocess_id, method, method_version, profile_config_id, '
            'profile_revision, params, calibration_id, pool_filter, input_digest, seed, backend, '
            'backend_detail, parent_partition_id, job_id, status, n_items, n_clusters, n_review, '
            'n_unassigned, metrics, diff, stage_durations, created_by) VALUES (:project, :target, '
            ':level, :field, :model, :preprocess, :method, :method_version, :config_id, :revision, '
            'CAST(:params AS jsonb), :calibration, CAST(:pool_filter AS jsonb), :digest, :seed, '
            ":backend, CAST(:backend_detail AS jsonb), :parent, :job, 'building', :n_items, "
            ':n_clusters, :n_review, :n_unassigned, CAST(:metrics AS jsonb), CAST(:diff AS jsonb), '
            'CAST(:durations AS jsonb), :created_by) RETURNING id'
        ),
        {
            'project': project,
            'target': target,
            'level': level,
            'field': space.embedding_field,
            'model': space.embedding_model,
            'preprocess': space.preprocess_id,
            'method': method,
            'method_version': record.method_version,
            'config_id': config_id,
            'revision': revision,
            'params': json.dumps(params),
            'calibration': record.calibration_id,
            'pool_filter': json.dumps(record.pool_filter),
            'digest': record.input_digest,
            'seed': record.seed,
            'backend': record.backend,
            'backend_detail': json.dumps(record.backend_detail),
            'parent': record.parent_partition_id,
            'job': record.job_id,
            'n_items': len(members),
            'n_clusters': len(clusters),
            'n_review': counts['review'],
            'n_unassigned': counts[UNASSIGNED],
            'metrics': json.dumps(record.metrics),
            'diff': None if record.diff is None else json.dumps(record.diff),
            'durations': json.dumps(record.stage_durations),
            'created_by': record.created_by,
        },
    )
    partition_id: uuid.UUID = row.scalar_one()
    return partition_id


async def _insert_clusters(
    session: AsyncSession,
    *,
    partition_id: uuid.UUID,
    project: uuid.UUID | None,
    target: str,
    level: str,
    clusters: Sequence[_Cluster],
    record: PartitionRecord,
) -> None:
    if not clusters:
        return
    await session.execute(
        text(
            'INSERT INTO cluster_identities (cluster_uid, project_id, target, level, '
            'first_partition_id, last_partition_id) '
            'SELECT u, :project, :target, :level, :partition, :partition FROM unnest(CAST(:uids '
            'AS uuid[])) AS u ON CONFLICT (cluster_uid) DO UPDATE SET '
            'last_partition_id = EXCLUDED.last_partition_id'
        ),
        {
            'project': project,
            'target': target,
            'level': level,
            'partition': partition_id,
            'uids': [c.uid for c in clusters],
        },
    )
    rows = []
    for number, cluster in enumerate(clusters, start=1):
        lineage = record.lineage.get(cluster.uid, ClusterLineage())
        parent = lineage.predecessor_uids[0] if lineage.transition == 'split' else None
        rows.append(
            {
                'partition': partition_id,
                'uid': cluster.uid,
                'project': project,
                'display_no': number,
                'parent': parent,
                'transition': lineage.transition,
                'predecessors': list(lineage.predecessor_uids),
                'size': cluster.size,
                'dispersion': record.dispersion.get(cluster.uid),
                'prototypes': json.dumps(list(record.prototypes.get(cluster.uid, ()))),
            }
        )
    await session.execute(
        text(
            'INSERT INTO partition_clusters (partition_id, cluster_uid, project_id, display_no, '
            'parent_uid, transition, predecessor_uids, size, dispersion, prototypes) VALUES '
            '(:partition, :uid, :project, :display_no, :parent, :transition, '
            'CAST(:predecessors AS uuid[]), :size, :dispersion, CAST(:prototypes AS jsonb))'
        ),
        rows,
    )


async def _copy_memberships(
    session: AsyncSession,
    partition_id: uuid.UUID,
    project: uuid.UUID | None,
    members: Iterable[PartitionMember],
) -> None:
    await session.execute(text(CREATE_STAGING))
    connection = await session.connection()
    raw = await connection.get_raw_connection()
    async with raw.driver_connection.cursor() as cursor, cursor.copy(COPY_STAGING) as copy:  # type: ignore[union-attr]
        for m in members:
            await copy.write_row((m.item_key, m.cluster_uid, m.state, m.score, m.margin))
    await session.execute(text(MOVE_STAGING), {'partition': partition_id, 'project': project})
    await session.execute(text('TRUNCATE cluster_memberships_staging'))


async def mint_uids(session: AsyncSession, count: int) -> list[uuid.UUID]:
    """``count`` fresh cluster identities from the database's ``uuidv7()`` (time-ordered)."""
    if count <= 0:
        return []
    rows = await session.execute(text('SELECT uuidv7() FROM generate_series(1, :n)'), {'n': count})
    return [row[0] for row in rows]


async def record_partition(
    session: AsyncSession,
    *,
    project: uuid.UUID | None,
    target: str,
    level: str,
    space: PartitionSpace | ResolvedSpace,
    method: str,
    params: Mapping[str, Any],
    members: Sequence[PartitionMember],
    record: PartitionRecord | None = None,
) -> uuid.UUID:
    """Write a partition, its clusters and memberships; return its id, status ``ready``.

    ``project`` is ``None`` only for a core partition (faces) written under ``core_tx``. The
    caller's scope decides what the row-level policy lets through; a mismatch raises.
    """
    if not isinstance(space, PartitionSpace):
        space = PartitionSpace.of(space)
    record = record or PartitionRecord()
    _validate(members)
    clusters = _clusters(members)
    partition_id = await _insert_partition(
        session,
        project=project,
        target=target,
        level=level,
        space=space,
        method=method,
        params=params,
        members=members,
        clusters=clusters,
        record=record,
    )
    await _insert_clusters(
        session,
        partition_id=partition_id,
        project=project,
        target=target,
        level=level,
        clusters=clusters,
        record=record,
    )
    await _copy_memberships(session, partition_id, project, members)
    await session.execute(
        text("UPDATE cluster_partitions SET status = 'ready', finished_at = now() WHERE id = :id"),
        {'id': partition_id},
    )
    return partition_id


__all__ = [
    'ClusterLineage',
    'PartitionError',
    'PartitionMember',
    'PartitionRecord',
    'PartitionRow',
    'PartitionSpace',
    'get_partition',
    'list_partitions',
    'mint_uids',
    'record_partition',
]


@dataclass(frozen=True)
class PartitionRow:
    id: uuid.UUID
    target: str
    level: str
    embedding_field: str
    method: str
    status: str
    n_items: int | None
    n_clusters: int | None
    n_review: int | None
    n_unassigned: int | None
    metrics: Mapping[str, Any]
    diff: Mapping[str, Any] | None
    created_at: str
    parent_partition_id: uuid.UUID | None
    serving: bool


_SELECT_PARTITION = (
    'SELECT p.id, p.target, p.level, p.embedding_field, p.method, p.status, p.n_items, '
    'p.n_clusters, p.n_review, p.n_unassigned, p.metrics, p.diff, p.created_at::text, '
    'p.parent_partition_id, COALESCE(a.serving_partition_id = p.id, false) AS serving '
    'FROM cluster_partitions p LEFT JOIN cluster_activations a ON a.project_id = p.project_id '
    'AND a.target = p.target AND a.level = p.level AND a.embedding_field = p.embedding_field '
)


async def list_partitions(
    session: AsyncSession, *, target: str | None = None, limit: int = 50
) -> list[PartitionRow]:
    """The project's partitions, newest first (the session's scope limits it to one project)."""
    rows = await session.execute(
        text(
            _SELECT_PARTITION + 'WHERE (CAST(:target AS text) IS NULL OR p.target = :target) '
            'ORDER BY p.created_at DESC, p.id DESC LIMIT :limit'
        ),
        {'target': target, 'limit': limit},
    )
    return [PartitionRow(*row) for row in rows]


async def get_partition(session: AsyncSession, partition_id: uuid.UUID) -> PartitionRow | None:
    row = (
        await session.execute(text(_SELECT_PARTITION + 'WHERE p.id = :id'), {'id': partition_id})
    ).first()
    return None if row is None else PartitionRow(*row)
