"""Step ``clustering``: one ready partition per project from the items' 0.6.0 cluster ids.

0.6.0 stored an integer ``cluster_id`` on every item: below ``RESIDUAL_CLUSTER_ID_OFFSET`` it was
the item's class id (the overload the rebuild removes), from the offset up it was a candidate
cluster of the IVF pass. Only the candidate clusters are clusters in the new model, so the
partition has exactly the items with ``cluster_id >= RESIDUAL_CLUSTER_ID_OFFSET``, one cluster per
old id (uid ``uuid5(project id, old id)``, so a re-run writes the same identities), method ``ivf``,
activated with reason ``upgrade``. The activation queues the projection of the new fields onto the
items, so the served view equals the 0.6.0 assignment once it ran. The state files
(``last_clustering.json``, the region-cluster markers) are job files, removed with the others.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import text

from src.domain.embedding_spaces import Target, legacy_names, space_key
from src.services.clustering.activation import activate
from src.services.clustering.partitions import PartitionMember, PartitionSpace, record_partition


if TYPE_CHECKING:
    from src.services.upgrade.from_060.steps import StepContext

ORDER = 40
RESIDUAL_CLUSTER_ID_OFFSET = 10000
"""0.6.0's first candidate-cluster id; below it a cluster_id was the item's class id."""
MODEL_KEY = 'pe_core_l14_336'
MODEL_VERSION = 1
ACTOR = 'upgrade'
METHOD_VERSION = 'v0.6.0-assignment'


def cluster_uid(project_id: uuid.UUID, old_cluster_id: int) -> uuid.UUID:
    return uuid.uuid5(project_id, f'cluster:{old_cluster_id}')


def members_of(project_id: uuid.UUID, docs: list[tuple[str, dict]]) -> list[PartitionMember]:
    """The candidate-cluster members, keyed by item id, scored ``1 - cluster_distance`` when known."""
    out = []
    for item_id, source in docs:
        cluster_id = source.get('cluster_id')
        if (
            isinstance(cluster_id, bool)
            or not isinstance(cluster_id, int)
            or cluster_id < RESIDUAL_CLUSTER_ID_OFFSET
        ):
            continue
        distance = source.get('cluster_distance')
        score = None if distance is None else max(0.0, 1.0 - float(distance))
        out.append(PartitionMember(item_id, cluster_uid(project_id, cluster_id), 'assigned', score))
    return out


async def run(ctx: StepContext) -> None:
    from src.services.clustering.partitions import PartitionRecord

    space = PartitionSpace(
        legacy_names(Target.ITEM).vector_field,
        f'{MODEL_KEY}@v{MODEL_VERSION}',
        space_key(MODEL_KEY, MODEL_VERSION),
    )
    for pd in ctx.parsed.projects:
        project_id = ctx.project_ids[pd.project.slug]
        served = await ctx.session.execute(
            text(
                "SELECT 1 FROM cluster_activations WHERE project_id = :p AND target = 'item' "
                "AND level = 'class'"
            ),
            {'p': project_id},
        )
        index = pd.project.indexes.get('items')
        if served.first() is not None or index is None:
            continue
        docs = [
            (item_id, source)
            async for item_id, source in ctx.reader.iter_docs(index)
            if source.get('cluster_id') is not None
        ]
        members = members_of(project_id, docs)
        if not members:
            continue
        partition = await record_partition(
            ctx.session,
            project=project_id,
            target='item',
            level='class',
            space=space,
            method='ivf',
            params={'source': 'v0.6.0 assignment'},
            members=members,
            record=PartitionRecord(method_version=METHOD_VERSION, created_by=ACTOR),
        )
        await activate(
            ctx.session,
            project=project_id,
            partition_id=partition,
            expected_serving=None,
            expected_revision=None,
            by=ACTOR,
            reason='upgrade',
        )
        ctx.report.add('cluster_partitions', 1)
        ctx.report.add('cluster_memberships', len(members))


async def verify(ctx: StepContext) -> list[str]:
    """Every project that had candidate clusters has a serving item partition."""
    problems = []
    for pd in ctx.parsed.projects:
        project_id = ctx.project_ids[pd.project.slug]
        row = await ctx.session.execute(
            text(
                "SELECT count(*) FROM cluster_activations WHERE project_id = :p AND target = 'item'"
            ),
            {'p': project_id},
        )
        index = pd.project.indexes.get('items')
        if index is None or row.scalar_one() > 0:
            continue
        has_candidates = any(
            [
                isinstance(s.get('cluster_id'), int)
                and s['cluster_id'] >= RESIDUAL_CLUSTER_ID_OFFSET
                async for _id, s in ctx.reader.iter_docs(index)
            ]
        )
        if has_candidates:
            problems.append(f'{pd.project.slug}: candidate clusters have no served partition')
    return problems
