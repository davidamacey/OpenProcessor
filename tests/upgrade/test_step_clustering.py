"""The clustering step: one ready item partition per project from the 0.6.0 candidate clusters."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fixtures.job_rows import make_project
from sqlalchemy import text

from src.db.session import async_session_factory, deployment_tx
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.upgrade.from_060.steps import clustering as step


OFFSET = RESIDUAL_CLUSTER_ID_OFFSET


def test_only_candidate_clusters_become_members_and_the_uid_is_stable() -> None:
    project = uuid.uuid4()
    docs = [
        ('a', {'cluster_id': OFFSET + 3, 'cluster_distance': 0.2}),
        ('b', {'cluster_id': OFFSET + 3}),
        ('c', {'cluster_id': 7}),  # a class id: the overload the rebuild removes
        ('d', {'cluster_id': None}),
        ('e', {'cluster_id': OFFSET + 9}),
    ]
    members = step.members_of(project, docs)
    assert [m.item_key for m in members] == ['a', 'b', 'e']
    assert members[0].cluster_uid == members[1].cluster_uid != members[2].cluster_uid
    assert members[0].score == pytest.approx(0.8)
    assert members[1].score is None
    assert step.members_of(project, docs)[0].cluster_uid == members[0].cluster_uid


class _Reader:
    def __init__(self, docs):
        self.docs = docs

    async def iter_docs(self, _index):
        for item in self.docs:
            yield item


@pytest.mark.db
@pytest.mark.asyncio
async def test_the_served_view_after_the_step_equals_the_060_assignment(
    pg_env, pg, bound_async_engine
) -> None:
    project = make_project(pg)
    docs = [(f'c{i}', {'cluster_id': OFFSET + i % 3, 'cluster_distance': 0.1}) for i in range(9)]
    docs.append(('class_item', {'cluster_id': 2}))
    counts: dict[str, int] = {}
    report = SimpleNamespace(add=counts.__setitem__)
    parsed = SimpleNamespace(
        projects=[SimpleNamespace(project=SimpleNamespace(slug='p', indexes={'items': 'p__items'}))]
    )
    for _ in range(2):  # idempotent: the second pass adds nothing
        async with async_session_factory()() as session, deployment_tx(session, 'test'):
            ctx = SimpleNamespace(
                session=session,
                parsed=parsed,
                project_ids={'p': project},
                reader=_Reader(docs),
                report=report,
            )
            await step.run(ctx)
            assert await step.verify(ctx) == []
    async with async_session_factory()() as session, deployment_tx(session, 'test'):
        rows = (
            await session.execute(
                text(
                    'SELECT p.status, p.method, p.n_items, p.n_clusters, a.serving_partition_id = p.id, '
                    'e.reason FROM cluster_partitions p JOIN cluster_activations a ON '
                    'a.project_id = p.project_id JOIN cluster_activation_events e ON '
                    'e.to_partition_id = p.id WHERE p.project_id = :p'
                ),
                {'p': project},
            )
        ).all()
        tasks = (
            await session.execute(
                text(
                    "SELECT count(*) FROM search_projection_tasks WHERE kind = 'cluster_partition'"
                )
            )
        ).scalar_one()
    assert rows == [('ready', 'ivf', 9, 3, True, 'upgrade')]
    assert tasks >= 1
