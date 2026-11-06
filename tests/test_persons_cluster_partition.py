"""One persons_cluster run writes one ready face partition that GET /persons/cluster reads."""

from __future__ import annotations

import contextlib
import uuid
from typing import Any

import pytest
from sqlalchemy import text

from src.db.session import async_session_factory, deployment_tx
from src.services.face_identity import get_face_identity_service
from src.services.face_identity.person_partition import (
    person_uid,
    persons_cluster,
    persons_grouping_result,
)


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


class _Ctx:
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.job_id = None
        self.stages: list[str | None] = []

    async def progress(self, _done, _total=None, stage=None, *_a, **_k) -> None:
        self.stages.append(stage)

    @contextlib.asynccontextmanager
    async def session(self):
        async with async_session_factory()() as session, deployment_tx(session, 'test'):
            yield session


@pytest.fixture
def faces(monkeypatch):
    service = get_face_identity_service()
    state = {'assign': {'f1': 'person_a', 'f2': 'person_a', 'f3': 'person_b'}}

    async def auto_cluster_faces(**kwargs):
        return {'status': 'success', 'total_faces': 3, **kwargs}

    async def person_assignments():
        return dict(state['assign'])

    monkeypatch.setattr(service, 'auto_cluster_faces', auto_cluster_faces)
    monkeypatch.setattr(service, 'person_assignments', person_assignments)
    return state


async def _count(where: str) -> int:
    async with async_session_factory()() as session, deployment_tx(session, 'test'):
        return (
            await session.execute(text(f'SELECT count(*) FROM cluster_partitions WHERE {where}'))
        ).scalar_one()


async def test_one_run_writes_one_ready_face_partition_and_get_reads_it(pg_env, faces) -> None:
    before = await _count("target = 'face'")
    ctx = _Ctx({'similarity_threshold': 0.7, 'min_faces_per_person': 2})
    result = await persons_cluster(ctx)
    assert await _count("target = 'face'") == before + 1
    async with async_session_factory()() as session, deployment_tx(session, 'test'):
        found = await persons_grouping_result(session)
        status = (
            await session.execute(
                text('SELECT status, method, project_id FROM cluster_partitions WHERE id = :id'),
                {'id': uuid.UUID(result['partition_id'])},
            )
        ).one()
    assert found is not None
    assert (found.n_faces, found.n_persons) == (3, 2)
    assert str(found.partition_id) == result['partition_id']
    assert status == ('ready', 'union_find', None)


async def test_a_rerun_keeps_person_identities_and_replaces_the_serving_partition(
    pg_env, faces
) -> None:
    first = await persons_cluster(_Ctx({'similarity_threshold': 0.7, 'min_faces_per_person': 2}))
    faces['assign']['f4'] = 'person_a'
    second = await persons_cluster(_Ctx({'similarity_threshold': 0.7, 'min_faces_per_person': 2}))
    assert first['partition_id'] != second['partition_id']
    async with async_session_factory()() as session, deployment_tx(session, 'test'):
        uids = (
            (
                await session.execute(
                    text(
                        'SELECT DISTINCT cluster_uid FROM cluster_memberships WHERE partition_id = :p'
                    ),
                    {'p': uuid.UUID(second['partition_id'])},
                )
            )
            .scalars()
            .all()
        )
        found = await persons_grouping_result(session)
    assert set(uids) == {person_uid('person_a'), person_uid('person_b')}
    assert found is not None
    assert str(found.partition_id) == second['partition_id']
    assert found.n_faces == 4
