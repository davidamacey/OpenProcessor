"""Shared builders for the near-duplicate engine tests: vectors, documents, the run environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import psycopg

from src.services.near_duplicates.plan import LevelPlan
from src.services.near_duplicates.profiles import active_profile
from src.services.near_duplicates.runs import Env

from .conftest import scope_for
from .fake_opensearch import FakeNearDupOpenSearch


if TYPE_CHECKING:
    import uuid


DIM = 32
IMAGES = 'op_prj_nd__images'


def basis(count: int) -> np.ndarray:
    """``count`` orthonormal directions: two different directions have cosine 0."""
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(DIM, count)))
    return q.T.astype(np.float32)


def near(direction: np.ndarray, seed: int) -> list[float]:
    """A vector within cosine ~0.9999 of ``direction``."""
    noise = np.random.default_rng(seed).normal(size=direction.shape).astype(np.float32)
    vec = direction + 0.005 * noise / np.linalg.norm(noise)
    return (vec / np.linalg.norm(vec)).tolist()


def image_docs(groups: dict[str, list[str]], singles: list[str]) -> dict[str, dict[str, Any]]:
    """Group name -> member ids (one direction per group), then one direction per single."""
    directions = basis(len(groups) + len(singles))
    docs: dict[str, dict[str, Any]] = {}
    seed = 0
    for i, members in enumerate(groups.values()):
        for member in members:
            seed += 1
            docs[member] = {'pe_embedding': near(directions[i], seed)}
    for j, single in enumerate(singles):
        docs[single] = {'pe_embedding': directions[len(groups) + j].tolist()}
    return docs


def plan(level: str = 'image', *, index: str = IMAGES) -> LevelPlan:
    return LevelPlan(
        level=level,
        index=index,
        field='pe_embedding',
        space_key='pe_core_l14_336_v1',
        dedup_cosine=0.98,
        leak_cosine=0.98,
        k=64,
        max_group_size=1000,
        same_detector_class=level == 'item',
    )


def make_job(pg: Any, project: uuid.UUID, kind: str = 'noop') -> uuid.UUID:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        return conn.execute(
            'INSERT INTO jobs (project_id, kind, queue, status, max_queue_wait, requested_by) '
            "VALUES (%s, %s, 'jobs', 'queued', interval '1 hour', 'test') RETURNING id",
            (project, kind),
        ).fetchone()[0]


async def make_env(
    pg: Any,
    project: uuid.UUID,
    docs: dict[str, dict[str, dict[str, Any]]],
    *,
    level: str = 'image',
    index: str = IMAGES,
) -> tuple[Env, FakeNearDupOpenSearch]:
    client = FakeNearDupOpenSearch(docs)
    scope = scope_for(project)
    async with scope() as session:
        profile = await active_profile(session, project)
    env = Env(
        scope=scope,
        client=client,  # type: ignore[arg-type]
        project_id=project,
        profile=profile,
        plan=plan(level, index=index),
        job_id=make_job(pg, project),
    )
    return env, client


def snapshot(
    pg: Any, run_id: uuid.UUID, purpose: str
) -> dict[frozenset[str], tuple[str, uuid.UUID]]:
    """``members -> (representative, lineage)`` of one run's groups."""
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        rows = conn.execute(
            'SELECT g.id, g.representative_id, g.lineage_id, m.member_id FROM near_dup_groups g '
            'JOIN near_dup_members m ON m.group_id = g.id WHERE g.run_id = %s AND g.purpose = %s',
            (run_id, purpose),
        ).fetchall()
    by_group: dict[uuid.UUID, tuple[str, uuid.UUID, set[str]]] = {}
    for gid, rep, lineage, member in rows:
        by_group.setdefault(gid, (rep, lineage, set()))[2].add(member)
    return {frozenset(m): (rep, lineage) for rep, lineage, m in by_group.values()}
