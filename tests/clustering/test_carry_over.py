"""Identities follow members: the same vectors in any arrival order keep every cluster_uid."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from _config_fixtures import make_project
from sklearn.metrics import adjusted_rand_score
from sqlalchemy import text

from src.db.session import async_session_factory, project_tx
from src.services.clustering.partitions import PartitionSpace
from src.services.clustering.run import fit_pool, run_partition, unit_rows


pytestmark = [pytest.mark.db, pytest.mark.asyncio]

SPACE = PartitionSpace('pe_embedding', 'pe_core@v1', 'pe_core_v1')


def _blobs(n_per: int = 60, n_blobs: int = 8, dim: int = 32, seed: int = 3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_blobs, dim)) * 4.0
    vectors = np.vstack([c + rng.normal(size=(n_per, dim)) * 0.3 for c in centers])
    keys = [f'crop{i:05d}' for i in range(len(vectors))]
    truth = np.repeat(np.arange(n_blobs), n_per)
    return keys, vectors.astype(np.float32), truth


@pytest.fixture
def project(pg) -> uuid.UUID:
    return make_project(pg, f'co-{uuid.uuid4().hex[:10]}')


async def _assignment(session, partition) -> dict[str, uuid.UUID]:
    rows = await session.execute(
        text('SELECT item_key, cluster_uid FROM cluster_memberships WHERE partition_id = :p'),
        {'p': partition},
    )
    return dict(rows.all())


async def _run(session, project, keys, vectors, parent=None, **kw):
    return await run_partition(
        session,
        project=project,
        target='item',
        level='class',
        space=SPACE,
        method='ivf',
        params={'n_clusters': kw.pop('k', 8), 'nredo': 10},
        keys=keys,
        vectors=vectors,
        parent_partition_id=parent,
        **kw,
    )


async def test_permuted_order_keeps_every_cluster_uid_and_ari_is_one(pg_env, project) -> None:
    keys, vectors, truth = _blobs()
    perm = np.random.default_rng(11).permutation(len(keys))
    async with async_session_factory()() as session, project_tx(session, project):
        first = await _run(session, project, keys, vectors)
    async with async_session_factory()() as session, project_tx(session, project):
        second = await _run(
            session,
            project,
            [keys[i] for i in perm],
            vectors[perm],
            parent=first.partition_id,
        )
        a = await _assignment(session, first.partition_id)
        b = await _assignment(session, second.partition_id)
    assert a == b  # every cluster_uid survives, member for member
    labels_a = [str(a[k]) for k in keys]
    labels_b = [str(b[k]) for k in keys]
    assert adjusted_rand_score(labels_a, labels_b) == 1.0
    assert adjusted_rand_score(truth, labels_a) == 1.0  # the fixture is well separated
    assert second.stability == 1.0
    assert second.diff['moved_items'] == 0


async def test_a_new_cluster_gets_a_new_uid_and_old_ones_stay(pg_env, project) -> None:
    keys, vectors, _ = _blobs()
    async with async_session_factory()() as session, project_tx(session, project):
        first = await _run(session, project, keys[:-60], vectors[:-60], k=7)
        second = await _run(session, project, keys, vectors, parent=first.partition_id)
        a = await _assignment(session, first.partition_id)
        b = await _assignment(session, second.partition_id)
    assert {b[k] for k in a} >= set(a.values())
    assert len(set(b.values())) == len(set(a.values())) + 1
    assert second.diff['new'] == 1


async def test_anchored_members_are_pinned_and_counted(pg_env, pg, project) -> None:
    keys, vectors, _ = _blobs()
    async with async_session_factory()() as session, project_tx(session, project):
        first = await _run(session, project, keys, vectors)
        a = await _assignment(session, first.partition_id)
    import psycopg

    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute(
            "UPDATE cluster_identities SET anchored_at = now(), anchored_by = 'test' "
            'WHERE cluster_uid = ANY(%s)',
            (list(set(a.values())),),
        )
    # a run with fewer clusters would merge anchored clusters; the pin keeps them apart
    async with async_session_factory()() as session, project_tx(session, project):
        second = await run_partition(
            session,
            project=project,
            target='item',
            level='class',
            space=SPACE,
            method='ivf',
            params={'n_clusters': 4, 'nredo': 10},
            keys=keys,
            vectors=vectors,
            parent_partition_id=first.partition_id,
        )
        b = await _assignment(session, second.partition_id)
    assert b == a  # every anchored identity keeps all of its members
    assert second.anchored_moves > 0


def test_ivf_fit_matches_the_existing_kmeans_on_a_thousand_items() -> None:
    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

    _, vectors, _ = _blobs(n_per=125, n_blobs=8)
    x = unit_rows(vectors)
    ours = fit_pool('ivf', x, {'n_clusters': 8}, seed=42).labels
    store = IVFCentroidStore(store_dir='/tmp/op-test-ivf-unused')
    centroids, _ = store.train(x, n_clusters=8, niter=50, nredo=1, use_gpu=False)
    import faiss

    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    theirs = index.search(x, 1)[1][:, 0]
    assert adjusted_rand_score(theirs, ours) == 1.0
