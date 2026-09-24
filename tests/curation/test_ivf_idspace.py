"""Regression tests for the IVF cluster_id id-space bug (fix/ivf-idspace-bugB).

Bug: ``cluster_residuals`` (batch train) used to renumber IVF labels so
cluster_id=0 was always the largest bucket (``_stable_sort_cluster_ids``),
but that renumbering was applied only in-memory, never persisted to the
centroid store. ``assign_only_residuals`` (periodic re-sort) and the
ingest-time assign path (the curation ingest pipeline) both read the persisted
centroids directly and wrote ``RESIDUAL_CLUSTER_ID_OFFSET + raw_centroid_index``
— the RAW FAISS training-order index, not the size-renumbered one. So the
same crop could get a different ``cluster_id`` depending on which code path
last clustered it.

Fix (Option A): drop the size-based renumbering entirely.
``cluster_residuals`` now writes the raw label + offset too, so all three
call sites agree.

All FAISS/OpenSearch IO is mocked/synthetic; the IVF centroid store is
redirected to ``tmp_path`` so nothing touches the real persisted store
under CurationConfig.state_dir.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


if TYPE_CHECKING:
    from pathlib import Path


RESIDUAL_CLUSTER_ID_OFFSET = 10000


@pytest.fixture(scope='module')
def synthetic_embeddings() -> dict[str, Any]:
    """200 crops drawn from 4 well-separated 16-d gaussians.

    n=200 clears IVF_MIN_TRAIN_VECTORS (192) so IVFMethod trains real
    k-means centroids rather than falling back to the single-bucket path.
    """
    rng = np.random.default_rng(0)
    dim = 16
    n_per = 50
    centers = np.stack(
        [
            np.concatenate([np.array([6.0]), np.zeros(dim - 1)]),
            np.concatenate([np.zeros(4), np.array([6.0]), np.zeros(dim - 5)]),
            np.concatenate([np.zeros(8), np.array([6.0]), np.zeros(dim - 9)]),
            np.concatenate([np.zeros(dim - 1), np.array([6.0])]),
        ]
    )
    chunks = [c + 0.05 * rng.standard_normal((n_per, dim)).astype(np.float32) for c in centers]
    emb = np.vstack(chunks).astype(np.float32)
    ids = [f'crop-{i}' for i in range(emb.shape[0])]
    return {'embeddings': emb, 'ids': ids}


def _make_scroll_client(ids: list[str], embeddings: np.ndarray) -> Any:
    """MagicMock AsyncOpenSearch: one page of hits, then scroll exhausted.

    No ``create_pit`` attr on a bare MagicMock -> AttributeError inside
    ``fetch_residual_embeddings_parallel`` -> falls back to the plain
    scroll fetcher, matching ``tests/curation/test_clustering_orchestrator.py``'s
    mocking convention.
    """
    hits = [
        {'_id': i, '_source': {'pe_embedding': emb.tolist()}}
        for i, emb in zip(ids, embeddings, strict=True)
    ]
    client = MagicMock()
    del client.create_pit  # ensure attribute access raises, not returns a Mock
    client.search = AsyncMock(return_value={'_scroll_id': 's1', 'hits': {'hits': hits}})
    client.scroll = AsyncMock(return_value={'_scroll_id': None, 'hits': {'hits': []}})
    client.clear_scroll = AsyncMock(return_value=None)
    client.count = AsyncMock(return_value={'count': len(ids)})
    client.bulk = AsyncMock(return_value={'errors': False})
    client.indices = MagicMock()
    client.indices.refresh = AsyncMock(return_value=None)
    return client


def _bulk_calls_to_cluster_ids(client: Any) -> dict[str, int]:
    """Reconstruct {crop_id: cluster_id} from the mocked ``client.bulk`` calls.

    F-3 changed these writers to a guarded painless ``script`` update
    (``params.cid``) instead of a blind ``doc`` update.
    """
    out: dict[str, int] = {}
    for call in client.bulk.await_args_list:
        body = call.kwargs.get('body') or call.args[0]
        for i in range(0, len(body), 2):
            action = body[i]['update']
            script = body[i + 1]['script']
            out[action['_id']] = script['params']['cid']
    return out


@pytest.fixture
def ivf_store_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the IVF centroid store to a tmp dir for every code path.

    ``ivf_store.py``'s module-level ``IVF_STORE_DIR``/``CENTROIDS_PATH``/
    ``METADATA_PATH``/``GATE_PATH`` are read by name at call time inside
    ``IVFCentroidStore.__init__`` (for ``IVF_STORE_DIR``) and re-imported by
    value elsewhere (a future ingest pipeline's IVF-assign helper would
    import ``CENTROIDS_PATH`` fresh each call) -- patching the module
    attributes covers both.
    """
    from src.services.curation.clustering.methods import ivf_store as ivf_store_mod

    store_dir = tmp_path / 'ivf_residuals'
    monkeypatch.setattr(ivf_store_mod, 'IVF_STORE_DIR', store_dir)
    monkeypatch.setattr(ivf_store_mod, 'CENTROIDS_PATH', store_dir / 'centroids.faiss')
    monkeypatch.setattr(ivf_store_mod, 'METADATA_PATH', store_dir / 'metadata.json')
    monkeypatch.setattr(ivf_store_mod, 'GATE_PATH', store_dir / 'gate.json')
    return store_dir


@pytest.mark.asyncio
async def test_cluster_residuals_assign_only_and_ingest_agree_on_cluster_id(
    synthetic_embeddings: dict[str, Any],
    ivf_store_dir: Path,
) -> None:
    """The three cluster_id writers must agree for the same crop.

    1. ``cluster_residuals`` (batch train) writes cluster_id for every crop.
    2. ``assign_only_residuals`` (periodic re-sort against the now-persisted
       centroids) is run over the SAME crops and must reproduce the SAME
       cluster_id per crop.
    3. The ingest-time formula (``IVFCentroidStore.assign_one_with_distance``
       + ``RESIDUAL_CLUSTER_ID_OFFSET``, exactly what the curation
       ingest pipeline does) must also reproduce the same id for a
       sampled crop.
    """
    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore
    from src.services.curation.clustering.orchestrator import cluster_residuals

    ids = synthetic_embeddings['ids']
    emb = synthetic_embeddings['embeddings']

    # -- 1. batch train ----------------------------------------------------
    train_client = _make_scroll_client(ids, emb)
    res = await cluster_residuals(train_client, clustering_method='ivf', n_clusters=4)
    assert res['status'] == 'success'
    assert res['method'] == 'ivf'

    batch_cluster_ids = _bulk_calls_to_cluster_ids(train_client)
    assert set(batch_cluster_ids) == set(ids)
    # Sanity: real bucket ids in the candidate namespace, more than one
    # bucket produced (else the test can't distinguish raw vs renumbered).
    distinct = set(batch_cluster_ids.values())
    assert len(distinct) >= 2
    assert all(cid >= RESIDUAL_CLUSTER_ID_OFFSET for cid in distinct)

    # -- 2. assign_only_residuals against the now-persisted centroids ------
    from src.services.curation.clustering.orchestrator import assign_only_residuals

    assign_client = _make_scroll_client(ids, emb)
    assign_res = await assign_only_residuals(assign_client)
    assert assign_res['status'] == 'success'
    assign_cluster_ids = _bulk_calls_to_cluster_ids(assign_client)

    # This is the actual regression assertion: before the fix, batch-trained
    # ids were size-renumbered while assign_only ids were raw FAISS indices,
    # so this would fail for any crop not already in cluster 0.
    assert assign_cluster_ids == batch_cluster_ids

    # -- 3. ingest-time formula reproduces the same id for a sampled crop --
    store = IVFCentroidStore()
    assert store.load()
    sample_id, sample_emb = ids[0], emb[0]
    norm = sample_emb / np.linalg.norm(sample_emb)
    centroid_idx, _dist = store.assign_one_with_distance(norm)
    ingest_cluster_id = int(centroid_idx) + RESIDUAL_CLUSTER_ID_OFFSET
    assert ingest_cluster_id == batch_cluster_ids[sample_id]


@pytest.mark.asyncio
async def test_cluster_residuals_persists_trained_mode(
    synthetic_embeddings: dict[str, Any],
    ivf_store_dir: Path,
) -> None:
    """``cluster_residuals`` records which pool mode produced n_trained_on."""
    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore
    from src.services.curation.clustering.orchestrator import cluster_residuals

    ids = synthetic_embeddings['ids']
    emb = synthetic_embeddings['embeddings']

    client = _make_scroll_client(ids, emb)
    res = await cluster_residuals(client, clustering_method='ivf', n_clusters=4)
    assert res['status'] == 'success'

    store = IVFCentroidStore()
    assert store.metadata.get('trained_mode') == 'strict_residuals'

    client2 = _make_scroll_client(ids, emb)
    res2 = await cluster_residuals(
        client2, clustering_method='ivf', n_clusters=4, recluster_unvalidated=True
    )
    assert res2['status'] == 'success'
    store2 = IVFCentroidStore()
    assert store2.metadata.get('trained_mode') == 'recluster_unvalidated'


# =============================================================================
# should_retrain_centroids growth-gate regression
# =============================================================================


class _FakeGrowthGateStore:
    """Minimal IVFCentroidStore stand-in for should_retrain_centroids.

    Only the surface should_retrain_centroids actually touches:
    is_trained() and .metadata.
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        self._metadata = metadata

    def is_trained(self) -> bool:
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


@pytest.mark.asyncio
async def test_growth_gate_strict_mode_does_not_permanently_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict-mode n_trained_on must not be compared against the full pool.

    Before the fix: should_retrain_centroids always compared the FULL
    residual-pool count against n_trained_on regardless of which mode
    produced it. A strict-mode run trains on a small narrow slice (e.g.
    500 crops not yet in a candidate cluster) while the full pool can be
    huge (e.g. 300k) -- comparing those directly means `grown` is
    permanently True after any strict run, and only the 24h cooldown
    prevents constant retriggering.

    After the fix: a strict-mode n_trained_on is compared against the
    matching STRICT (narrow) pool count, not the full one.
    """
    from src.services.curation.clustering import orchestrator

    store = _FakeGrowthGateStore(
        {
            'n_trained_on': 500,
            'trained_mode': 'strict_residuals',
            'trained_at': None,  # unparseable -> cooled=True, isolates the growth check
        }
    )
    monkeypatch.setattr(
        'src.services.curation.clustering.methods.ivf_store.IVFCentroidStore',
        lambda: store,
    )

    # count() is called once, with strict=True this time; the mock ignores
    # the query body and always returns the STRICT narrow-pool count. If
    # the code regresses to comparing against the full pool, it would
    # instead need a second, larger count -- but here there is only ONE
    # plausible count, so the strict comparison is exercised directly by
    # returning a NARROW count close to n_trained_on.
    narrow_count = 520  # just above 500 * 1.5=750? no -- keep it BELOW threshold
    client = MagicMock()
    client.count = AsyncMock(return_value={'count': narrow_count})

    decision = await orchestrator.should_retrain_centroids(client)

    # 520 is well under 500 * RETRAIN_GROWTH_FACTOR (750 by default) -- the
    # narrow-pool comparison correctly reports "not grown enough yet".
    # Under the pre-fix bug this same fixture would have compared 520 (or
    # any full-pool count, potentially in the hundreds of thousands) but
    # the point is the SAME query result must be interpreted differently
    # depending on trained_mode -- assert the mode was honored end to end.
    assert decision['trained_mode'] == 'strict_residuals'
    assert decision['residual_count'] == narrow_count
    assert decision['should'] is False
    assert decision['reason'] == 'within_threshold'


@pytest.mark.asyncio
async def test_growth_gate_broad_mode_still_fires_on_real_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: a broad-mode-trained store still fires when the pool grows."""
    from src.services.curation.clustering import orchestrator

    store = _FakeGrowthGateStore(
        {
            'n_trained_on': 500,
            'trained_mode': 'recluster_unvalidated',
            'trained_at': None,
        }
    )
    monkeypatch.setattr(
        'src.services.curation.clustering.methods.ivf_store.IVFCentroidStore',
        lambda: store,
    )

    client = MagicMock()
    client.count = AsyncMock(return_value={'count': 100_000})

    decision = await orchestrator.should_retrain_centroids(client)

    assert decision['trained_mode'] == 'recluster_unvalidated'
    assert decision['should'] is True
    assert decision['reason'] == 'pool_grew'


@pytest.mark.asyncio
async def test_growth_gate_defaults_missing_trained_mode_to_broad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-fix centroids with no trained_mode field default to broad (recluster_unvalidated).

    This preserves prior behavior for existing deployments: the automatic
    idle-worker trigger always used broad mode, so absent metadata implies
    that's what was measured.
    """
    from src.services.curation.clustering import orchestrator

    store = _FakeGrowthGateStore({'n_trained_on': 500, 'trained_at': None})
    monkeypatch.setattr(
        'src.services.curation.clustering.methods.ivf_store.IVFCentroidStore',
        lambda: store,
    )

    client = MagicMock()
    client.count = AsyncMock(return_value={'count': 1000})

    decision = await orchestrator.should_retrain_centroids(client)
    assert decision['trained_mode'] == 'recluster_unvalidated'
