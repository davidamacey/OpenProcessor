"""CM-4: the IVF single-bucket branch must persist a centroid.

Before the fix, ``IVFMethod.fit_predict`` on a pool smaller than
``IVF_MIN_TRAIN_VECTORS`` returned an all-zero label array without ever
writing to :class:`IVFCentroidStore`. That leaves ``is_trained()`` False
forever on a small/fresh install, so
:func:`should_retrain_centroids` reports ``no_centroids_yet`` on every poll
(every 1800s) instead of applying its growth/cooldown gates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.services.curation.clustering.backend import BackendInfo
from src.services.curation.clustering.methods.ivf import IVF_MIN_TRAIN_VECTORS, IVFMethod


def _cpu_backend() -> BackendInfo:
    return BackendInfo(name='cpu', detail='forced cpu for test', free_vram_mb=None, build_algo=None)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ivf_store_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.services.curation.clustering.methods import ivf_store as ivf_store_mod

    store_dir = tmp_path / 'ivf_residuals'
    monkeypatch.setattr(ivf_store_mod, 'IVF_STORE_DIR', store_dir)
    monkeypatch.setattr(ivf_store_mod, 'CENTROIDS_PATH', store_dir / 'centroids.faiss')
    monkeypatch.setattr(ivf_store_mod, 'METADATA_PATH', store_dir / 'metadata.json')
    monkeypatch.setattr(ivf_store_mod, 'GATE_PATH', store_dir / 'gate.json')
    return store_dir


def _small_pool(n: int = 50, dim: int = 16) -> np.ndarray:
    assert n < IVF_MIN_TRAIN_VECTORS
    rng = np.random.default_rng(1)
    return rng.standard_normal((n, dim)).astype(np.float32)


@pytest.mark.asyncio
async def test_single_bucket_persists_one_trained_centroid(ivf_store_dir: Path) -> None:
    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

    embeddings = _small_pool()
    method = IVFMethod()
    result = await method.fit_predict(embeddings, backend_info=_cpu_backend(), progress=None)

    assert (result.labels == 0).all()
    assert result.extra['reason'] == 'too_few_vectors_for_training'

    store = IVFCentroidStore()
    assert store.is_trained()
    assert store.n_clusters == 1
    assert store.metadata['n_trained_on'] == embeddings.shape[0]
    assert store.metadata['trained_mode'] == 'single_bucket'

    assert result.distances is not None
    assert result.distances.shape[0] == embeddings.shape[0]
    assert np.all(result.distances >= 0.0)
    assert np.all(result.distances <= 2.0)


@pytest.mark.asyncio
async def test_single_bucket_respects_persist_false(ivf_store_dir: Path) -> None:
    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

    embeddings = _small_pool()
    method = IVFMethod(persist=False)
    await method.fit_predict(embeddings, backend_info=_cpu_backend(), progress=None)

    store = IVFCentroidStore()
    assert not store.is_trained()


@pytest.mark.asyncio
async def test_should_retrain_centroids_sees_trained_store_after_single_bucket(
    ivf_store_dir: Path,
) -> None:
    from src.services.curation.clustering.orchestrator import should_retrain_centroids

    embeddings = _small_pool()
    method = IVFMethod()
    await method.fit_predict(embeddings, backend_info=_cpu_backend(), progress=None)

    client = AsyncMock()
    client.count = AsyncMock(return_value={'count': embeddings.shape[0]})
    decision = await should_retrain_centroids(client)
    assert decision['reason'] != 'no_centroids_yet'
