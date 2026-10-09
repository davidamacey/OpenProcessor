"""The "Last clustering" record is written by the clustering code path itself,
so a run started by the cluster-refresh daemon (the synchronous
``/pipeline/auto_label``, which writes no job state) is still reflected."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.services.curation.clustering import embedding_reduce, last_run
from src.services.curation.clustering.orchestrator import cluster_residuals


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        last_run, 'get_curation_config', lambda: SimpleNamespace(autolabel_dir=tmp_path)
    )
    return tmp_path


def _client(embeddings: np.ndarray) -> Any:
    hits = [
        {'_id': f'c{i}', '_source': {'pe_embedding': e.tolist()}} for i, e in enumerate(embeddings)
    ]
    client = MagicMock()
    client.search = AsyncMock(return_value={'_scroll_id': 's1', 'hits': {'hits': hits}})
    client.scroll = AsyncMock(return_value={'_scroll_id': None, 'hits': {'hits': []}})
    client.clear_scroll = AsyncMock(return_value=None)
    client.bulk = AsyncMock(return_value={'errors': False})
    client.indices = MagicMock()
    client.indices.refresh = AsyncMock(return_value=None)
    client.get = AsyncMock(side_effect=Exception('no cached reducer'))
    client.index = AsyncMock(return_value=None)
    return client


def test_nothing_recorded_reads_none(state_dir: Path) -> None:
    assert last_run.read_last_run() is None


def test_record_round_trips_and_ignores_unwritable_dir(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    last_run.record_last_run({'method': 'ahc', 'n_clusters': 3, 'n_residuals': 50, 'n_noise': 1})
    rec = last_run.read_last_run()
    assert rec is not None
    assert (rec['method'], rec['n_clusters'], rec['n_residuals'], rec['n_noise']) == (
        'ahc',
        3,
        50,
        1,
    )
    assert rec['finished_at']

    blocked = state_dir / 'file'
    blocked.write_text('x')
    monkeypatch.setattr(
        last_run, 'get_curation_config', lambda: SimpleNamespace(autolabel_dir=blocked / 'sub')
    )
    last_run.record_last_run({'method': 'ahc'})  # must not raise


@pytest.mark.asyncio
async def test_cluster_residuals_records_the_run(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(0)
    emb = np.vstack(
        [
            np.eye(16, dtype=np.float32)[0] * 5 + 0.05 * rng.standard_normal((20, 16)),
            np.eye(16, dtype=np.float32)[8] * 5 + 0.05 * rng.standard_normal((20, 16)),
        ]
    ).astype(np.float32)

    async def _passthrough(client: Any, embeddings: np.ndarray, *, mode: str) -> Any:
        return MagicMock(), embeddings.astype(np.float32), True

    monkeypatch.setattr(embedding_reduce, 'get_or_fit_reducer', _passthrough)

    assert last_run.read_last_run() is None
    res = await cluster_residuals(_client(emb), clustering_method='ahc')
    assert res['status'] == 'success'

    rec = last_run.read_last_run()
    assert rec is not None
    assert rec['method'] == 'ahc'
    assert rec['n_clusters'] == res['n_clusters']
    assert rec['n_residuals'] == 40
