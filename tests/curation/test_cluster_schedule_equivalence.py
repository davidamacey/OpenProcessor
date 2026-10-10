"""WP-1.5 (#212) equivalence gate: moving the retrain into the background job
changes where it runs, never what it computes.

* The job route and the synchronous route carry the same defaults for every
  parameter that reaches the clustering stage, and the daemon sends none.
* The real IVF method gives identical cluster assignments for a fixed seed
  whether it is awaited inline (old path) or run on a worker thread's own
  event loop (the job runs in another process; a separate loop is the
  offline stand-in), including the sampled-train branch (`default_rng(42)`).
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.services.curation.clustering.backend import BackendInfo
from src.services.curation.clustering.methods.ivf import IVFMethod


if TYPE_CHECKING:
    from pathlib import Path

CLUSTER_PARAMS = (
    'train_clusters',
    'clustering_method',
    'recluster_unvalidated',
    'reassign_only',
    'gate_max_rank',
    'gate_min_blur_ratio',
    'n_clusters',
    'class_id',
    'cluster_id',
    'run_auto_promote',
    'run_vlm',
)


def test_job_route_and_sync_route_share_cluster_defaults() -> None:
    from src.routers.curation.pipeline_public import pipeline_auto_label
    from src.routers.curation.pipeline_start import pipeline_auto_label_start

    sync = inspect.signature(pipeline_auto_label).parameters
    job = inspect.signature(pipeline_auto_label_start).parameters
    for name in CLUSTER_PARAMS:
        assert job[name].default == sync[name].default, name


def test_daemon_sends_no_overrides_to_the_job_route() -> None:
    import httpx

    from scripts.curation import cluster_refresh_daemon as d

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={'job_id': 'j1', 'status': 'queued'})

    async def go() -> str | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await d._start_auto_label_job(c, 'http://api', '/curation', 'alpha')

    assert asyncio.run(go()) == 'j1'
    req = seen[0]
    assert req.url.path == '/curation/projects/alpha/pipeline/auto_label/start'
    assert req.url.query == b''
    assert req.content in (b'', b'{}')


@pytest.fixture
def ivf_store_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.services.curation.clustering.methods import ivf_store as ivf_store_mod

    class _Cfg:
        project_state_dir = tmp_path

    monkeypatch.setattr(ivf_store_mod, 'get_curation_config', lambda: _Cfg())
    return tmp_path


def _pool(n: int = 600, dim: int = 24) -> np.ndarray:
    rng = np.random.default_rng(7)
    centers = rng.standard_normal((6, dim)).astype(np.float32) * 4
    x = centers[rng.integers(0, 6, n)] + rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _cpu() -> BackendInfo:
    return BackendInfo(name='cpu', detail='test', free_vram_mb=None, build_algo=None)


async def _fit(x: np.ndarray, max_train_sample: int) -> np.ndarray:
    result = await IVFMethod(
        n_clusters=8, niter=20, nredo=2, max_train_sample=max_train_sample
    ).fit_predict(x, backend_info=_cpu())
    return np.asarray(result.labels)


@pytest.mark.parametrize('max_train_sample', [50_000, 300])  # full pool, and the seeded subsample
def test_scheduled_path_matches_synchronous_path(
    ivf_store_dir: Path, max_train_sample: int
) -> None:
    x = _pool()

    sync_labels = asyncio.run(_fit(x, max_train_sample))

    def scheduled() -> np.ndarray:
        return asyncio.run(_fit(x.copy(), max_train_sample))

    async def via_thread() -> np.ndarray:
        return await asyncio.to_thread(scheduled)

    scheduled_labels = asyncio.run(via_thread())
    again = asyncio.run(_fit(x, max_train_sample))

    assert len(set(sync_labels.tolist())) > 1
    assert np.array_equal(sync_labels, scheduled_labels)
    assert np.array_equal(sync_labels, again)
