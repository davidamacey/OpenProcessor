"""Fetch and write-back changes must not change what the clustering methods produce.

For one fixed embedding set the three registered methods (and the AHC refinement and the
IVF assign-only re-sort) are run end to end through ``cluster_residuals`` against an
in-memory index, and every stored ``cluster_id`` and ``cluster_distance`` is compared,
exactly, with a golden file written from the code before the plumbing was touched.

Regenerate only on purpose, after a change meant to alter the clustering itself:
``OP_REGEN_CLUSTER_GOLDEN=1 pytest tests/curation/test_cluster_plumbing_golden.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.services.curation.clustering.methods import available_methods
from tests.curation._fake_items_pit import FakeItemsClient


GOLDEN_DIR = Path(__file__).parent / 'golden'
EMBEDDINGS = GOLDEN_DIR / 'cluster_embeddings.npz'
GOLDEN = GOLDEN_DIR / 'cluster_plumbing_golden.json'
REGEN = bool(os.environ.get('OP_REGEN_CLUSTER_GOLDEN'))


def _make_embeddings() -> tuple[list[str], np.ndarray]:
    """Nine gaussian blobs of unequal size plus scattered points, fixed seed."""
    rng = np.random.default_rng(20261012)
    dim = 48
    centers = rng.standard_normal((9, dim))
    sizes = [70, 60, 55, 50, 45, 40, 30, 25, 15]
    blobs = [c + 0.35 * rng.standard_normal((n, dim)) for c, n in zip(centers, sizes, strict=True)]
    blobs.append(rng.standard_normal((20, dim)))
    vectors = np.vstack(blobs).astype(np.float32)
    order = rng.permutation(len(vectors))
    ids = [f'doc-{i:04d}' for i in range(len(vectors))]
    return ids, vectors[order]


def _load_embeddings() -> tuple[list[str], np.ndarray]:
    if REGEN and not EMBEDDINGS.exists():
        ids, vectors = _make_embeddings()
        np.savez_compressed(EMBEDDINGS, ids=np.array(ids), vectors=vectors)
    z = np.load(EMBEDDINGS)
    return [str(i) for i in z['ids']], z['vectors']


@pytest.fixture
def ivf_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.clustering.methods import ivf_store as ivf_store_mod

    class _Config:
        project_state_dir = tmp_path

    monkeypatch.setattr(ivf_store_mod, 'get_curation_config', lambda: _Config())


def _stored(client: FakeItemsClient) -> dict[str, list[Any]]:
    return {i: [cid, dist] for i, (cid, dist) in sorted(client.written.items())}


async def _refine_largest(client: FakeItemsClient, vectors: np.ndarray) -> dict[str, str]:
    """AHC refinement of the largest cluster the AHC pass produced."""
    from src.services.curation.clustering.refine import refine_members

    by_cluster: dict[int, list[str]] = {}
    for doc_id, (cid, _) in client.written.items():
        by_cluster.setdefault(cid, []).append(doc_id)
    cid, members = max(sorted(by_cluster.items()), key=lambda kv: len(kv[1]))
    row = {doc_id: vectors[client.ids.index(doc_id)] for doc_id in members}
    written: dict[str, str] = {}

    async def count() -> int:
        return len(members)

    async def fetch() -> list[dict[str, Any]]:
        return [
            {'_id': d, 'embedding': row[d].tolist(), 'class_name': None} for d in sorted(members)
        ]

    async def write(updates: list[tuple[Any, str]]) -> int:
        written.update({str(k): v for k, v in updates})
        return len(updates)

    await refine_members(
        cid,
        count_members=count,
        fetch_members=fetch,
        write_subids=write,
        unit='items',
        distance_threshold=0.15,
    )
    return dict(sorted(written.items()))


async def _run_everything() -> dict[str, Any]:
    from src.services.curation.clustering.orchestrator import (
        assign_only_residuals,
        cluster_residuals,
    )

    ids, vectors = _load_embeddings()
    out: dict[str, Any] = {}
    for method in available_methods():
        client = FakeItemsClient(ids, vectors)
        summary = await cluster_residuals(client, clustering_method=method)
        assert summary['status'] == 'success'
        out[method] = {'n_clusters': summary['n_clusters'], 'stored': _stored(client)}
        if method == 'ahc':
            out['ahc_refine'] = await _refine_largest(client, vectors)
        if method == 'ivf':
            client.written.clear()
            await assign_only_residuals(client)
            out['ivf_assign_only'] = _stored(client)
    return out


@pytest.mark.asyncio
async def test_every_method_stores_exactly_what_the_golden_file_says(ivf_state: None) -> None:
    got = json.loads(json.dumps(await _run_everything()))
    if REGEN:
        GOLDEN.write_text(json.dumps(got, indent=0, sort_keys=True) + '\n')
    want = json.loads(GOLDEN.read_text())
    assert sorted(got) == sorted(want)
    for key in want:
        assert got[key] == want[key], key


def test_the_golden_run_covers_every_registered_method() -> None:
    assert set(available_methods()) <= set(json.loads(GOLDEN.read_text()))
