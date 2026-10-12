"""The geometry pass must store exactly what it stored before its fetch and writes were rewritten.

One fixed set (class clusters, candidate clusters, an excluded item, a cluster whose
centroid vanishes, items with stale or current geometry) goes through
``write_cluster_geometry`` against an in-memory index, once with the scroll fetch and once
with the point-in-time one; every stored field and the summary must equal the golden
file, which was written from the code before the change.

Distances come from a matrix product whose last bits depend on the CPU's BLAS kernel, so
the golden file is compared exactly for every integer field and to 1e-5 for distances; the
exact check is against a reference of the old algorithm (one cluster at a time, in document
order) computed in the same process.

Regenerate only on purpose: ``OP_REGEN_CLUSTER_GOLDEN=1 pytest <this file>``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.services.curation.clustering.cluster_geometry import write_cluster_geometry
from tests.curation._fake_geometry import GeometryFake


GOLDEN_DIR = Path(__file__).parent / 'golden'
GOLDEN = GOLDEN_DIR / 'cluster_geometry_golden.json'
REGEN = bool(os.environ.get('OP_REGEN_CLUSTER_GOLDEN'))


def _docs() -> dict[str, dict[str, Any]]:
    vectors = np.load(GOLDEN_DIR / 'cluster_embeddings.npz')['vectors']
    clusters = [3] * 70 + [5] * 60 + [7] * 40 + [10001] * 60 + [10002] * 50 + [10003] * 40
    clusters += [10004] * 25 + [9] * 2 + [-1] * 3
    docs: dict[str, dict[str, Any]] = {}
    for i, cid in enumerate(clusters):
        vec = vectors[i].tolist()
        doc: dict[str, Any] = {'crop_id': f'd{i:04d}', 'cluster_id': cid, 'pe_embedding': vec}
        if cid >= 0 and i % 7 == 0:
            doc['cluster_distance'] = 0.5
            doc['cluster_distance_cluster_id'] = cid if i % 14 == 0 else 3
        if cid == 9:
            doc['pe_embedding'] = (
                vectors[i].tolist() if i == len(clusters) - 5 else (-vectors[i - 1]).tolist()
            )
        if i % 31 == 0:
            doc['class_excluded'] = True
        if cid == -1:
            doc.pop('cluster_id')
        docs[doc['crop_id']] = doc
    return docs


async def _run(pit: bool) -> dict[str, Any]:
    client = GeometryFake(_docs(), pit=pit)
    summary = await write_cluster_geometry(client, index='items')
    return {'summary': summary, 'docs': client.docs}


@pytest.mark.asyncio
@pytest.mark.parametrize('pit', [False, True])
async def test_geometry_is_stored_exactly_as_the_golden_file_says(pit: bool) -> None:
    got = json.loads(json.dumps(await _run(pit)))
    if REGEN and not pit:
        GOLDEN.write_text(json.dumps(got, sort_keys=True) + '\n')
    want = json.loads(GOLDEN.read_text())
    # n_unchanged is new; every key the golden file has must be equal.
    assert {k: got['summary'][k] for k in want['summary']} == want['summary']
    assert got['docs'].keys() == want['docs'].keys()
    for doc_id, doc in want['docs'].items():
        for field, value in doc.items():
            have = got['docs'][doc_id][field]
            if field == 'cluster_distance' and isinstance(value, float):
                assert have == pytest.approx(value, abs=1e-5), (doc_id, field)
            else:
                assert have == value, (doc_id, field)
        assert got['docs'][doc_id].keys() == doc.keys()


def _reference() -> dict[str, dict[str, Any]]:
    """The old algorithm, one cluster at a time in document order, on the fixture."""
    from src.services.curation.clustering.cluster_geometry import _normalize, unit_centroid

    docs = _docs()
    usable = {
        i: d
        for i, d in docs.items()
        if d.get('cluster_id', -1) >= 0 and not d.get('class_excluded') and d['pe_embedding']
    }
    by_cluster: dict[int, list[str]] = {}
    for i, d in usable.items():
        by_cluster.setdefault(d['cluster_id'], []).append(i)
    matrices = {
        c: np.asarray([usable[i]['pe_embedding'] for i in m], np.float32)
        for c, m in by_cluster.items()
    }
    cents = {c: unit_centroid(x) for c, x in sorted(matrices.items())}
    cids = [c for c, v in cents.items() if v is not None]
    stack = np.stack([cents[c] for c in cids])
    for c in cids:
        sims = _normalize(matrices[c]) @ stack.T
        own = np.clip(1.0 - sims[:, cids.index(c)], 0.0, 2.0)
        for i, dist, k in zip(
            by_cluster[c], own.tolist(), np.argmax(sims, axis=1).tolist(), strict=True
        ):
            docs[i]['cluster_distance_cluster_id'] = c
            docs[i]['cluster_nearest_id'] = cids[k]
            if c < 10000:
                docs[i]['cluster_distance'] = float(dist)
    return docs


@pytest.mark.asyncio
@pytest.mark.parametrize('pit', [False, True])
async def test_geometry_equals_the_old_algorithm_exactly(pit: bool) -> None:
    got = json.loads(json.dumps(await _run(pit)))['docs']
    assert got == json.loads(json.dumps(_reference()))


@pytest.mark.asyncio
@pytest.mark.parametrize('pit', [False, True])
async def test_a_second_pass_writes_nothing_and_changes_nothing(pit: bool) -> None:
    client = GeometryFake(_docs(), pit=pit)
    first = await write_cluster_geometry(client, index='items')
    stored = json.loads(json.dumps(client.docs))
    written = client.docs_written
    second = await write_cluster_geometry(client, index='items')
    assert first['n_unchanged'] == 0
    assert second['n_unchanged'] == second['n_items']
    assert client.docs_written == written, 'unchanged items were written again'
    assert json.loads(json.dumps(client.docs)) == stored


@pytest.mark.asyncio
async def test_geometry_writes_are_concurrent(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.clustering import cluster_geometry

    monkeypatch.setattr(cluster_geometry, 'BULK_CHUNK', 20)
    client = GeometryFake(_docs(), pit=True)
    client.bulk_delay_s = 0.01
    await write_cluster_geometry(client, index='items')
    assert client.max_in_flight >= 2
