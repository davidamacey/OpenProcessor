"""Exact range-search grouping against the dense reference (#354)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.services.face_identity.person_grouping import similarity_components


REPO = Path(__file__).resolve().parent.parent


def _dense_components(x: np.ndarray, t: float) -> list[list[int]]:
    """The old dense method, kept only here as the reference."""
    n = len(x)
    sim = x @ x.T
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= t:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _fixture() -> np.ndarray:
    rng = np.random.default_rng(354)
    centers = rng.standard_normal((40, 512)).astype(np.float32)
    rows = []
    for k in range(550):
        c = k % 40
        spread = (0.02, 0.05, 0.08, 0.12)[c % 4]
        rows.append(centers[c] + spread * rng.standard_normal(512).astype(np.float32))
    rows.extend(rng.standard_normal((50, 512)).astype(np.float32))
    x = np.asarray(rows, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_range_search_matches_dense_reference():
    x = _fixture()
    sim = x @ x.T
    np.fill_diagonal(sim, -1.0)
    for t in (0.5, 0.6, 0.7, 0.8):
        assert np.abs(sim - t).min() > 1e-4, 'fixture has a pair too close to the threshold'
    for t in (0.5, 0.6, 0.7, 0.8):
        ref = _dense_components(x, t)
        assert len(ref) < len(x)
        for batch in (7, 64, 4096):
            assert similarity_components(x, t, batch=batch) == ref, (t, batch)


def test_pair_exactly_at_threshold_is_linked():
    t = np.float32(0.6)
    x = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    assert float(x[0] @ x[1]) == float(t)
    assert similarity_components(x, 0.6) == [[0, 1]]
    assert similarity_components(x, 0.61) == [[0], [1]]


def test_empty_input():
    assert similarity_components(np.zeros((0, 8), dtype=np.float32), 0.7) == []


_CHILD = """
import importlib.util, json, time
import numpy as np
# Load the module by path: importing it through its package would pull the whole
# service (torch, ultralytics, ...) into the child and measure that, not the grouping.
spec = importlib.util.spec_from_file_location('person_grouping', 'src/services/face_identity/person_grouping.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
similarity_components = mod.similarity_components
rng = np.random.default_rng(354)
n, d, k = 200000, 64, 50000
c = rng.standard_normal((k, d), dtype=np.float32)
x = c[rng.integers(0, k, n)] + 0.05 * rng.standard_normal((n, d), dtype=np.float32)
x /= np.linalg.norm(x, axis=1, keepdims=True)
def hwm_mb():
    # VmHWM is this process's own peak and resets on exec; ru_maxrss would carry over
    # the forking parent's high-water mark (about 3 GB under CI coverage).
    for line in open('/proc/self/status'):
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('VmHWM missing')
before = hwm_mb()
t0 = time.time()
groups = similarity_components(x, 0.7)
el = time.time() - t0
print(json.dumps({'covered': sum(len(g) for g in groups),
  'unique': len({i for g in groups for i in g}),
  'groups': len(groups), 'elapsed': el, 'rss_before_mb': before,
  'rss_mb': hwm_mb()}))
"""


def test_200k_vectors_fit_without_dense_matrix():
    proc = subprocess.run(
        [sys.executable, '-c', _CHILD],
        cwd=REPO,
        env=os.environ.copy(),  # threads are capped by the code under test, not pinned here
        capture_output=True,
        text=True,
        check=True,
        timeout=380,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    print(out)
    assert out['covered'] == out['unique'] == 200000
    assert out['rss_mb'] < 1024
    assert out['elapsed'] < 300
