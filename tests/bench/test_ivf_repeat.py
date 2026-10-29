"""ivf_repeat: the production IVF fit is repeatable on identical vectors and row-order sensitive."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from scripts.bench import ivf_repeat


if TYPE_CHECKING:
    from pathlib import Path


def _blobs(n: int = 6000, dim: int = 16, k: int = 20) -> np.ndarray:
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(k, dim))
    x = centres[rng.integers(0, k, n)] + 0.3 * rng.normal(size=(n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_same_order_runs_are_identical_and_the_permuted_run_is_reported() -> None:
    out = ivf_repeat.repeat(_blobs(), runs=2, shuffle_seed=3)
    assert out['same_order_identical'] is True
    assert out['same_order_ari'] == [1.0]
    assert len(out['labels'][0]) == 6000
    assert -1.0 <= out['permuted_ari'] <= 1.0
    assert len(out['fit_seconds']) == 2


def test_partition_rows_carry_item_ids(tmp_path: Path) -> None:
    path = tmp_path / 'p.jsonl'
    ivf_repeat.write_partition(path, ['a', 'b'], np.array([4, 4]))
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {'item_id': 'a', 'cluster_id': 4},
        {'item_id': 'b', 'cluster_id': 4},
    ]
