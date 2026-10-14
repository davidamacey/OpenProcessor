"""Pure logic of the cluster pass benchmark (scripts/bench/cluster_bench.py)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from scripts.bench import cluster_bench as cb


def _summary(n: int, **over: object) -> dict[str, object]:
    return {'status': 'success', 'n_residuals': n, 'n_clusters': 7, **over}


def test_timed_pass_records_wall_method_and_counts() -> None:
    ticks = iter([10.0, 12.5])

    async def fake_pass(method: str) -> dict[str, object]:
        assert method == 'ivf'
        return _summary(2000)

    row = asyncio.run(
        cb.timed_pass(fake_pass, method='ivf', size=2000, rep=1, clock=lambda: next(ticks))
    )
    assert row['wall_s'] == 2.5
    assert (row['method'], row['size'], row['rep']) == ('ivf', 2000, 1)
    assert (row['n_residuals'], row['n_clusters'], row['status']) == (2000, 7, 'success')


@pytest.mark.parametrize(
    'summary',
    [_summary(1999), _summary(2000, status='too_few_residuals'), {'status': 'success'}],
)
def test_timed_pass_refuses_a_pass_that_did_not_cluster_the_whole_pool(
    summary: dict[str, object],
) -> None:
    async def fake_pass(method: str) -> dict[str, object]:
        return summary

    with pytest.raises(cb.ClusterBenchError):
        asyncio.run(cb.timed_pass(fake_pass, method='ivf', size=2000, rep=0))


def test_parse_sizes_rejects_nonpositive_and_sorts() -> None:
    assert cb.parse_sizes('10000, 2000') == [2000, 10000]
    with pytest.raises(ValueError, match='positive'):
        cb.parse_sizes('0')


def test_prefix_is_nested_unit_and_refuses_oversize() -> None:
    rng = np.random.default_rng(0)
    pool = rng.standard_normal((50, 8)).astype(np.float32) * 3
    small, large = cb.take_prefix(pool, 10), cb.take_prefix(pool, 20)
    assert np.allclose(np.linalg.norm(large, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(small, large[:10])
    with pytest.raises(ValueError, match='only 50'):
        cb.take_prefix(pool, 51)
