"""train/rebalance route responses: error results carry zeros, success maps the result fields."""

from __future__ import annotations

from typing import Any

import pytest

from src.routers.clusters import rebalance_clusters, train_clusters


class _Search:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    async def train_clusters(self, **_kw: Any) -> dict[str, Any]:
        return self.result

    async def rebalance_clusters(self, **_kw: Any) -> dict[str, Any]:
        return self.result


_OK = {
    'index_name': 'global',
    'n_vectors': 10,
    'n_clusters': 4,
    'avg_cluster_size': 2.5,
    'empty_clusters': 1,
    'documents_updated': 9,
    'training_time_s': 1.5,
}


@pytest.mark.asyncio
async def test_train_success_maps_every_result_field() -> None:
    out = await train_clusters(_Search(_OK), 'global', None, None)  # type: ignore[arg-type]
    assert out.status == 'success'
    assert out.model_dump(exclude_none=True) == {
        'status': 'success',
        **_OK,
    }


@pytest.mark.asyncio
async def test_rebalance_error_result_is_zeros_with_the_error() -> None:
    out = await rebalance_clusters(
        _Search({'status': 'error', 'error': 'too few vectors'}),  # type: ignore[arg-type]
        'global',
    )
    assert out.status == 'error'
    assert out.error == 'too few vectors'
    assert (out.n_vectors, out.n_clusters, out.documents_updated, out.training_time_s) == (
        0,
        0,
        0,
        0,
    )
