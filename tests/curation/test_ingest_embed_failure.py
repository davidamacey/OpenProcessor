"""An embedding failure at ingest is marked on the item and counted, never silent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.test_ingest_service import FakePEEncoder, _jpeg_bytes, _make_service
from src.routers.curation.ingest import _batch_response


if TYPE_CHECKING:
    import numpy as np


class _FailingEncoder(FakePEEncoder):
    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        raise RuntimeError('triton down')


@pytest.mark.asyncio
async def test_encoder_failure_marks_items_failed_and_counts_them() -> None:
    svc, os_fake, _ = _make_service()
    svc.pe_encoder = _FailingEncoder()
    result = await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')

    assert result.status == 'success'
    assert (result.n_crops, result.n_embedded, result.n_not_embedded) == (1, 0, 1)
    [doc] = list(os_fake.items.values())
    assert doc['embedding_state'] == 'failed'
    assert 'pe_embedding' not in doc


@pytest.mark.asyncio
async def test_success_marks_items_embedded() -> None:
    svc, os_fake, _ = _make_service()
    result = await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')

    assert (result.n_embedded, result.n_not_embedded) == (1, 0)
    [doc] = list(os_fake.items.values())
    assert doc['embedding_state'] == 'embedded'
    assert 'pe_embedding' in doc


@pytest.mark.asyncio
async def test_batch_summary_and_wire_sum_the_counters() -> None:
    svc, _, _ = _make_service()
    svc.pe_encoder = _FailingEncoder()
    batch = await svc.ingest_batch(
        [_jpeg_bytes(seed=1), _jpeg_bytes(seed=2)], ['/tmp/a.jpg', '/tmp/b.jpg']
    )

    assert (batch.summary.n_embedded, batch.summary.n_not_embedded) == (0, 2)
    wire: Any = _batch_response(batch, [])
    assert (wire.summary.n_embedded, wire.summary.n_not_embedded) == (0, 2)
    assert [r.n_not_embedded for r in wire.results] == [1, 1]
