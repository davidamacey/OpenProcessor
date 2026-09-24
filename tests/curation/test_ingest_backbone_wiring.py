"""Triton-pool plumbing behind ingest's backbone-embedding path (N3).

The pipeline behavior itself is covered in ``test_ingest_service.py``
(``TestBackboneEmbedding``); this pins the real pool's output-name probe
that decides whether the secondary detector's feature map is requested.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.clients.triton_pool import AsyncTritonPool


class _FakeGrpcClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def get_model_metadata(self, model_name: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({'model_name': model_name, **kwargs})
        return {
            'name': model_name,
            'outputs': [
                {'name': 'output0', 'datatype': 'FP32', 'shape': ['-1', '100800', '85']},
                {'name': 'sppf_feat', 'datatype': 'FP32', 'shape': ['-1', '768', '40', '40']},
            ],
        }


class TestPoolOutputNames:
    @pytest.mark.asyncio
    async def test_reads_output_names_from_json_metadata(self) -> None:
        pool = AsyncTritonPool(pool_size=1)
        client = _FakeGrpcClient()
        pool._clients = [client]  # type: ignore[list-item]
        pool._lock = asyncio.Lock()
        pool._initialized = True

        names = await pool.get_model_output_names('dual_head_detector')

        assert names == ['output0', 'sppf_feat']
        assert client.calls[0]['model_name'] == 'dual_head_detector'
        assert client.calls[0]['as_json'] is True

    @pytest.mark.asyncio
    async def test_uninitialized_pool_raises(self) -> None:
        with pytest.raises(RuntimeError, match='not initialized'):
            await AsyncTritonPool(pool_size=1).get_model_output_names('m')
