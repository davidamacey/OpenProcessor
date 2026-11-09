"""The ingest primary reads the fused ``[300, 6]`` detector tensor: one ``output0`` request, boxes
in letterbox pixels undone to frame pixels, zero padding dropped, any chunking of a batch."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from PIL import Image

from curation.ingest_fakes import PRIMARY_NET, FakeInferResult, FakeTritonPool
from src.config import DetectionProfile
from src.services.curation.ingest_detect import WholeImageDetector


class _Registry:
    def get(self, _class_id: int) -> Any:
        return None


class _RecordingPool(FakeTritonPool):
    def __init__(self, detections: list[tuple[float, float, float, float, float, int]]) -> None:
        super().__init__(detections)
        self.requested: list[list[str]] = []

    async def infer(self, model_name: str, inputs: list, outputs: list) -> FakeInferResult:
        self.requested.append([o.name() for o in outputs])
        return await super().infer(model_name, inputs, outputs)


def _detector(pool: FakeTritonPool, *, batch_limit: int = 16) -> WholeImageDetector:
    profile = DetectionProfile(
        name='item', detector_model='det', input_size=PRIMARY_NET, batch_limit=batch_limit
    )
    return WholeImageDetector(triton_pool=pool, registry=_Registry(), profile=profile)


# A 400x300 frame in a 320 letterbox: scale 0.8, 320x240 content, 40 px of vertical padding.
_IMG = Image.new('RGB', (400, 300), (10, 20, 30))


def test_the_primary_requests_only_the_fused_tensor_and_undoes_the_letterbox() -> None:
    pool = _RecordingPool([(0.1, 0.225, 0.6, 0.725, 0.9, 3)])

    (item,) = asyncio.run(_detector(pool).run_primary(_IMG))

    assert pool.requested == [['output0']]
    # letterbox pixels (32, 72, 192, 232) -> ((x - 0) / 0.8, (y - 40) / 0.8)
    assert item.bbox_pixel == pytest.approx((40.0, 40.0, 240.0, 240.0))
    assert item.score == pytest.approx(0.9)


def test_zero_padding_rows_are_not_detections() -> None:
    pool = _RecordingPool([])

    assert asyncio.run(_detector(pool).run_primary(_IMG)) == []


def test_a_batch_split_across_triton_calls_decodes_every_image() -> None:
    pool = _RecordingPool([(0.1, 0.225, 0.6, 0.725, 0.9, 0), (0.2, 0.3, 0.4, 0.5, 0.8, 0)])

    per_image = asyncio.run(_detector(pool, batch_limit=2).run_primary_batch([_IMG] * 5))

    assert pool.batch_sizes == [2, 2, 1]
    assert [len(items) for items in per_image] == [2] * 5
