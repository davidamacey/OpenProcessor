"""Detector and PE requests carry uint8 pixels; the engines normalize (#214)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.clients.triton_pool_core import uint8_images_input
from src.services.curation.ingest_detect import WholeImageDetector


class TestUint8ImagesInput:
    def test_datatype_shape_and_bytes(self) -> None:
        pixels = np.random.default_rng(0).integers(0, 256, (2, 3, 8, 8), dtype=np.uint8)
        inp = uint8_images_input(pixels)
        assert inp.name() == 'images'
        assert inp.datatype() == 'UINT8'
        assert inp.shape() == [2, 3, 8, 8]
        assert len(inp._get_content()) == pixels.nbytes  # one byte per pixel, a quarter of FP32

    def test_non_contiguous_view_is_copied_in_order(self) -> None:
        pixels = np.arange(2 * 3 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4, 3).transpose(0, 3, 1, 2)
        inp = uint8_images_input(pixels)
        assert bytes(inp._get_content()) == np.ascontiguousarray(pixels).tobytes()

    @pytest.mark.parametrize('dtype', [np.float32, np.float16, np.int32])
    def test_refuses_anything_but_uint8(self, dtype: Any) -> None:
        with pytest.raises(TypeError, match='uint8'):
            uint8_images_input(np.zeros((1, 3, 4, 4), dtype=dtype))


class _Pool:
    def __init__(self) -> None:
        self.inputs: list[Any] = []

    async def infer(self, _model: str, inputs: list[Any], **_kw: Any) -> Any:
        self.inputs.extend(inputs)
        n = inputs[0].shape()[0]
        outputs = {
            'num_dets': np.zeros((n, 1), dtype=np.int32),
            'det_boxes': np.zeros((n, 300, 4), dtype=np.float32),
            'det_scores': np.zeros((n, 300), dtype=np.float16),
            'det_classes': np.zeros((n, 300), dtype=np.int32),
        }
        return SimpleNamespace(as_numpy=lambda name: outputs[name])


def _detector(pool: _Pool) -> WholeImageDetector:
    profile = SimpleNamespace(
        input_size=64,
        letterbox_fill=(114, 114, 114),
        batch_limit=2,
        detector_model='det',
        confidence_floor=0.5,
    )
    return WholeImageDetector(triton_pool=pool, registry=None, profile=profile)  # type: ignore[arg-type]


def test_run_primary_sends_one_uint8_tensor() -> None:
    pool = _Pool()
    asyncio.run(_detector(pool).run_primary(Image.new('RGB', (100, 50), (10, 20, 30))))
    (inp,) = pool.inputs
    assert (inp.datatype(), inp.shape()) == ('UINT8', [1, 3, 64, 64])


def test_run_primary_batch_chunks_by_batch_limit_in_uint8() -> None:
    pool = _Pool()
    imgs = [Image.new('RGB', (80, 40), (i, i, i)) for i in range(3)]
    asyncio.run(_detector(pool).run_primary_batch(imgs))
    assert [(i.datatype(), i.shape()) for i in pool.inputs] == [
        ('UINT8', [2, 3, 64, 64]),
        ('UINT8', [1, 3, 64, 64]),
    ]
