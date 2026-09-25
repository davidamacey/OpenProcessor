"""Unit tests for src.services.detection.region_embed."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.services.detection.region_embed import decode_region_jpeg, embed_region_crops


def _jpeg_bytes(size: tuple[int, int] = (64, 48), seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return buf.getvalue()


class TestDecodeRegionJpeg:
    def test_decodes_valid_jpeg_to_hwc_uint8_rgb(self) -> None:
        arr = decode_region_jpeg(_jpeg_bytes((64, 48)))
        assert arr is not None
        assert arr.dtype == np.uint8
        assert arr.shape == (48, 64, 3)

    def test_returns_none_for_garbage_bytes(self) -> None:
        assert decode_region_jpeg(b'not a jpeg') is None

    def test_returns_none_for_empty_bytes(self) -> None:
        assert decode_region_jpeg(b'') is None


class _FakePE:
    def __init__(self) -> None:
        self.embed_crops_calls: list[int] = []

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        self.embed_crops_calls.append(len(crops))
        # A fixed, already-unit-norm vector per crop.
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(crops), 1))


@pytest.mark.asyncio
class TestEmbedRegionCrops:
    async def test_returns_one_embedding_per_good_input(self) -> None:
        pe = _FakePE()
        jpegs = [_jpeg_bytes(seed=i) for i in range(3)]

        out = await embed_region_crops(pe, jpegs)

        assert len(out) == 3
        assert pe.embed_crops_calls == [3]
        for vec in out:
            assert vec is not None
            assert len(vec) == 3
            assert vec == pytest.approx([1.0, 0.0, 0.0])
            # Round-tripped through PEEncoder, which L2-normalizes.
            assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)

    async def test_skips_undecodable_entries_without_failing_the_batch(self) -> None:
        pe = _FakePE()
        jpegs = [_jpeg_bytes(seed=0), b'not a jpeg', _jpeg_bytes(seed=1)]

        out = await embed_region_crops(pe, jpegs)

        assert len(out) == 3
        assert out[0] is not None
        assert out[1] is None
        assert out[2] is not None
        # Only the 2 decodable crops reach the encoder.
        assert pe.embed_crops_calls == [2]

    async def test_all_undecodable_never_calls_the_encoder(self) -> None:
        pe = _FakePE()
        out = await embed_region_crops(pe, [b'garbage', b'also garbage'])
        assert out == [None, None]
        assert pe.embed_crops_calls == []
