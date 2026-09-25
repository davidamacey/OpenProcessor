"""Unit tests for scripts.curation.worker.region_embed_stage (LG-1).

Every task is a full `_ItemTask` (mirrors `_make_task` in
test_region_worker.py); the crop JPEG is real so `_crop_region_jpeg` runs
for real, and the PE encoder is a stub returning fixed unit vectors.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

import scripts.curation.worker.region_embed_stage as stage
from scripts.curation.worker.state import _ItemTask
from src.config import get_region_fields


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), (50, 80, 120)).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def _make_task(
    *,
    crop_id: str = 'crop-1',
    status: str | None = None,
    candidate_in_crop: tuple[float, float, float, float] | None = (0.1, 0.1, 0.5, 0.5),
    crop_jpeg: bytes | None = None,
) -> _ItemTask:
    F = get_region_fields()
    t = _ItemTask(
        crop_id=crop_id,
        image_path='/dev/null/never-read',
        vehicle_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        region_status='pending',
        class_name='',
        candidate_in_crop=candidate_in_crop,
        crop_jpeg=crop_jpeg if crop_jpeg is not None else _make_jpeg(),
    )
    if status is not None:
        t.update_doc = {F.status: status}
    return t


class _FakePE:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[int] = []
        self._fail = fail

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        if self._fail:
            msg = 'triton down'
            raise RuntimeError(msg)
        self.calls.append(len(crops))
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(crops), 1))


@pytest.mark.asyncio
class TestEmbedWrittenRegions:
    async def test_detected_task_gets_a_unit_norm_vector(self) -> None:
        F = get_region_fields()
        t = _make_task(status='detected')
        pe = _FakePE()

        await stage.embed_written_regions([t], pe)

        vec = t.update_doc[F.embedding]
        assert len(vec) == 3
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)
        assert pe.calls == [1]

    async def test_rejected_task_gets_nothing(self) -> None:
        F = get_region_fields()
        t = _make_task(status='no_region_box')
        pe = _FakePE()

        await stage.embed_written_regions([t], pe)

        assert F.embedding not in t.update_doc
        assert pe.calls == []

    async def test_task_with_no_update_gets_nothing(self) -> None:
        F = get_region_fields()
        t = _make_task(status=None)  # empty update_doc, e.g. a stale-skip
        pe = _FakePE()

        await stage.embed_written_regions([t], pe)

        assert F.embedding not in t.update_doc
        assert pe.calls == []

    async def test_encoder_exception_leaves_docs_untouched_and_does_not_raise(self) -> None:
        F = get_region_fields()
        t = _make_task(status='detected')
        pe = _FakePE(fail=True)

        await stage.embed_written_regions([t], pe)

        assert F.embedding not in t.update_doc

    async def test_batches_at_encode_batch_size(self) -> None:
        F = get_region_fields()
        tasks = [
            _make_task(crop_id=f'c{i}', status='detected') for i in range(stage.ENCODE_BATCH + 5)
        ]
        pe = _FakePE()

        await stage.embed_written_regions(tasks, pe)

        assert pe.calls == [stage.ENCODE_BATCH, 5]
        for t in tasks:
            assert F.embedding in t.update_doc

    async def test_no_candidate_in_crop_is_skipped(self) -> None:
        F = get_region_fields()
        t = _make_task(status='detected', candidate_in_crop=None)
        pe = _FakePE()

        await stage.embed_written_regions([t], pe)

        assert F.embedding not in t.update_doc
        assert pe.calls == []
