"""Unit tests for scripts.curation.worker.region_embed_stage (LG-1).

Every task is a full `_ItemTask` (mirrors `_make_task` in
test_region_worker.py); the crop JPEG is real so `_crop_region_jpeg` runs
for real, and the PE encoder is a stub returning fixed unit vectors. The
stage embeds each ACCEPTED box of ``pending_boxes`` and keys the vector by
the box's pre-merge id.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

import scripts.curation.worker.region_embed_stage as stage
from scripts.curation.worker.state import _ItemTask
from src.services.curation.region_boxes import RegionBox


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), (50, 80, 120)).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def _box(box_id: str, state: str = 'accepted', x: float = 0.1) -> RegionBox:
    return RegionBox(box_id=box_id, bbox_norm=(x, 0.1, x + 0.2, 0.4), state=state)


def _make_task(
    *,
    crop_id: str = 'crop-1',
    boxes: list[RegionBox] | None = None,
    crop_jpeg: bytes | None = None,
) -> _ItemTask:
    t = _ItemTask(
        crop_id=crop_id,
        image_path='/dev/null/never-read',
        item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        region_status='pending',
        class_name='',
        crop_jpeg=crop_jpeg if crop_jpeg is not None else _make_jpeg(),
    )
    t.pending_boxes = boxes
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
    async def test_every_accepted_box_gets_its_own_unit_norm_vector(self) -> None:
        t = _make_task(boxes=[_box('b1'), _box('b2', x=0.5)])
        pe = _FakePE()

        await stage.embed_written_regions([t], pe)

        assert set(t.box_vectors) == {'b1', 'b2'}
        for vec in t.box_vectors.values():
            assert len(vec) == 3
            assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)
        assert pe.calls == [2]

    async def test_only_accepted_boxes_are_embedded(self) -> None:
        t = _make_task(boxes=[_box('b1'), _box('b2', 'rejected', 0.4), _box('b3', 'proposed', 0.6)])
        pe = _FakePE()

        await stage.embed_written_regions([t], pe)

        assert set(t.box_vectors) == {'b1'}
        assert pe.calls == [1]

    async def test_each_box_is_cropped_from_its_own_geometry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[float, float, float, float]] = []
        real = stage._crop_region_jpeg

        def _recording(jpeg: bytes, region_in_crop: tuple[float, float, float, float]) -> bytes:
            seen.append(region_in_crop)
            return real(jpeg, region_in_crop)

        monkeypatch.setattr(stage, '_crop_region_jpeg', _recording)
        t = _make_task(boxes=[_box('b1', x=0.1), _box('b2', x=0.5)])

        await stage.embed_written_regions([t], _FakePE())

        assert seen == [pytest.approx((0.1, 0.1, 0.3, 0.4)), pytest.approx((0.5, 0.1, 0.7, 0.4))]

    async def test_task_with_no_boxes_gets_nothing(self) -> None:
        t = _make_task(boxes=None)  # e.g. a stale-skip with no box write
        pe = _FakePE()

        await stage.embed_written_regions([t], pe)

        assert t.box_vectors == {}
        assert pe.calls == []

    async def test_task_without_a_loaded_crop_gets_nothing(self) -> None:
        t = _make_task(boxes=[_box('b1')])
        t.crop_jpeg = None
        pe = _FakePE()

        await stage.embed_written_regions([t], pe)

        assert t.box_vectors == {}
        assert pe.calls == []

    async def test_encoder_exception_leaves_tasks_untouched_and_does_not_raise(self) -> None:
        t = _make_task(boxes=[_box('b1')])

        await stage.embed_written_regions([t], _FakePE(fail=True))

        assert t.box_vectors == {}

    async def test_batches_at_encode_batch_size_across_tasks(self) -> None:
        tasks = [
            _make_task(crop_id=f'c{i}', boxes=[_box('b1')]) for i in range(stage.ENCODE_BATCH + 5)
        ]
        pe = _FakePE()

        await stage.embed_written_regions(tasks, pe)

        assert pe.calls == [stage.ENCODE_BATCH, 5]
        for t in tasks:
            assert 'b1' in t.box_vectors
