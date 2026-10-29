"""Unit tests for scripts.curation.worker.region_embed_stage (LG-1).

Every task is a full `_ItemTask` (mirrors `_make_task` in
region_worker_helpers.py) over a real source image on disk, and the PE encoder
is a stub returning fixed unit vectors. The stage embeds each ACCEPTED box of
``pending_boxes`` cut from the decoded source pixels (never the item-crop JPEG,
#360) and keys the vector by the box's pre-merge id.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

import scripts.curation.worker.region_embed_stage as stage
from scripts.curation.worker.state import _ItemTask
from src.domain.region_boxes import RegionBox


if TYPE_CHECKING:
    from pathlib import Path


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), (50, 80, 120)).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _source_image(tmp_path: Path) -> None:
    global _SOURCE  # noqa: PLW0603
    path = tmp_path / 'source.jpg'
    path.write_bytes(_make_jpeg())
    _SOURCE = str(path)


_SOURCE = ''


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
        image_path=_SOURCE,
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

    async def test_each_box_is_cropped_from_its_own_geometry(self) -> None:
        sizes: list[tuple[int, ...]] = []

        class _Recording(_FakePE):
            async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:
                sizes.extend(c.shape for c in crops)
                return await super().embed_crops(crops, max_batch)

        t = _make_task(boxes=[_box('b1', x=0.1), _box('b2', x=0.3)])
        t.pending_boxes = [
            RegionBox(box_id='b1', bbox_norm=(0.0, 0.0, 0.5, 0.5), state='accepted'),
            RegionBox(box_id='b2', bbox_norm=(0.5, 0.25, 1.0, 1.0), state='accepted'),
        ]

        await stage.embed_written_regions([t], _Recording())

        assert sizes == [(120, 160, 3), (180, 160, 3)]  # source is 320x240

    async def test_unreadable_source_leaves_boxes_without_a_vector(self) -> None:
        t = _make_task(boxes=[_box('b1')])
        t.image_path = '/nonexistent/source.jpg'

        await stage.embed_written_regions([t], _FakePE())

        assert t.box_vectors == {}

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


def _noise_photo(path: Path, name: str, size: tuple[int, int] = (200, 150)) -> str:
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    out = path / f'{name}.png'
    Image.fromarray(arr).save(out)
    return str(out)


def _task_on(path: str, crop_id: str, boxes: list[RegionBox]) -> _ItemTask:
    t = _make_task(crop_id=crop_id, boxes=boxes)
    t.image_path = path
    return t


class _RecordingPE:
    def __init__(self) -> None:
        self.crops: list[np.ndarray] = []

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        self.crops.extend(crops)
        return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(crops), 1))


@pytest.mark.asyncio
class TestDecodeOncePerPhoto:
    async def test_two_items_of_one_photo_decode_it_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        photo = _noise_photo(tmp_path, 'a')
        other = _noise_photo(tmp_path, 'b')
        tasks = [
            _task_on(photo, 'c1', [_box('b1')]),
            _task_on(other, 'c2', [_box('b2')]),
            _task_on(photo, 'c3', [_box('b3', x=0.5)]),
        ]
        decoded: list[str] = []
        real = stage.decode_source_rgb
        monkeypatch.setattr(stage, 'decode_source_rgb', lambda p: (decoded.append(p), real(p))[1])

        await stage.embed_written_regions(tasks, _FakePE())

        assert sorted(decoded) == sorted([photo, other])
        assert all(set(t.box_vectors) == {t.pending_boxes[0].box_id} for t in tasks)

    async def test_concurrent_decodes_are_bounded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import threading
        import time

        tasks = [
            _task_on(_noise_photo(tmp_path, f'p{i}'), f'c{i}', [_box(f'b{i}')]) for i in range(8)
        ]
        live = peak = 0
        lock = threading.Lock()
        real = stage.decode_source_rgb

        def _spy(p: str) -> Image.Image | None:
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.05)
            try:
                return real(p)
            finally:
                with lock:
                    live -= 1

        monkeypatch.setattr(stage, 'decode_source_rgb', _spy)
        monkeypatch.setattr(stage, 'decode_concurrency', lambda: 2)

        await stage.embed_written_regions(tasks, _FakePE())

        assert peak == 2
        assert all(t.box_vectors for t in tasks)

    async def test_vectors_match_the_per_item_decode(self, tmp_path: Path) -> None:
        photo = _noise_photo(tmp_path, 'm', (320, 240))
        boxes_a = [_box('a1'), _box('a2', x=0.5)]
        boxes_b = [_box('b1', x=0.3)]
        tasks = [_task_on(photo, 'c1', boxes_a), _task_on(photo, 'c2', boxes_b)]
        pe = _RecordingPE()

        await stage.embed_written_regions(tasks, pe)

        from src.services.curation.crop_bytes import decode_source_rgb
        from src.services.curation.reprocess_embed import crop_arrays

        expected: list[np.ndarray] = []
        for boxes in (boxes_a, boxes_b):
            img = decode_source_rgb(photo)
            assert img is not None
            expected.extend(crop_arrays(img, [b.bbox_norm for b in boxes]))
        assert len(pe.crops) == len(expected)
        for got, want in zip(pe.crops, expected, strict=True):
            assert np.array_equal(got, want)


def test_decode_concurrency_is_cpu_and_memory_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage.os, 'cpu_count', lambda: 64)
    monkeypatch.setenv('OP_INGEST_INFLIGHT_MB', '256')
    assert stage.decode_concurrency() == 256 // stage.DECODED_MB_21MP
    monkeypatch.setenv('OP_INGEST_INFLIGHT_MB', '100000')
    assert stage.decode_concurrency() == 64 // 2
    monkeypatch.setattr(stage.os, 'cpu_count', lambda: 1)
    assert stage.decode_concurrency() == 1
