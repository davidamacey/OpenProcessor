"""A region vector is the vector of the stored box's decoded pixels on every CPU path (#360).

The CPU region paths used to embed the crop after a JPEG round trip at quality 92 (the
backfill and the ``embed`` reprocess scope re-encoded each crop, the detection worker cut the
region from the item-crop JPEG), while the GPU route and the crop vectors embed the pixels. On
a large photo the encoder shrinks a small box by a large factor, which turns the JPEG noise
into a large vector change. The stub encoder here is deliberately sensitive to that noise.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image

import scripts.curation.worker.region_embed_stage as stage
from curation.reprocess_fixtures import box, docs, item, make_fake, servable_root
from scripts.curation.worker.state import _ItemTask
from src.config import get_region_fields
from src.domain.region_boxes import RegionBox
from src.services.curation import reprocess_embed
from src.services.curation.reprocess_embed import EmbedTarget, reembed_items


if TYPE_CHECKING:
    from pathlib import Path

F = get_region_fields()
W, H = 1600, 1200
ITEM_BOX = (0.05, 0.05, 0.95, 0.95)
SMALL = (0.30, 0.40, 0.34, 0.44)  # 64 x 48 px of a noisy 1600 x 1200 photo
EDGE = (0.90, 0.90, 1.0, 1.0)  # touches the bottom-right corner
BOXES = (SMALL, EDGE)


def _noise_jpeg() -> bytes:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def _unit(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64)
    return v / np.linalg.norm(v)


def stub_vector(crop: np.ndarray) -> np.ndarray:
    """High-frequency pixels, centred: the part of a crop a JPEG round trip rewrites."""
    return _unit(crop[:16, :16, 0].astype(np.float64).ravel() - 127.5)


class NoiseSensitivePE:
    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        return np.stack([stub_vector(c) for c in crops])

    async def embed_whole_frame_image(self, img: Any) -> np.ndarray:  # noqa: ARG002
        return np.array([0.0, 1.0, 0.0])


def _raw_crop(data: bytes, bbox: tuple[float, float, float, float]) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert('RGB')
    return np.asarray(reprocess_embed.crop_pil(img, reprocess_embed._pixel_box(bbox, *img.size)))


def _jpeg_crop(data: bytes, bbox: tuple[float, float, float, float]) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(_raw_crop(data, bbox)).save(buf, format='JPEG', quality=92)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))


def _item_crop_jpeg(data: bytes, bbox: tuple[float, float, float, float]) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(_raw_crop(data, bbox)).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def test_the_stub_would_notice_a_jpeg_round_trip() -> None:
    data = _noise_jpeg()
    for bbox in BOXES:
        a = stub_vector(_raw_crop(data, bbox))
        b = stub_vector(_jpeg_crop(data, bbox))
        assert float(a @ b) < 0.99


@pytest.mark.asyncio
@pytest.mark.parametrize('bbox', BOXES)
async def test_reembed_region_vector_is_the_vector_of_the_raw_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bbox: tuple[float, float, float, float]
) -> None:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'big.jpg'
    data = _noise_jpeg()
    path.write_bytes(data)
    doc = item(
        'c1',
        image_path=str(path),
        bbox_norm=ITEM_BOX,
        boxes=(box('b1', state='accepted', bbox=bbox, reason=None),),
    )
    fake = make_fake([doc], [{'image_id': 'img-1', 'image_path': str(path)}])
    target = EmbedTarget(image_id='img-1', image_path=str(path), items=[('c1', doc)])

    counts = await reembed_items(fake, NoiseSensitivePE(), [target], parts=frozenset({'region'}))

    assert counts['region_written'] == 1
    (entry,) = docs(fake)['c1'][F.box_embeddings]
    expected = stub_vector(_raw_crop(data, bbox))
    assert float(_unit(np.array(entry['embedding'])) @ expected) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_worker_region_vector_is_the_vector_of_the_raw_source_pixels(tmp_path: Path) -> None:
    path = tmp_path / 'big.jpg'
    data = _noise_jpeg()
    path.write_bytes(data)
    item_bbox = (0.2, 0.2, 1.0, 1.0)  # the region box lies inside the item crop
    t = _ItemTask(
        crop_id='c1',
        image_path=str(path),
        item_bbox_norm=item_bbox,
        region_status='pending',
        class_name='',
        crop_jpeg=_item_crop_jpeg(data, item_bbox),
    )
    t.pending_boxes = [
        RegionBox(box_id=f'b{i}', bbox_norm=bbox, state='accepted') for i, bbox in enumerate(BOXES)
    ]

    await stage.embed_written_regions([t], NoiseSensitivePE())

    for i, bbox in enumerate(BOXES):
        expected = stub_vector(_raw_crop(data, bbox))
        assert float(_unit(np.array(t.box_vectors[f'b{i}'])) @ expected) == pytest.approx(
            1.0, abs=1e-6
        )
