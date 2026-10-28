"""#259: the PE vector is computed from the box that is stored.

Ingest used to cut the unclamped detector box (Pillow zero-pads past the frame) while the stored
``bbox_norm`` is clamped to the image; the vector then described pixels the stored box does not
contain. Both decode paths now cut the image-clamped rectangle; interior boxes are unchanged.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from curation.ingest_fakes import FakePEEncoder, _jpeg_bytes, _make_service
from src.services.curation.ingest_index import crop_pil
from src.services.detection.geometry import crop_rect


_PATH = Path(__file__).resolve().parents[2] / 'models' / 'gpu_image_prep' / '1' / 'prep_geometry.py'
_spec = importlib.util.spec_from_file_location('prep_geometry_259', _PATH)
assert _spec is not None
assert _spec.loader is not None
gpu_geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu_geometry)

W, H = 100, 80
EDGE_BOXES = [
    (-5.4, -3.2, 30.0, 40.0),  # left and top
    (60.0, 20.0, 112.6, 90.3),  # right and bottom
    (-1.0, -1.0, 150.0, 120.0),  # all four sides
]
INTERIOR_BOXES = [(10.4, 5.5, 50.6, 60.49), (0.0, 0.0, 100.0, 80.0), (33.3, 33.3, 33.4, 33.4)]


def _old_crop(img: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """``crop_pil`` as it was before #259: unclamped, Pillow pads outside the frame."""
    x1, y1, x2, y2 = box
    x1i, y1i = max(0, round(x1)), max(0, round(y1))
    return img.crop((x1i, y1i, max(x1i + 1, round(x2)), max(y1i + 1, round(y2))))


def _photo() -> Image.Image:
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(1, 256, (H, W, 3), dtype=np.uint8))  # no zero pixels


@pytest.mark.parametrize('box', EDGE_BOXES)
def test_an_edge_box_is_cut_from_the_image_clamped_rectangle(box) -> None:
    img = _photo()
    x1, y1, x2, y2 = crop_rect(box, W, H)
    assert 0 <= x1 < x2 <= W
    assert 0 <= y1 < y2 <= H
    out = np.asarray(crop_pil(img, box))
    np.testing.assert_array_equal(out, np.asarray(img)[y1:y2, x1:x2])
    assert (out == 0).sum() == 0  # no black padding


@pytest.mark.parametrize('box', INTERIOR_BOXES)
def test_an_interior_box_is_bit_identical_to_the_old_crop(box) -> None:
    img = _photo()
    np.testing.assert_array_equal(np.asarray(crop_pil(img, box)), np.asarray(_old_crop(img, box)))


def test_a_box_wholly_outside_the_frame_keeps_one_pixel_inside() -> None:
    assert crop_rect((120.0, 90.0, 130.0, 95.0), W, H) == (99, 79, 100, 80)


@pytest.mark.parametrize('box', [*EDGE_BOXES, *INTERIOR_BOXES, (120.0, 90.0, 130.0, 95.0)])
def test_the_gpu_rectangle_is_the_cpu_rectangle(box) -> None:
    assert gpu_geometry.crop_rect(box, W, H) == crop_rect(box, W, H)
    assert crop_pil(Image.new('RGB', (W, H)), box).size == (
        gpu_geometry.crop_rect(box, W, H)[2] - gpu_geometry.crop_rect(box, W, H)[0],
        gpu_geometry.crop_rect(box, W, H)[3] - gpu_geometry.crop_rect(box, W, H)[1],
    )


class _RecordingPE(FakePEEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.crops: list[np.ndarray] = []

    async def embed_crops(self, crops, max_batch: int = 32):
        self.crops.extend(crops)
        return await super().embed_crops(crops, max_batch)


@pytest.mark.asyncio
async def test_ingest_embeds_the_crop_of_the_stored_box(tmp_path) -> None:
    # Network-square boxes; the 400x300 frame letterboxes to 320x240 with 40 px bars, so the
    # first box runs past the left and top edges and the second past the right edge.
    svc, os_fake, _ = _make_service(
        detections=[(-0.05, 0.05, 0.5, 0.6, 0.9, 1), (0.3, 0.4, 1.05, 0.8, 0.9, 1)]
    )
    pe = _RecordingPE()
    svc.pe_encoder = pe
    data = _jpeg_bytes((400, 300), seed=5)
    result = await svc.ingest_batch([data], ['/x/edge.jpg'])
    assert result.summary.successful == 1
    img = Image.open(io.BytesIO(data)).convert('RGB')
    assert len(os_fake.items) == len(pe.crops) == 2
    by_size = {c.shape[:2]: c for c in pe.crops}
    for doc in os_fake.items.values():
        x1, y1, x2, y2 = doc['bbox_norm']
        assert min(x1, y1) >= 0
        assert max(x2, y2) <= 1
        stored = np.asarray(crop_pil(img, (x1 * 400, y1 * 300, x2 * 400, y2 * 300)))
        np.testing.assert_array_equal(by_size[stored.shape[:2]], stored)
