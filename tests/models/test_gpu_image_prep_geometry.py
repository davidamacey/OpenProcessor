"""The GPU decode's host-side arithmetic reproduces the CPU ingest path it stands in for
(``models/gpu_image_prep/1/prep_geometry.py``; the CUDA kernels apply these weights)."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.services.curation.ingest_index import crop_pil
from src.services.detection.crop_quality import crop_lap_var, variance_from_sums
from src.services.detection.geometry import letterbox_to_square, undo_letterbox
from src.utils.pe_preprocess import resize_crop_rgb, whole_frame_rgb_from_image


_PATH = Path(__file__).resolve().parents[2] / 'models' / 'gpu_image_prep' / '1' / 'prep_geometry.py'
_spec = importlib.util.spec_from_file_location('prep_geometry', _PATH)
assert _spec is not None
assert _spec.loader is not None
fg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fg)


def _jpeg(w: int, h: int, *, progressive: bool = False, orientation: int | None = None) -> bytes:
    rng = np.random.default_rng(w + h)
    img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    kw: dict = {'quality': 90, 'progressive': progressive}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kw['exif'] = exif.tobytes()
    img.save(buf, 'JPEG', **kw)
    return buf.getvalue()


def _resample(img: np.ndarray, out_w: int, out_h: int, *, antialias: bool) -> np.ndarray:
    """What the two CUDA passes compute, in float64 on the host."""
    fx, cx, wx = fg.filter_weights(img.shape[1], out_w, antialias=antialias)
    fy, cy, wy = fg.filter_weights(img.shape[0], out_h, antialias=antialias)
    src = img.astype(np.float64)
    tmp = np.zeros((img.shape[0], out_w, 3))
    for x in range(out_w):
        taps = src[:, fx[x] : fx[x] + cx[x]]
        tmp[:, x] = np.tensordot(taps, wx[x, : cx[x]], axes=([1], [0]))
    out = np.zeros((out_h, out_w, 3))
    for y in range(out_h):
        out[y] = np.tensordot(wy[y, : cy[y]], tmp[fy[y] : fy[y] + cy[y]], axes=([0], [0]))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class TestRoute:
    def test_a_baseline_jpeg_above_the_gate_goes_to_the_gpu(self) -> None:
        reason, header = fg.route(_jpeg(400, 300), 0.1, 100)
        assert reason == 'gpu'
        assert (header.width, header.height) == (400, 300)

    @pytest.mark.parametrize(
        ('data', 'reason'),
        [
            (b'\x89PNG\r\n\x1a\n' + b'\x00' * 40, 'not_jpeg'),
            (b'\xff\xd8\xff\xe0\x00\x02', 'corrupt'),
            (_jpeg(400, 300, progressive=True), 'progressive'),
            (_jpeg(400, 300)[:-200], 'truncated'),
            (_jpeg(200, 100), 'small'),
        ],
    )
    def test_everything_else_stays_on_the_cpu(self, data: bytes, reason: str) -> None:
        assert fg.route(data, 0.1, 100)[0] == reason

    def test_the_upper_gate(self) -> None:
        assert fg.route(_jpeg(400, 300), 0.0, 0.1)[0] == 'huge'

    @pytest.mark.parametrize('orientation', range(1, 9))
    def test_oriented_size_matches_exif_transpose(self, orientation: int) -> None:
        from PIL import ImageOps

        data = _jpeg(64, 32, orientation=orientation)
        _, header = fg.route(data, 0, 100)
        assert (
            fg.oriented_size(header) == ImageOps.exif_transpose(Image.open(io.BytesIO(data))).size
        )


class TestGeometry:
    @pytest.mark.parametrize('size', [(5184, 3456), (3000, 4000), (640, 480), (333, 500)])
    def test_letterbox_and_undo_match_the_detector_path(self, size: tuple[int, int]) -> None:
        img = Image.new('RGB', size)
        _chw, scale, pad = letterbox_to_square(img, 640)
        lb = fg.letterbox_geometry(*size, 640)
        assert (lb.scale, lb.pad_w, lb.pad_h) == (scale, *pad)
        box = (0.1, 0.2, 0.7, 0.9)
        x1, y1, x2, y2 = (v * 640 for v in box)
        assert fg.undo_letterbox(box, 640, lb) == undo_letterbox((x1, y1, x2, y2), scale, pad)

    @pytest.mark.parametrize(
        'box', [(10.4, 20.6, 50.5, 70.49), (-5.0, -3.2, 30.0, 400.7), (1.0, 1.0, 1.2, 1.3)]
    )
    def test_crop_and_blur_rectangles_match_the_cpu_path(
        self, box: tuple[float, float, float, float]
    ) -> None:
        img = Image.new('RGB', (120, 90))
        x1, y1, x2, y2 = fg.crop_rect(box, 120, 90)
        assert crop_pil(img, box).size == (x2 - x1, y2 - y1)
        rgb = np.random.default_rng(1).integers(0, 256, (90, 120, 3), dtype=np.uint8)
        rect = fg.blur_rect(box, 120, 90)
        expected = crop_lap_var(rgb, box, 'rgb')
        if rect is None:
            assert expected is None
        else:
            gray = cv2.cvtColor(rgb[rect[1] : rect[3], rect[0] : rect[2]], cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_16S).astype(np.int64)
            assert fg.lap_var(lap.size, int(lap.sum()), int((lap * lap).sum())) == expected

    def test_variance_from_sums_is_the_cpu_rounding(self) -> None:
        assert fg.lap_var(10, 10, 30) == variance_from_sums(10, 10, 30) == 2.0
        assert fg.lap_var(0, 0, 0) is None


class TestStoredBoxes:
    """The embed pass sends stored ``bbox_norm`` boxes (#259); the model cuts the rectangle the
    CPU embed pass cuts (``reprocess_embed``: the box scaled to pixels, then ``crop_pil``)."""

    BOXES = [
        (0.1, 0.1, 0.6, 0.6),
        (0.0, 0.0, 1.0, 1.0),
        (0.83, 0.9, 1.0, 1.0),  # touches the right and bottom edge
        (0.123456789, 0.5, 0.123456790, 0.5000001),  # under a pixel: one pixel kept
        (0.99999, 0.99999, 1.0, 1.0),
        (0.3051, 0.0049, 0.6949, 0.4951),
    ]

    def _boxes(self, seed: int) -> list[tuple[float, ...]]:
        rng = np.random.default_rng(seed)
        out = list(self.BOXES)
        for _ in range(300):
            x1, x2 = sorted(float(v) for v in rng.random(2))
            y1, y2 = sorted(float(v) for v in rng.random(2))
            out.append((x1, y1, x2, y2))
        return out

    @pytest.mark.parametrize('size', [(5472, 3648), (3648, 5472), (120, 90), (1001, 999)])
    def test_each_stored_box_is_the_cpu_embed_pass_rectangle(self, size: tuple[int, int]) -> None:
        from src.services.curation.reprocess_embed import _pixel_box

        w, h = size
        # Edges stored at half-pixel positions: rounding them in float32 instead of float64
        # moves some by a pixel, so the input must carry the stored floats exactly.
        halves = [((k + 0.5) / w, (k + 0.5) / h) for k in range(0, min(w, h) - 1, 7)]
        boxes = [*self._boxes(w), *((x, y, 1.0, 1.0) for x, y in halves)]
        rects = fg.stored_box_rects(np.array(boxes, np.float64), w, h)
        assert rects == [fg.crop_rect(_pixel_box(b, w, h), w, h) for b in boxes]

    def test_the_pixels_are_the_cpu_crop(self) -> None:
        from src.services.curation.reprocess_embed import _pixel_box

        rgb = np.random.default_rng(3).integers(0, 256, (90, 120, 3), dtype=np.uint8)
        pil = Image.fromarray(rgb)
        boxes = self._boxes(9)
        for box, (x1, y1, x2, y2) in zip(
            boxes, fg.stored_box_rects(np.array(boxes, np.float64), 120, 90), strict=True
        ):
            cpu = np.asarray(crop_pil(pil, _pixel_box(box, 120, 90)))
            np.testing.assert_array_equal(cpu, rgb[y1:y2, x1:x2])

    def test_no_boxes_is_no_rectangles(self) -> None:
        assert fg.stored_box_rects(np.zeros((0, 4), np.float64), 640, 480) == []


class TestWeights:
    @pytest.mark.parametrize('size', [(900, 600), (400, 640), (500, 500)])
    def test_the_whole_square_matches_pillow_within_a_grey_level(
        self, size: tuple[int, int]
    ) -> None:
        rgb = np.random.default_rng(size[0]).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        g = fg.whole_square_geometry(*size, 336)
        full = _resample(rgb, g.new_w, g.new_h, antialias=True)
        ours = full[g.top : g.top + 336, g.left : g.left + 336]
        ref = whole_frame_rgb_from_image(Image.fromarray(rgb))
        assert np.abs(ours.astype(int) - ref).max() <= 1

    @pytest.mark.parametrize('size', [(700, 500), (120, 90), (40, 300)])
    def test_crop_squares_match_opencv_linear_within_a_grey_level(
        self, size: tuple[int, int]
    ) -> None:
        rgb = np.random.default_rng(size[1]).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        g = fg.whole_square_geometry(*size, 336)
        full = _resample(rgb, g.new_w, g.new_h, antialias=False)
        ours = full[g.top : g.top + 336, g.left : g.left + 336]
        ref = resize_crop_rgb(rgb)
        assert np.abs(ours.astype(int) - ref).max() <= 1

    def test_weights_are_normalized(self) -> None:
        for antialias in (True, False):
            _f, c, w = fg.filter_weights(5184, 640, antialias=antialias)
            np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
            assert (c > 0).all()
