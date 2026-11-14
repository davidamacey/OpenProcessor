"""The GPU decode's host-side arithmetic reproduces the CPU ingest path it stands in for
(``frame_engine/prep_geometry.py``; the CUDA kernels apply these weights)."""

from __future__ import annotations

import io
import time
from collections.abc import Callable

import cv2
import numpy as np
import pytest
from PIL import Image

from frame_engine import prep_geometry as fg
from src.domain.image_geometry import (
    crop_rect,
    letterbox_geometry,
    undo_letterbox,
    whole_square_geometry,
)
from src.services.curation.ingest_index import crop_pil
from src.services.detection.crop_quality import crop_lap_var, variance_from_sums
from src.services.detection.geometry import letterbox_to_square
from src.utils.pe_preprocess import resize_crop_rgb, whole_frame_rgb_from_image


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


def _stream(rng: np.random.Generator, *, scans: int, scan_bytes: int) -> bytes:
    """A JPEG-shaped byte stream: header segments, then ``scans`` entropy-coded scans whose
    bytes hold stuffed ``FF00`` pairs, restart markers and ``FF`` fill runs, then the end marker."""
    out = bytearray(b'\xff\xd8')
    out += b'\xff\xe1' + (12).to_bytes(2, 'big') + b'Exif\xff\xd9tail'  # an EOI inside a header
    for _ in range(scans):
        out += b'\xff\xc4' + (5).to_bytes(2, 'big') + b'\x00\x01\x02'
        out += b'\xff\xda' + (8).to_bytes(2, 'big') + bytes(6)
        body = rng.integers(0, 256, scan_bytes, dtype=np.uint8).tobytes()
        body = body.replace(b'\xff', b'\xff\x00')
        if scan_bytes > 64:
            mid = len(body) // 2
            body = body[:mid] + b'\xff\xd3' + b'\xff\xff\x00' + body[mid:]
        out += body
    out += b'\xff\xff\xd9' if scans % 2 else b'\xff\xd9'
    return bytes(out)


def _walk_end(data: bytes) -> int | None:
    """The end-of-image offset by a plain marker-by-marker walk (the reference for the scan)."""
    n, i, in_scan = len(data), 2, False
    while True:
        if in_scan:
            while True:
                i = data.find(b'\xff', i)
                if i < 0:
                    return None
                while i < n and data[i] == 0xFF:
                    i += 1
                if i >= n:
                    return None
                if data[i] != 0 and not 0xD0 <= data[i] <= 0xD7:
                    break
            marker, i = data[i], i + 1
        else:
            if i >= n or data[i] != 0xFF:
                return None
            while i < n and data[i] == 0xFF:
                i += 1
            if i >= n:
                return None
            marker, i = data[i], i + 1
        if marker == 0xD9:
            return i
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > n:
            return None
        i += int.from_bytes(data[i : i + 2], 'big')
        in_scan = marker == 0xDA


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

    @pytest.mark.parametrize(
        'trailer', [b'\x00' * 7, b'SEFT' + bytes(range(256)) * 4, b'\xff\xd9junk']
    )
    def test_a_complete_jpeg_with_a_trailer_goes_to_the_gpu_without_it(
        self, trailer: bytes
    ) -> None:
        jpeg = _jpeg(400, 300)
        reason, header = fg.route(jpeg + trailer, 0.1, 100)
        assert reason == 'gpu'
        assert header.end == len(jpeg)

    def test_an_end_marker_inside_a_header_segment_is_not_the_end_of_the_image(self) -> None:
        jpeg = _jpeg(400, 300)
        payload = b'Exif\x00\x00thumbnail\xff\xd9tail'
        app1 = b'\xff\xe1' + (len(payload) + 2).to_bytes(2, 'big') + payload
        data = jpeg[:2] + app1 + jpeg[2:]
        assert fg.route(data, 0.1, 100)[0] == 'gpu'
        assert fg.route(data[:-2], 0.1, 100)[0] == 'truncated'

    def test_a_missing_end_marker_with_a_trailer_is_still_truncated(self) -> None:
        assert fg.route(_jpeg(400, 300)[:-2] + b'\x00' * 50, 0.1, 100)[0] == 'truncated'

    def test_the_end_scan_agrees_with_a_marker_by_marker_walk(self) -> None:
        rng = np.random.default_rng(7)
        for case in range(60):
            scans = int(rng.integers(1, 4))
            data = _stream(rng, scans=scans, scan_bytes=int(rng.integers(0, 4000)))
            cut = int(rng.integers(0, 3))
            if cut == 1:
                data = data[: int(rng.integers(2, len(data)))]  # truncated anywhere
            elif cut == 2:
                data += bytes(rng.integers(0, 256, 300, dtype=np.uint8))  # a trailer
            assert fg.jpeg_end(data) == _walk_end(data), case

    def test_the_end_scan_of_a_large_photo_is_a_vectorised_pass(self) -> None:
        """The scan runs in every worker thread of the prep model: a per-byte interpreter loop
        over a 7 MB photo held the interpreter lock for about 80 ms and capped one prep process
        at about 7 photos per second."""
        data = _stream(np.random.default_rng(1), scans=1, scan_bytes=7_000_000)
        assert fg.jpeg_end(data) == len(data)

        def best(fn: Callable[[], object]) -> float:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                fn()
                times.append(time.perf_counter() - start)
            return min(times)

        # The scan may cost a few C-speed passes over the bytes (a pass is ``bytes.count``); the
        # interpreter loop cost about 17 of them.
        assert best(lambda: fg.jpeg_end(data)) < 4 * best(lambda: data.count(b'\xff'))

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
        lb = letterbox_geometry(*size, 640)
        assert (lb.scale, lb.pad_w, lb.pad_h) == (scale, *pad)
        box = (0.1, 0.2, 0.7, 0.9)
        x1, y1, x2, y2 = (v * 640 for v in box)
        assert undo_letterbox((x1, y1, x2, y2), lb.scale, (lb.pad_w, lb.pad_h)) == undo_letterbox(
            (x1, y1, x2, y2), scale, pad
        )

    @pytest.mark.parametrize(
        'box', [(10.4, 20.6, 50.5, 70.49), (-5.0, -3.2, 30.0, 400.7), (1.0, 1.0, 1.2, 1.3)]
    )
    def test_crop_and_blur_rectangles_match_the_cpu_path(
        self, box: tuple[float, float, float, float]
    ) -> None:
        img = Image.new('RGB', (120, 90))
        x1, y1, x2, y2 = crop_rect(box, 120, 90)
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
        out: list[tuple[float, ...]] = list(self.BOXES)
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
        assert rects == [crop_rect(_pixel_box(b, w, h), w, h) for b in boxes]

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
        g = whole_square_geometry(*size, 336)
        full = _resample(rgb, g.new_w, g.new_h, antialias=True)
        ours = full[g.top : g.top + 336, g.left : g.left + 336]
        ref = whole_frame_rgb_from_image(Image.fromarray(rgb))
        assert np.abs(ours.astype(int) - ref).max() <= 1

    @pytest.mark.parametrize('size', [(700, 500), (120, 90), (40, 300)])
    def test_crop_squares_match_opencv_linear_within_a_grey_level(
        self, size: tuple[int, int]
    ) -> None:
        rgb = np.random.default_rng(size[1]).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        g = whole_square_geometry(*size, 336)
        full = _resample(rgb, g.new_w, g.new_h, antialias=False)
        ours = full[g.top : g.top + 336, g.left : g.left + 336]
        ref = resize_crop_rgb(rgb)
        assert np.abs(ours.astype(int) - ref).max() <= 1

    def test_weights_are_normalized(self) -> None:
        for antialias in (True, False):
            _f, c, w = fg.filter_weights(5184, 640, antialias=antialias)
            np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
            assert (c > 0).all()


class TestLanczosWeights:
    def test_lanczos_weights_reproduce_pillow_lanczos(self) -> None:
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, (230, 310, 3), dtype=np.uint8)
        fx, cx, wx = fg.filter_weights(310, 77, lanczos=True)
        fy, cy, wy = fg.filter_weights(230, 58, lanczos=True)
        src = img.astype(np.float64)
        tmp = np.stack(
            [
                np.tensordot(src[:, fx[x] : fx[x] + cx[x]], wx[x, : cx[x]], axes=([1], [0]))
                for x in range(77)
            ],
            axis=1,
        )
        out = np.stack(
            [
                np.tensordot(wy[y, : cy[y]], tmp[fy[y] : fy[y] + cy[y]], axes=([0], [0]))
                for y in range(58)
            ]
        )
        ref = np.asarray(Image.fromarray(img).resize((77, 58), Image.Resampling.LANCZOS))
        assert np.abs(np.clip(np.rint(out), 0, 255) - ref).max() <= 2

    def test_triangle_weights_are_not_lanczos(self) -> None:
        a = fg.filter_weights(300, 50)[2]
        b = fg.filter_weights(300, 50, lanczos=True)[2]
        assert a.shape != b.shape or not np.allclose(a, b)


class TestFitSize:
    @pytest.mark.parametrize('side', [32, 128, 256, 511, 512, 1024])
    def test_fit_size_is_pillow_thumbnail_for_every_shape(self, side: int) -> None:
        rng = np.random.default_rng(side)
        shapes = [(int(w), int(h)) for w, h in rng.integers(1, 6000, (60, 2))]
        shapes += [(5000, 4000), (4000, 5000), (3, 9000), (9000, 3), (side, side), (side + 1, 7)]
        for w, h in shapes:
            im = Image.new('L', (w, h))
            im.thumbnail((side, side))
            assert fg.fit_size(w, h, side) == im.size, (w, h)
