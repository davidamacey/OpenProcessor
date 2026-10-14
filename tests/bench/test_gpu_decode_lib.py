"""Pure logic of the GPU decode probes: JPEG header routing, geometry that must match the CPU
path exactly, the tensor-level gate, the routing crossover and the throughput ceiling."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from scripts.bench import gpu_decode_lib as lib
from src.services.detection.geometry import letterbox_params


def _jpeg(size: tuple[int, int] = (64, 48), mode: str = 'RGB', **save: object) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (10, 200, 30) if mode == 'RGB' else 0).save(buf, 'JPEG', **save)
    return buf.getvalue()


def _with_orientation(data: bytes, orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    img = Image.open(io.BytesIO(data))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', exif=exif.tobytes())
    return buf.getvalue()


# ------------------------------------------------------------------------------ JPEG header


def test_header_baseline_rgb() -> None:
    h = lib.jpeg_header(_jpeg((64, 48)))
    assert (h.kind, h.width, h.height, h.components, h.precision) == ('jpeg', 64, 48, 3, 8)
    assert not h.progressive
    assert h.orientation == 1


def test_header_progressive_gray_cmyk() -> None:
    assert lib.jpeg_header(_jpeg(progressive=True)).progressive
    assert lib.jpeg_header(_jpeg(mode='L')).components == 1
    assert lib.jpeg_header(_jpeg(mode='CMYK')).components == 4


@pytest.mark.parametrize('orientation', [1, 3, 6, 8])
def test_header_exif_orientation(orientation: int) -> None:
    assert lib.jpeg_header(_with_orientation(_jpeg(), orientation)).orientation == orientation


def test_header_icc_profile_seen() -> None:
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    assert lib.jpeg_header(_jpeg(icc_profile=icc)).has_icc
    assert not lib.jpeg_header(_jpeg()).has_icc


def test_header_other_formats_and_garbage() -> None:
    buf = io.BytesIO()
    Image.new('RGB', (8, 8)).save(buf, 'PNG')
    assert lib.jpeg_header(buf.getvalue()).kind == 'png'
    assert lib.jpeg_header(b'RIFF\x00\x00\x00\x00WEBPVP8 ').kind == 'webp'
    assert lib.jpeg_header(b'\x00\x00\x00\x18ftypheic').kind == 'heif'
    assert lib.jpeg_header(b'not an image').kind == 'unknown'
    truncated = _jpeg()[:20]
    assert lib.jpeg_header(truncated).kind == 'corrupt'


def test_header_twelve_bit_sof_parsed() -> None:
    data = bytearray(_jpeg((16, 16)))
    sof = data.index(b'\xff\xc0')
    data[sof + 4] = 12  # precision byte of the frame header
    assert lib.jpeg_header(bytes(data)).precision == 12


# ------------------------------------------------------------------------------- routing


def test_route_explains_every_cpu_case() -> None:
    policy = lib.RoutePolicy(min_megapixels=2.0, max_megapixels=60.0, progressive_on_gpu=False)
    big = lib.JpegHeader('jpeg', 4000, 3000, 3, 8, False, 1, False)
    assert lib.route(big, policy) == 'gpu'
    assert lib.route(big._replace(width=640, height=480), policy) == 'cpu:small'
    assert lib.route(big._replace(width=20000, height=20000), policy) == 'cpu:huge'
    assert lib.route(big._replace(progressive=True), policy) == 'cpu:progressive'
    assert lib.route(big._replace(components=4), policy) == 'cpu:cmyk'
    assert lib.route(big._replace(precision=12), policy) == 'cpu:precision'
    assert lib.route(big._replace(kind='png'), policy) == 'cpu:not_jpeg'
    assert lib.route(big._replace(kind='corrupt'), policy) == 'cpu:corrupt'
    lenient = policy._replace(progressive_on_gpu=True)
    assert lib.route(big._replace(progressive=True), lenient) == 'gpu'


# ------------------------------------------------------------------------------ geometry


@pytest.mark.parametrize('size', [(640, 480), (480, 640), (5472, 3648), (1, 900), (1023, 767)])
def test_letterbox_geometry_matches_product(size: tuple[int, int]) -> None:
    w, h = size
    g = lib.letterbox_geometry(w, h, 640)
    scale, (pad_w, pad_h) = letterbox_params(Image.new('RGB', size), target=640)
    assert g.scale == pytest.approx(scale)
    assert (g.pad_w, g.pad_h) == (pad_w, pad_h)
    assert (g.left, g.top) == (int(pad_w), int(pad_h))
    assert g.left + g.new_w <= 640
    assert g.top + g.new_h <= 640


@pytest.mark.parametrize('size', [(336, 336), (1000, 400), (400, 1000), (37, 3), (5472, 3648)])
def test_pe_square_geometry_matches_product(size: tuple[int, int]) -> None:
    from src.services.detection.pe_preprocess import resize_crop_rgb

    w, h = size
    g = lib.pe_square_geometry(w, h, 336)
    rgb = np.arange(h * w * 3, dtype=np.uint64).reshape(h, w, 3).astype(np.uint8)
    assert resize_crop_rgb(rgb).shape == (336, 336, 3)
    assert g.new_w >= 336
    assert g.new_h >= 336
    assert g.top == (g.new_h - 336) // 2
    assert g.left == (g.new_w - 336) // 2
    scale = 336 / min(h, w)
    assert (g.new_w, g.new_h) == (max(336, round(w * scale)), max(336, round(h * scale)))


def test_crop_rect_matches_crop_pil() -> None:
    from src.services.curation.ingest_index import crop_pil

    img = Image.new('RGB', (100, 80))
    for box in [(10.4, 5.5, 50.6, 60.49), (-3.2, -1.0, 20.0, 20.0), (90.0, 70.0, 90.2, 70.1)]:
        x1, y1, x2, y2 = lib.crop_rect(box)
        assert crop_pil(img, box).size == (x2 - x1, y2 - y1)
        assert min(x1, y1) >= 0
        assert x2 > x1
        assert y2 > y1


# ---------------------------------------------------------------------------- tensor gate


def test_tensor_diff_identity_and_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
    same = lib.tensor_diff(a, a.copy())
    assert same.best_shift == (0, 0)
    assert same.mean_abs == 0.0
    assert same.max_abs == 0
    shifted = np.roll(a, 1, axis=1)
    d = lib.tensor_diff(a, shifted)
    assert d.best_shift == (0, 1)
    assert d.mean_abs >= 0.5


def test_tensor_diff_small_noise_passes_large_fails() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(10, 240, size=(64, 64, 3), dtype=np.uint8)
    noisy = (a.astype(np.int16) + rng.integers(-1, 2, size=a.shape)).astype(np.uint8)
    d = lib.tensor_diff(a, noisy)
    assert d.best_shift == (0, 0)
    assert 0 < d.mean_abs < 0.8
    assert d.max_abs == 1
    far = a.copy()
    far[0, 0, 0] = 255 if a[0, 0, 0] < 128 else 0
    assert lib.tensor_diff(a, far).max_abs > 8


def test_tensor_diff_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='shape'):
        lib.tensor_diff(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8))


# --------------------------------------------------------------------- stats and ceilings


def test_summarize() -> None:
    s = lib.summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s['n'] == 5
    assert s['median'] == 3.0
    assert s['min'] == 1.0
    assert s['max'] == 100.0
    assert lib.summarize([])['n'] == 0


def test_ceiling_takes_the_binding_resource() -> None:
    c = lib.ceiling(host_cpu_s=0.25, host_cores=40, gpu_ms=[(14.0, 2), (5.0, 1)])
    assert c['host'] == pytest.approx(160.0)
    assert c['gpu'] == pytest.approx(min(2 * 1000 / 14.0, 1000 / 5.0))
    assert c['images_per_s'] == pytest.approx(min(c['host'], c['gpu']))
    assert c['binding'] == 'gpu'
