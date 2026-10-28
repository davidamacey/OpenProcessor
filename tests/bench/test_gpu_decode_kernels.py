"""The torch kernels of the GPU decode spike against the CPU functions they replace.

Run on CPU torch: geometry, fill, the integer blur math and the grey conversion are
device-independent, so an exact match here is an exact match on the GPU for the same pixels; the
resampling is held to the tensor gate (T) of the parent plan.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from scripts.bench import gpu_decode_kernels as k, gpu_decode_lib as lib
from src.services.curation.ingest_index import crop_pil
from src.services.detection.crop_quality import crop_lap_var, image_lap_var
from src.services.detection.geometry import crop_rect, letterbox_to_square
from src.utils.pe_preprocess import resize_crop_rgb


torch = pytest.importorskip('torch')


def _photo(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Smooth gradients plus texture: closer to a photo than white noise (resampling of pure
    noise exaggerates every rounding difference)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = np.stack(
        [128 + 100 * np.sin(xx / 37 + c) * np.cos(yy / 53 - c) for c in range(3)], axis=-1
    )
    return np.clip(base + rng.normal(0, 6, size=base.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize('size', [(480, 640), (640, 427), (1365, 2048), (300, 50)])
def test_letterbox_matches_product_geometry_and_passes_t(size: tuple[int, int]) -> None:
    rgb = _photo(*size)
    ref, scale, pad = letterbox_to_square(Image.fromarray(rgb), 640)
    got, g_scale, g_pad = k.letterbox_u8(torch.from_numpy(rgb), 640)
    assert (g_scale, g_pad) == (scale, pad)
    assert got.dtype == torch.uint8
    assert tuple(got.shape) == ref.shape
    ref_hwc, got_hwc = ref[0].transpose(1, 2, 0), got[0].numpy().transpose(1, 2, 0)
    geo = lib.letterbox_geometry(size[1], size[0], 640)
    pad_mask = np.ones((640, 640), dtype=bool)
    pad_mask[geo.top : geo.top + geo.new_h, geo.left : geo.left + geo.new_w] = False
    assert (got_hwc[pad_mask] == 114).all()
    assert (ref_hwc[pad_mask] == 114).all()
    d = lib.tensor_diff(ref_hwc, got_hwc)
    assert d.best_shift == (0, 0), d
    assert d.mean_abs < 0.5, d
    assert d.max_abs <= 8, d


@pytest.mark.parametrize('size', [(400, 300), (90, 1200), (2000, 1500), (12, 7)])
def test_pe_square_matches_cv2_linear(size: tuple[int, int]) -> None:
    rgb = _photo(*size, seed=2)
    ref = resize_crop_rgb(rgb)
    got = k.pe_square_u8(torch.from_numpy(rgb)).numpy()
    assert got.shape == ref.shape == (336, 336, 3)
    d = lib.tensor_diff(ref, got)
    assert d.best_shift == (0, 0), d
    assert d.mean_abs < 0.5, d
    assert d.max_abs <= 8, d


def test_crop_matches_crop_pil_for_the_stored_box() -> None:
    rgb = _photo(80, 100)
    img = Image.fromarray(rgb)
    for box in [(10.4, 5.5, 50.6, 60.49), (-3.2, -1.0, 20.0, 20.0), (90.0, 70.0, 130.2, 95.1)]:
        ref = np.asarray(crop_pil(img, box))
        got = k.crop_u8(torch.from_numpy(rgb), crop_rect(box, 100, 80)).numpy()
        np.testing.assert_array_equal(got, ref)


def test_gray_is_opencv_rgb2gray_bit_for_bit() -> None:
    rgb = np.random.default_rng(3).integers(0, 256, size=(97, 131, 3), dtype=np.uint8)
    ref = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    np.testing.assert_array_equal(k.gray_u8(torch.from_numpy(rgb)).numpy(), ref)
    every = np.stack(np.meshgrid(np.arange(256), np.arange(256), indexing='ij'), -1)
    for blue in (0, 77, 255):
        cube = np.concatenate([every, np.full((256, 256, 1), blue)], axis=-1).astype(np.uint8)
        np.testing.assert_array_equal(
            k.gray_u8(torch.from_numpy(cube)).numpy(), cv2.cvtColor(cube, cv2.COLOR_RGB2GRAY)
        )


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_blur_variance_is_the_product_value(seed: int) -> None:
    rgb = np.random.default_rng(seed).integers(0, 256, size=(123, 77, 3), dtype=np.uint8)
    assert k.lap_var(torch.from_numpy(rgb)) == image_lap_var(rgb, 'rgb')
    photo = _photo(200, 300, seed)
    assert k.lap_var(torch.from_numpy(photo)) == image_lap_var(photo, 'rgb')
    box = (20.3, 10.0, 180.6, 150.2)
    x1, y1, x2, y2 = (round(v) for v in box)
    assert k.lap_var(torch.from_numpy(photo[y1:y2, x1:x2])) == crop_lap_var(photo, box, 'rgb')


@pytest.mark.parametrize('chunk', [1, 7, 50, 0])
def test_chunked_laplacian_sums_are_exact(chunk: int) -> None:
    rgb = np.random.default_rng(9).integers(0, 256, size=(61, 45, 3), dtype=np.uint8)
    t = torch.from_numpy(rgb)
    assert k.lap_sums(t, rows_per_chunk=chunk) == k.lap_sums(t)
    assert k.var_from_sums(*k.lap_sums_gray(k.gray_u8(t), chunk)) == image_lap_var(rgb, 'rgb')


def test_whole_frame_square_shape_and_dtype() -> None:
    rgb = _photo(900, 1600)
    for mode in ('antialias', 'area', 'dct8'):
        out = k.whole_frame_u8(torch.from_numpy(rgb), mode=mode)
        assert out.dtype == torch.uint8
        assert tuple(out.shape) == (336, 336, 3)
    with pytest.raises(ValueError, match='mode'):
        k.whole_frame_u8(torch.from_numpy(rgb), mode='nearest')


def test_dct8_whole_frame_tracks_the_reduced_decode() -> None:
    """The 1/8 emulation lands near OpenCV's ``IMREAD_REDUCED_COLOR_8`` path (the whole frame
    before the single decode); the antialiased full-resolution resize lands near today's
    whole frame (Pillow ``BILINEAR`` from the full frame)."""
    from PIL import Image

    from src.utils.pe_preprocess import resize_crop_rgb, whole_frame_rgb_from_image

    rgb = _photo(480, 640, seed=5)
    ok, enc = cv2.imencode(
        '.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    assert ok
    data = np.frombuffer(enc.tobytes(), np.uint8)
    decoded = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    reduced = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_8), cv2.COLOR_BGR2RGB)
    old = resize_crop_rgb(reduced)
    new = whole_frame_rgb_from_image(Image.fromarray(decoded))
    dct8 = k.whole_frame_u8(torch.from_numpy(decoded), mode='dct8').numpy()
    aa = k.whole_frame_u8(torch.from_numpy(decoded), mode='antialias').numpy()
    assert lib.tensor_diff(old, dct8).mean_abs < 2.0
    assert lib.tensor_diff(old, dct8).mean_abs < lib.tensor_diff(old, aa).mean_abs
    assert lib.tensor_diff(new, aa).mean_abs < lib.tensor_diff(new, dct8).mean_abs


@pytest.mark.parametrize('orientation', range(1, 9))
def test_orientation_matches_pillow_exif_transpose(orientation: int) -> None:
    """For decoders that ignore the EXIF tag (measured: torchvision's GPU decode, DALI's
    default), the frame must be re-oriented as ``ImageOps.exif_transpose`` does."""
    from PIL import ImageOps

    rgb = _photo(30, 50, seed=orientation)
    img = Image.fromarray(rgb)
    exif = img.getexif()
    exif[0x0112] = orientation
    img.info['exif'] = exif.tobytes()
    ref = np.asarray(ImageOps.exif_transpose(img))
    got = k.apply_orientation(torch.from_numpy(rgb), orientation).numpy()
    np.testing.assert_array_equal(got, ref)
