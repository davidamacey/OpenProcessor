"""Embedded colour profiles to sRGB (``src/utils/icc_srgb.py``) and the API's CPU decode, which
converts tagged photos by the same rule as the frame engine.

``decode_image`` (``src/utils/image_decode.py``) is the route every photo the engine does not decode
takes. Ground truth: littleCMS through Pillow (``ImageCms.profileToProfile``) on the decoded
samples. The engine's half of the agreement is the table of ``icc_srgb.classify`` applied per
pixel, emulated here on the host exactly as ``frame_engine/colour.py`` indexes it.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import simplejpeg
from PIL import Image, ImageCms

from src.utils import icc_srgb
from src.utils.image_decode import decode_image


ICC = Path(__file__).resolve().parent / 'fixtures' / 'icc'
P3 = (ICC / 'DisplayP3-v4.icc').read_bytes()
ADOBE = (ICC / 'AdobeCompat-v4.icc').read_bytes()
COLORD_SRGB = (ICC / 'colord-sRGB.icc').read_bytes()
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))


def _pixels() -> np.ndarray:
    return np.random.default_rng(11).integers(0, 256, (40, 64, 3), dtype=np.uint8)


def _encode(fmt: str, icc: bytes | None, mode: str = 'RGB') -> bytes:
    img = Image.fromarray(_pixels())
    if mode != 'RGB':
        img = img.convert(mode)
    buf = io.BytesIO()
    kw = {'icc_profile': icc} if icc else {}
    img.save(buf, fmt, **({'quality': 95} if fmt == 'JPEG' else {}), **kw)
    return buf.getvalue()


def _lcms(rgb: np.ndarray, icc: bytes) -> np.ndarray:
    prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    return np.asarray(ImageCms.profileToProfile(Image.fromarray(rgb), prof, SRGB, outputMode='RGB'))


@pytest.mark.parametrize('icc', [P3, ADOBE])
def test_a_wide_gamut_jpeg_decodes_to_its_littlecms_conversion(icc: bytes) -> None:
    data = _encode('JPEG', icc)
    raw = simplejpeg.decode_jpeg(data, colorspace='RGB')
    out = decode_image(data)
    assert np.array_equal(np.asarray(out.pil), _lcms(raw, icc))
    assert out.rgb is not None
    assert np.array_equal(out.rgb, np.asarray(out.pil))


def test_the_cpu_and_gpu_routes_give_the_same_pixels_for_a_p3_jpeg() -> None:
    data = _encode('JPEG', P3)
    raw = simplejpeg.decode_jpeg(data, colorspace='RGB')
    _, table = icc_srgb.classify(P3)
    assert table is not None
    idx = raw[..., 0].astype(np.int64) * 65536 + raw[..., 1].astype(np.int64) * 256 + raw[..., 2]
    assert np.array_equal(np.asarray(decode_image(data).pil), table[idx])


@pytest.mark.parametrize('icc', [None, COLORD_SRGB])
def test_untagged_and_srgb_jpegs_decode_exactly_as_before(icc: bytes | None) -> None:
    data = _encode('JPEG', icc)
    assert np.array_equal(
        np.asarray(decode_image(data).pil), simplejpeg.decode_jpeg(data, colorspace='RGB')
    )


def test_a_p3_png_takes_the_pillow_path_and_is_converted() -> None:
    data = _encode('PNG', P3)
    assert np.array_equal(np.asarray(decode_image(data).pil), _lcms(_pixels(), P3))


def test_a_grey_jpeg_ignores_an_rgb_profile() -> None:
    data = _encode('JPEG', P3, mode='L')
    raw = simplejpeg.decode_jpeg(data, colorspace='RGB')
    assert np.array_equal(np.asarray(decode_image(data).pil), raw)


class TestRule:
    def test_untagged_and_srgb_profiles_are_left_alone(self) -> None:
        assert icc_srgb.profile_kind(None) == 'srgb'
        # colord's sRGB moves some inputs by one level through littleCMS (tone-curve rounding)
        assert icc_srgb.profile_kind(COLORD_SRGB) == 'srgb'

    @pytest.mark.parametrize('icc', [P3, ADOBE])
    def test_wide_gamut_profiles_are_converted(self, icc: bytes) -> None:
        assert icc_srgb.profile_kind(icc) == 'convert'

    def test_a_profile_littlecms_cannot_open_is_ignored(self) -> None:
        assert icc_srgb.profile_kind(b'not a profile' * 20) == 'other'

    def test_the_table_is_littlecms_at_every_input_it_is_asked(self) -> None:
        kind, table = icc_srgb.classify(P3)
        assert kind == 'convert'
        assert table is not None
        assert table.shape == (1 << 24, 3)
        rng = np.random.default_rng(3)
        px = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        px[0, :8] = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                     [255, 255, 0], [0, 255, 255], [255, 0, 255]]  # fmt: skip
        ref = np.asarray(ImageCms.profileToProfile(
            Image.fromarray(px), ImageCms.ImageCmsProfile(io.BytesIO(P3)), SRGB, outputMode='RGB'
        ))  # fmt: skip
        idx = px[..., 0].astype(np.int64) * 65536 + px[..., 1].astype(np.int64) * 256 + px[..., 2]
        assert np.array_equal(table[idx], ref)

    def test_to_srgb_converts_rgb_only(self) -> None:
        rgb = Image.new('RGB', (4, 4), (200, 30, 40))
        grey = Image.new('L', (4, 4), 90)
        assert icc_srgb.to_srgb(rgb, None) is rgb
        assert icc_srgb.to_srgb(rgb, COLORD_SRGB) is rgb
        assert icc_srgb.to_srgb(grey, P3) is grey
        assert icc_srgb.to_srgb(rgb, P3).getpixel((0, 0)) != (200, 30, 40)
