"""The engine's host decode (``frame_engine/host_decode.py``) gives the pixels the API's CPU path
gives, and reads an image's size from its header before any pixel is decoded.

Ground truth for the pixels is Pillow itself (``ImageOps.exif_transpose`` and ``convert('RGB')``),
not ``decode_image``: a wrapper that returned a bare PIL image, BGR, RGBA, a strided view or an
un-rotated frame would agree with ``decode_image`` on its own mistake only if both shared it.
"""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image, ImageOps

from frame_engine import host_decode
from src.utils.image_decode import decode_image


def _pixels(w: int = 48, h: int = 32, mode: str = 'RGB') -> Image.Image:
    rng = np.random.default_rng(w * h)
    img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    return img if mode == 'RGB' else img.convert(mode)


def _encode(img: Image.Image, fmt: str, **kw: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _exif(orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif.tobytes()


CASES = {
    'jpeg': lambda: _encode(_pixels(), 'JPEG', quality=90),
    'jpeg_progressive': lambda: _encode(_pixels(), 'JPEG', quality=90, progressive=True),
    'jpeg_grey': lambda: _encode(_pixels(mode='L'), 'JPEG', quality=90),
    'jpeg_cmyk': lambda: _encode(_pixels(mode='CMYK'), 'JPEG', quality=90),
    'jpeg_rotated': lambda: _encode(_pixels(), 'JPEG', quality=90, exif=_exif(6)),
    'png': lambda: _encode(_pixels(), 'PNG'),
    'png_rgba': lambda: _encode(_pixels(mode='RGBA'), 'PNG'),
    'png_palette': lambda: _encode(_pixels().quantize(16), 'PNG'),
    'png_rotated': lambda: _encode(_pixels(), 'PNG', exif=_exif(8)),
    'webp': lambda: _encode(_pixels(), 'WEBP', lossless=True),
    'gif': lambda: _encode(_pixels().quantize(32), 'GIF'),
    'bmp': lambda: _encode(_pixels(), 'BMP'),
}


def _reference(data: bytes) -> np.ndarray:
    upright = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    return np.asarray(upright.convert('RGB'))


@pytest.mark.parametrize('name', sorted(CASES))
def test_the_frame_is_the_upright_rgb_pixels_of_the_api_decode(name: str) -> None:
    data = CASES[name]()
    frame = host_decode.decode(data)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.flags['C_CONTIGUOUS']
    assert np.array_equal(frame, np.asarray(decode_image(data).pil))
    if not name.startswith('jpeg'):  # a lossy file differs by the decoder's rounding, not the lib
        assert np.array_equal(frame, _reference(data))
    else:
        assert np.abs(frame.astype(int) - _reference(data)).max() <= 2


def test_an_exif_rotated_file_is_upright_in_the_frame_and_in_the_probe() -> None:
    data = CASES['png_rotated']()
    assert host_decode.decode(data).shape[:2] == (48, 32)  # 48 wide, 32 high, turned by 8
    assert host_decode.probe(data) == host_decode.HostHeader(width=32, height=48)


@pytest.mark.parametrize('name', ['jpeg', 'png', 'gif', 'webp', 'jpeg_rotated'])
def test_the_probe_reads_the_size_the_decode_gives(name: str) -> None:
    data = CASES[name]()
    h, w = host_decode.decode(data).shape[:2]
    assert host_decode.probe(data) == host_decode.HostHeader(width=w, height=h)


def _png_header(width: int, height: int) -> bytes:
    """A PNG whose header says ``width`` x ``height`` and whose pixel data is a few bytes: what a
    decompression bomb is."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
        )

    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b'\x00' * 16)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b'')


def test_the_probe_reports_a_large_png_by_its_header_without_decoding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_pixels(*_a: object, **_k: object) -> None:
        raise AssertionError('the probe decoded pixels')

    monkeypatch.setattr(Image.Image, 'load', no_pixels)
    assert host_decode.probe(_png_header(9000, 9000)) == host_decode.HostHeader(9000, 9000)


def test_a_header_past_pillows_own_bomb_limit_is_too_large_not_undecodable() -> None:
    assert Image.MAX_IMAGE_PIXELS is not None
    side = int((2 * Image.MAX_IMAGE_PIXELS) ** 0.5) + 10  # past twice the limit: Pillow raises
    with pytest.raises(host_decode.TooLargeError):
        host_decode.probe(_png_header(side, side))


@pytest.mark.parametrize('data', [b'', b'not an image at all', b'\xff\xd8\xff\xe0\x00\x02'])
def test_a_file_pillow_cannot_read_fails_with_the_decode_error(data: bytes) -> None:
    with pytest.raises(host_decode.HostDecodeError):
        host_decode.probe(data)
    with pytest.raises(host_decode.HostDecodeError):
        host_decode.decode(data)


def test_a_truncated_jpeg_is_refused_not_decoded_with_a_grey_tail() -> None:
    data = CASES['jpeg']()
    with pytest.raises(host_decode.HostDecodeError):
        host_decode.decode(data[: len(data) // 2])
