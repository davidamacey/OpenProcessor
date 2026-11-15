"""JPEG marker vocabulary, the EXIF orientation reader and the ICC profile joiner, standard library only.

Shared by the frame engine's header routing (``frame_engine/prep_geometry.py``) and the decode
probes (``scripts/bench/gpu_decode_lib.py``), which read headers without decoding.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
PROGRESSIVE = frozenset({0xC2, 0xC6, 0xCA, 0xCE})
LOSSLESS = frozenset({0xC3, 0xC7, 0xCB, 0xCF})
#: Markers with no length field.
STANDALONE = frozenset({0x01, *range(0xD0, 0xD8)})
#: An APP2 segment that carries one chunk of the embedded ICC profile (ICC.1 annex B.4).
ICC_MARKER = 0xE2
ICC_TAG = b'ICC_PROFILE\x00'


def join_icc_chunks(chunks: list[bytes]) -> bytes | None:
    """The ICC profile split over the APP2 payloads ``chunks`` (each ``ICC_PROFILE\\0``, its
    sequence number, the chunk count, then its bytes), joined in sequence order; ``None`` when the
    count does not match. Pillow's ``JpegImagePlugin`` rule, so the GPU and CPU routes see the same
    profile."""
    if not chunks or any(len(c) < 14 for c in chunks):
        return None
    ordered = sorted(chunks)
    if ordered[0][13] != len(ordered):
        return None
    return b''.join(c[14:] for c in ordered)


def exif_orientation(payload: bytes) -> int:
    """Orientation tag (0x0112) of IFD0 of an APP1 ``Exif`` payload; 1 when absent or invalid."""
    tiff = payload[6:]
    if len(tiff) < 8 or tiff[:2] not in (b'II', b'MM'):
        return 1
    end = '<' if tiff[:2] == b'II' else '>'
    (ifd,) = struct.unpack(end + 'I', tiff[4:8])
    if ifd + 2 > len(tiff):
        return 1
    (count,) = struct.unpack(end + 'H', tiff[ifd : ifd + 2])
    for k in range(count):
        entry = tiff[ifd + 2 + 12 * k : ifd + 14 + 12 * k]
        if len(entry) < 12:
            break
        tag, _typ, _n = struct.unpack(end + 'HHI', entry[:8])
        if tag == 0x0112:
            (value,) = struct.unpack(end + 'H', entry[8:10])
            return value if 1 <= value <= 8 else 1
    return 1


def header_segments(data: bytes | memoryview) -> Iterator[tuple[int, bytes]]:
    """``(marker, payload)`` of each length-carrying segment after the start marker, up to the
    first scan (``SOS``) or the first byte that is not a marker. Fill bytes and standalone markers
    are skipped."""
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in STANDALONE:
            i += 2
            continue
        if marker == 0xDA:
            return
        (length,) = struct.unpack('>H', data[i + 2 : i + 4])
        yield (
            marker,
            bytes(data[i + 4 : i + 2 + length]),
        )  # bytes, also for a view: chunks are sorted
        i += 2 + length
