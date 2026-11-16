#!/usr/bin/env python3
"""Goldens for the native frame engine's geometry and routing ports (plan 17 section 5 N0).

Runs the Python functions the native engine ports, on inputs this script fixes, and writes what
they answered under ``native/frame_engine/tests/golden/``:

* ``geometry.json``: ``letterbox_geometry``, ``whole_square_geometry``, ``undo_letterbox``,
  ``crop_rect``, ``stored_box_rects`` and ``blur_rect``, floats written as ``repr`` writes them
  (they read back to the same bits);
* ``weights.jsonl``: for every ``(input, output, offset, take, antialias)`` tuple the detector
  letterbox, the frame square and eight crop squares produce on every COCO and Commons pin size
  (and its turned twin), plus a grid of small and awkward sizes, the first 128 bits of the SHA-256
  of the packed weight set (``prep_kernels._weights``: ``first int32[n] | count int32[n] | weights
  float32[n * kmax]``) and its ``n``, ``kmax``, ``lo`` and ``hi``: equal digests are equal bits;
* ``jpeg_header.json``: byte strings built here, marker by marker (so the file does not depend on
  an image library's output), their header read, end-of-image offset, route and oriented size, and
  seeded mutations of them.

The data is committed; the Rust tests read it. This script is deleted with the Python engine at
the switch (plan 17 N3). ``--check`` regenerates into memory and compares with the files.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import struct
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


ROOT = Path(__file__).resolve().parents[2]


def _setup() -> None:
    """Make the repository importable and give ``prep_kernels`` a CuPy to import: it imports the
    library at module level and uses it only inside its kernel class, and the weight packing it
    holds is plain numpy, so a placeholder module is enough."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.modules.setdefault('cupy', types.ModuleType('cupy'))


OUT = ROOT / 'native' / 'frame_engine' / 'tests' / 'golden'
MANIFESTS = ('coco_bench_2000', 'commons_hires_500')
DETECTOR, ENCODER, CROP = 640, 336, 224
EDGE_SIZES = [(1, 1), (1, 4000), (4000, 1), (3, 7), (7, 3), (101, 103), (8000, 6000), (6000, 8000)]
#: Crop boxes as fractions of the frame (x1, y1, x2, y2), small to whole, tall to wide.
CROP_FRACTIONS = [
    (0.0, 0.0, 1.0, 1.0), (0.1, 0.1, 0.5, 0.9), (0.4, 0.4, 0.45, 0.46), (0.0, 0.0, 0.02, 0.5),
    (0.2, 0.3, 0.95, 0.4), (0.33, 0.12, 0.61, 0.77), (0.05, 0.05, 0.9, 0.35), (0.6, 0.2, 0.7, 0.9),
]  # fmt: skip


def _sizes() -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    for name in MANIFESTS:
        path = ROOT / 'scripts' / 'datasets' / 'manifests' / f'{name}.json'
        for image in json.loads(path.read_text(encoding='utf-8'))['images']:
            seen.add((image['width'], image['height']))
            seen.add((image['height'], image['width']))  # an EXIF turn swaps them
    seen.update(EDGE_SIZES)
    return sorted(seen)


# ---------------------------------------------------------------------------------- geometry


def _pairwise(rng: random.Random) -> list[dict[str, Any]]:
    """Rows of float64 where the order of addition shows in the sum, and numpy's ``sum(axis=1)`` of
    each (``filter_weights`` normalises with it); values as their IEEE bit patterns."""
    import numpy as np

    def bits(x: float) -> int:
        return int(struct.unpack('<Q', struct.pack('<d', x))[0])

    out = []
    lengths = (3, 7, 8, 9, 17, 64, 100, 127, 128, 129, 130, 136, 200, 250, 300, 520, 777, 1000)
    for n in [*lengths, *lengths, *lengths]:
        values = [
            rng.choice((1e16, -1e16, 1.0, 0.1, 3.3e-5, 2.0**-30)) * rng.random() for _ in range(n)
        ]
        total = float(np.array(values, np.float64).reshape(1, -1).sum(axis=1)[0])
        out.append({'bits': [bits(v) for v in values], 'sum': bits(total)})
    return out


def _four(
    rng: random.Random, xs: tuple[float, float], ys: tuple[float, float]
) -> tuple[float, float, float, float]:
    return (rng.uniform(*xs), rng.uniform(*ys), rng.uniform(*xs), rng.uniform(*ys))


def _geometry(sizes: list[tuple[int, int]]) -> dict[str, Any]:
    import numpy as np

    from frame_engine import prep_geometry
    from src.domain.image_geometry import (
        crop_rect,
        letterbox_geometry,
        undo_letterbox,
        whole_square_geometry,
    )

    rng = random.Random(20261115)
    # each row is w, h, target, then the answer
    letterbox, square = [], []
    for w, h in sizes:
        for target in (CROP, ENCODER, DETECTOR):
            letterbox.append([w, h, target, *letterbox_geometry(w, h, target)])
            square.append([w, h, target, *whole_square_geometry(w, h, target)])
    undo = []
    for w, h in rng.sample(sizes, 120):
        lb = letterbox_geometry(w, h, DETECTOR)
        for _ in range(4):
            box = _four(rng, (-50, DETECTOR + 50), (-50, DETECTOR + 50))
            undo.append({'box': list(box), 'scale': lb.scale, 'pad': [lb.pad_w, lb.pad_h],
                         'out': list(undo_letterbox(box, lb.scale, (lb.pad_w, lb.pad_h)))})  # fmt: skip
    undo.append({'box': [1.0, 2.0, 3.0, 4.0], 'scale': 0.0, 'pad': [0.0, 0.0],
                 'out': list(undo_letterbox((1.0, 2.0, 3.0, 4.0), 0.0, (0.0, 0.0)))})  # fmt: skip
    rects, blur = [], []
    for w, h in rng.sample(sizes, 150):
        for _ in range(6):
            box = _four(rng, (-0.2 * w, 1.2 * w), (-0.2 * h, 1.2 * h))
            rects.append({'box': list(box), 'w': w, 'h': h, 'out': list(crop_rect(box, w, h))})
            b = prep_geometry.blur_rect(box, w, h)
            blur.append({'box': list(box), 'w': w, 'h': h, 'out': None if b is None else list(b)})
        for x1, y1, x2, y2 in ((0.0, 0.0, 0.0, 0.0), (w - 0.5, h - 0.5, w + 3.0, h + 3.0),
                               (2.5, 3.5, 2.5, 3.5), (0.5, 0.5, 1.5, 1.5)):  # fmt: skip
            box = (x1, y1, x2, y2)
            rects.append({'box': list(box), 'w': w, 'h': h, 'out': list(crop_rect(box, w, h))})
            b = prep_geometry.blur_rect(box, w, h)
            blur.append({'box': list(box), 'w': w, 'h': h, 'out': None if b is None else list(b)})
    stored = []
    for w, h in rng.sample(sizes, 60):
        norm = [[rng.uniform(-0.1, 1.1) for _ in range(4)] for _ in range(5)]
        norm.append([0.1, 0.2, 0.3, 0.4])

        out = prep_geometry.stored_box_rects(np.array(norm, np.float64), w, h)
        stored.append({'boxes': norm, 'w': w, 'h': h, 'out': [list(r) for r in out]})
    return {
        'generated_by': 'scripts/codegen/frame_engine_goldens.py',
        'sources': ['src/domain/image_geometry.py', 'frame_engine/prep_geometry.py'],
        'pairwise_sum': _pairwise(rng),
        'letterbox': letterbox,
        'square': square,
        'undo_letterbox': undo,
        'crop_rect': rects,
        'blur_rect': blur,
        'stored_box_rects': stored,
    }


# ---------------------------------------------------------------------------------- weights


def _weight_tuples(sizes: list[tuple[int, int]]) -> set[tuple[int, int, int, int, bool]]:
    from src.domain.image_geometry import crop_rect, letterbox_geometry, whole_square_geometry

    tuples: set[tuple[int, int, int, int, bool]] = set()
    for w, h in sizes:
        lb = letterbox_geometry(w, h, DETECTOR)
        sq = whole_square_geometry(w, h, ENCODER)
        tuples.update({
            (w, lb.new_w, 0, lb.new_w, True),
            (h, lb.new_h, 0, lb.new_h, True),
            (w, sq.new_w, sq.left, ENCODER, True),
            (h, sq.new_h, sq.top, ENCODER, True),
        })  # fmt: skip
        for x1, y1, x2, y2 in CROP_FRACTIONS:
            rect = crop_rect((x1 * w, y1 * h, x2 * w, y2 * h), w, h)
            rw, rh = rect[2] - rect[0], rect[3] - rect[1]
            g = letterbox_geometry(rw, rh, CROP)
            for antialias in (True, False):
                tuples.add((rw, g.new_w, 0, g.new_w, antialias))
                tuples.add((rh, g.new_h, 0, g.new_h, antialias))
    ins = (1, 2, 3, 7, 8, 9, 31, 32, 33, 63, 64, 127, 128, 129, 255, 256, 257, 640, 1000, 5472)
    outs = (1, 2, 3, 15, 16, 17, 224, 336, 640)
    for n_in in ins:
        for n_out in outs:
            for antialias in (True, False):
                tuples.add((n_in, n_out, 0, n_out, antialias))
                left = n_out // 3
                if n_out - left >= 1:
                    tuples.add((n_in, n_out, left, n_out - left, antialias))
    return tuples


def _weights_lines(sizes: list[tuple[int, int]]) -> list[str]:
    from frame_engine.prep_kernels import _weights

    lines = []
    for n_in, n_out, left, take, antialias in sorted(_weight_tuples(sizes)):
        w = _weights(n_in, n_out, left, take, antialias, False)
        digest = hashlib.sha256(w.blob.tobytes()).hexdigest()
        row = [n_in, n_out, left, take, antialias, w.n, w.kmax, w.lo, w.hi, digest[:32]]
        lines.append(json.dumps(row, separators=(',', ':')))
    return lines


# ---------------------------------------------------------------------------------- JPEG headers


def _seg(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack('>H', len(payload) + 2) + payload


def _sof(marker: int, w: int, h: int, comps: int = 3, precision: int = 8) -> bytes:
    body = struct.pack('>BHHB', precision, h, w, comps) + b''.join(
        bytes([c + 1, 0x11, 0]) for c in range(comps)
    )
    return _seg(marker, body)


def _sos(comps: int = 3) -> bytes:
    return _seg(0xDA, bytes([comps]) + b'\x01\x00' * comps + b'\x00\x3f\x00')


SOI, EOI = b'\xff\xd8', b'\xff\xd9'
ENTROPY = b'\x12\x34\xff\x00\x56\xff\x00\x00\x9a\xbc'
ENTROPY_RST = b'\x12\xff\xd0\x34\xff\xff\xd1\x56\xff\x00\xff\xd7\x78'
EXIF_HEAD = b'Exif\x00\x00'


def _exif(
    order: bytes,
    entries: list[tuple[int, int, int, int]],
    *,
    ifd: int = 8,
    count: int | None = None,
) -> bytes:
    """An APP1 Exif payload whose IFD0 holds ``entries`` of (tag, type, count, 16-bit value)."""
    e = '<' if order == b'II' else '>'
    body = order + struct.pack(e + 'HI', 42, ifd) + b'\x00' * (ifd - 8)
    body += struct.pack(e + 'H', len(entries) if count is None else count)
    for tag, typ, n, value in entries:
        body += struct.pack(e + 'HHIHH', tag, typ, n, value, 0)
    return EXIF_HEAD + body + b'\x00\x00\x00\x00'


def _icc(seq: int, total: int, data: bytes) -> bytes:
    return b'ICC_PROFILE\x00' + bytes([seq, total]) + data


def _jpeg(*parts: bytes, w: int = 640, h: int = 480, sof: bytes | None = None) -> bytes:
    return (
        SOI
        + b''.join(parts)
        + (sof if sof is not None else _sof(0xC0, w, h))
        + _sos()
        + ENTROPY
        + EOI
    )


def _fixtures() -> list[tuple[str, bytes, float, float]]:
    """``(name, bytes, min_megapixels, max_megapixels)``."""
    f: list[tuple[str, bytes, float, float]] = []

    def add(name: str, data: bytes, lo: float = 0.0, hi: float = 100.0) -> None:
        f.append((name, data, lo, hi))

    add('baseline_rgb', _jpeg())
    add('baseline_gray', SOI + _sof(0xC0, 320, 200, 1) + _sos(1) + ENTROPY + EOI)
    add('extended_sequential', SOI + _sof(0xC1, 640, 480) + _sos() + ENTROPY + EOI)
    add('progressive', SOI + _sof(0xC2, 640, 480) + _sos() + ENTROPY + EOI)
    add('lossless', SOI + _sof(0xC3, 640, 480) + _sos() + ENTROPY + EOI)
    add('arithmetic_sequential', SOI + _sof(0xC9, 640, 480) + _sos() + ENTROPY + EOI)
    add('arithmetic_progressive', SOI + _sof(0xCA, 640, 480) + _sos() + ENTROPY + EOI)
    add('twelve_bit', SOI + _sof(0xC1, 640, 480, 3, 12) + _sos() + ENTROPY + EOI)
    add('cmyk', SOI + _sof(0xC0, 640, 480, 4) + _sos(4) + ENTROPY + EOI)
    add('two_components', SOI + _sof(0xC0, 640, 480, 2) + _sos(2) + ENTROPY + EOI)
    add('no_end_marker', SOI + _sof(0xC0, 640, 480) + _sos() + ENTROPY)
    add('cut_inside_a_segment', (SOI + _seg(0xE1, b'x' * 40))[:20])
    add('only_the_start_marker', SOI)
    add('start_marker_and_garbage', SOI + b'\x00\x01\x02\x03\x04\x05')
    add(
        'no_frame_header',
        SOI + _seg(0xE0, b'JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00') + _sos() + ENTROPY + EOI,
    )
    add('bytes_after_the_end', _jpeg() + b'MPF\x00\xff\xd8\xff\xe1more' + b'\x00' * 20)
    add(
        'end_marker_inside_the_thumbnail',
        SOI
        + _seg(0xE1, EXIF_HEAD + b'II*\x00\x08\x00\x00\x00' + b'\xff\xd9' * 3)
        + _sof(0xC0, 640, 480)
        + _sos()
        + ENTROPY
        + EOI,
    )
    add('restart_markers_and_fill_bytes', SOI + _sof(0xC0, 640, 480) + _sos() + ENTROPY_RST + EOI)
    add(
        'fill_bytes_before_a_marker',
        SOI
        + b'\xff\xff\xff'
        + _seg(0xE0, b'JFIF\x00')
        + b'\xff\xff'
        + _sof(0xC0, 640, 480)
        + _sos()
        + ENTROPY
        + b'\xff\xff\xff'
        + EOI,
    )
    add(
        'standalone_markers_in_the_header',
        SOI + b'\xff\xd3\xff\x01' + _sof(0xC0, 640, 480) + _sos() + ENTROPY + EOI,
    )
    add(
        'temporary_marker_in_the_scan',
        SOI + _sof(0xC0, 640, 480) + _sos() + b'\x12\xff\x01\x34' + ENTROPY + EOI,
    )
    add(
        'two_scans',
        SOI
        + _sof(0xC2, 64, 64)
        + _sos()
        + ENTROPY
        + _seg(0xC4, b'\x00' * 20)
        + _sos()
        + ENTROPY
        + EOI,
    )
    add('frame_header_after_the_scan_header', SOI + _sos() + _sof(0xC0, 640, 480) + EOI)
    add('png_signature', b'\x89PNG\r\n\x1a\n' + b'\x00' * 30)
    add('empty', b'')
    add('one_byte', b'\xff')
    add('big_frame_over_the_limit', _jpeg(w=20000, h=10000))
    add('exactly_the_limit', _jpeg(w=10000, h=10000))
    add('just_over_the_limit', _jpeg(w=10001, h=10000))
    add('small_frame', _jpeg(w=100, h=100), lo=1.0)
    add('exactly_the_minimum', _jpeg(w=1000, h=1000), lo=1.0)
    add('zero_size_frame', _jpeg(w=0, h=0))
    add(
        'short_frame_header_then_a_good_one',
        SOI + _seg(0xC0, b'\x08\x00') + _sof(0xC0, 800, 600) + _sos() + ENTROPY + EOI,
    )
    add(
        'zero_length_segment',
        SOI + b'\xff\xe0\x00\x00' + _sof(0xC0, 640, 480) + _sos() + ENTROPY + EOI,
    )
    add(
        'length_one_segment',
        SOI + b'\xff\xe0\x00\x01' + _sof(0xC0, 640, 480) + _sos() + ENTROPY + EOI,
    )
    add('segment_longer_than_the_file', SOI + b'\xff\xe0\xff\xff' + b'abc')
    for value in range(10):
        for order in (b'II', b'MM'):
            add(
                f'orientation_{value}_{order.decode()}',
                _jpeg(_seg(0xE1, _exif(order, [(0x0112, 3, 1, value)]))),
            )
    add(
        'orientation_after_other_tags',
        _jpeg(_seg(0xE1, _exif(b'II', [(0x010F, 2, 1, 5), (0x0110, 2, 1, 6), (0x0112, 3, 1, 6)]))),
    )
    add(
        'orientation_ifd_beyond_the_data',
        _jpeg(_seg(0xE1, _exif(b'II', [(0x0112, 3, 1, 6)], ifd=500))),
    )
    add(
        'orientation_count_past_the_data',
        _jpeg(_seg(0xE1, _exif(b'MM', [(0x0100, 3, 1, 1)], count=40))),
    )
    add('orientation_exif_too_short', _jpeg(_seg(0xE1, EXIF_HEAD + b'II*')))
    add(
        'orientation_bad_byte_order',
        _jpeg(_seg(0xE1, EXIF_HEAD + b'XX*\x00\x08\x00\x00\x00' + b'\x00' * 20)),
    )
    add(
        'exif_with_the_wrong_header',
        _jpeg(_seg(0xE1, b'Exif\x00\x01' + b'II*\x00\x08\x00\x00\x00')),
    )
    add(
        'second_exif_wins',
        _jpeg(
            _seg(0xE1, _exif(b'II', [(0x0112, 3, 1, 3)])),
            _seg(0xE1, _exif(b'II', [(0x0112, 3, 1, 8)])),
        ),
    )
    add('icc_one_chunk', _jpeg(_seg(0xE2, _icc(1, 1, b'profile-bytes'))))
    add(
        'icc_two_chunks_in_order',
        _jpeg(_seg(0xE2, _icc(1, 2, b'AAAA')), _seg(0xE2, _icc(2, 2, b'BBBB'))),
    )
    add(
        'icc_two_chunks_out_of_order',
        _jpeg(_seg(0xE2, _icc(2, 2, b'BBBB')), _seg(0xE2, _icc(1, 2, b'AAAA'))),
    )
    add(
        'icc_a_chunk_is_missing',
        _jpeg(_seg(0xE2, _icc(1, 3, b'AAAA')), _seg(0xE2, _icc(3, 3, b'CCCC'))),
    )
    add('icc_chunk_too_short', _jpeg(_seg(0xE2, b'ICC_PROFILE\x00\x01')))
    add(
        'icc_after_the_frame_header_is_ignored',
        SOI + _sof(0xC0, 640, 480) + _seg(0xE2, _icc(1, 1, b'late')) + _sos() + ENTROPY + EOI,
    )
    add('app2_that_is_not_icc', _jpeg(_seg(0xE2, b'MPF\x00' + b'\x00' * 20)))
    return f


def _case(name: str, data: bytes, lo: float, hi: float) -> dict[str, Any]:
    from frame_engine import prep_geometry

    h = prep_geometry.jpeg_header(data)
    end = prep_geometry.jpeg_end(data)
    r = prep_geometry.route(data, lo, hi)

    def header(x: prep_geometry.JpegHeader) -> dict[str, Any]:
        return {
            'kind': x.kind, 'width': x.width, 'height': x.height, 'components': x.components,
            'precision': x.precision, 'progressive': x.progressive, 'orientation': x.orientation,
            'lossless': x.lossless, 'end': x.end,
            'icc_sha256': None if x.icc is None else hashlib.sha256(x.icc).hexdigest(),
            'icc_len': None if x.icc is None else len(x.icc),
        }  # fmt: skip

    return {
        'name': name,
        'data': base64.b64encode(data).decode(),
        'min_megapixels': lo,
        'max_megapixels': hi,
        'header': header(h),
        'jpeg_end': end,
        'route': {'kind': r.kind, 'reason': r.reason, 'header': header(r.header)},
        'oriented_size': list(prep_geometry.oriented_size(h)),
    }


def _mutations(
    fixtures: list[tuple[str, bytes, float, float]],
) -> list[tuple[str, bytes, float, float]]:
    """Seeded damage to the fixtures: a byte changed, a cut, an inserted marker byte."""
    rng = random.Random(20261116)
    out: list[tuple[str, bytes, float, float]] = []
    for k in range(400):
        name, data, lo, hi = fixtures[rng.randrange(len(fixtures))]
        if not data:
            continue
        b = bytearray(data)
        kind = rng.choice(('flip', 'cut', 'insert_ff', 'insert_marker'))
        at = rng.randrange(len(b))
        if kind == 'flip':
            b[at] = rng.choice((0x00, 0xFF, 0xD9, 0xDA, 0xC0, rng.randrange(256)))
        elif kind == 'cut':
            del b[at:]
        elif kind == 'insert_ff':
            b[at:at] = b'\xff'
        else:
            b[at:at] = bytes([0xFF, rng.choice((0xD9, 0xDA, 0xC0, 0xC2, 0xE1, 0xD3, 0x01, 0x00))])
        out.append((f'mutation_{k:03d}_{kind}_of_{name}', bytes(b), lo, hi))
    return out


def _jpeg_cases() -> dict[str, Any]:
    fixtures = _fixtures()
    cases = [_case(*f) for f in [*fixtures, *_mutations(fixtures)]]
    return {
        'generated_by': 'scripts/codegen/frame_engine_goldens.py',
        'sources': ['frame_engine/prep_geometry.py', 'src/domain/jpeg_markers.py'],
        'cases': cases,
    }


# ---------------------------------------------------------------------------------- files


def _dump(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n'


def render() -> dict[str, str]:
    _setup()
    from scripts.codegen import frame_engine_face_goldens

    sizes = _sizes()
    return {
        'faces.json': _dump(frame_engine_face_goldens.render()),
        'geometry.json': _dump(_geometry(sizes)),
        'weights.jsonl': '\n'.join(_weights_lines(sizes)) + '\n',
        'jpeg_header.json': _dump(_jpeg_cases()),
    }


def main(
    argv: list[str] | None = None,
    make: Callable[[], dict[str, str]] = render,
    script: str = 'scripts/codegen/frame_engine_goldens.py',
) -> int:
    """Write the files ``make`` renders, or with ``--check`` compare them with the committed ones
    (the text goldens' script runs its own renderer through this)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='compare with the committed files')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(argv)
    files = make()
    if args.check:
        stale = [n for n, text in files.items() if not (args.out / n).is_file()
                 or (args.out / n).read_text(encoding='utf-8') != text]  # fmt: skip
        for name in stale:
            sys.stderr.write(f'{args.out / name} is stale: run {script}\n')
        return 1 if stale else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text, encoding='utf-8')
        sys.stdout.write(f'wrote {args.out / name} ({len(text)} bytes)\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
