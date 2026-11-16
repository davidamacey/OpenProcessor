#!/usr/bin/env python3
"""Goldens for the Rust port of the duplicate hash (plan 18a, slice R1x, module ``imohash``).

Runs the Python ``imohash`` package at the version ``requirements.txt`` pins (the product's
``content_imohash`` calls ``imohash.hashfileobject``; the source-backfill path calls
``imohash.hashfile``, which is the same function on a file) over inputs this script fixes, and
writes what it answered to ``native/op_media/tests/goldens/imohash.json``:

* ``synthetic``: sizes around the sampling threshold (128 KiB) and four times the sample size
  (64 KiB), whose bytes follow the position-dependent ``pattern`` below, so a sample read from the
  wrong offset gives another hash;
* ``tiled``: a committed image repeated until it passes the threshold (a real file's bytes on the
  sampled path);
* ``sparse``: sizes whose varint is two to five bytes long, built as a sparse file with a marker
  at each sampled offset;
* ``files``: every committed test image, whole (all are under the threshold).

The Rust tests read the file and rebuild the same inputs. ``--check`` regenerates in memory and
compares with the committed file (the pre-commit drift hook runs it).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import imohash


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'native' / 'op_media' / 'tests' / 'goldens' / 'imohash.json'

THRESHOLD = 128 * 1024
SAMPLE = 16 * 1024
#: byte i of a synthetic file: ((i * 2654435761) >> 13 ^ (i >> 16)) & 0xFF.
PATTERN = '((i * 2654435761) >> 13 ^ (i >> 16)) & 0xFF'
SYNTHETIC_SIZES = [
    0, 1, 2, 15, 16, 17, 127, 128, 129, 16383, 16384, 16385, 32768, 49151, 49152, 49153,
    65535, 65536, 65537, 65538, 98303, 131071, 131072, 131073, 131074, 147455, 147456, 147457,
    196608, 262143, 262144, 262145, 1000003, 1048576,
]  # fmt: skip
#: sparse sizes with a 2-, 3-, 4- and 5-byte size varint.
SPARSE_SIZES = [16384 * 8 + 3, 2**21 + 3, 2**28 + 5, 2**33 + 1]
TILED = [('tests/fixtures/sample_image.jpg', 40), ('tests/fixtures/sample_image.jpg', 22)]
IMAGE_DIRS = [
    'native/frame_engine/tests/golden/host',
    'tests/fixtures',
    'frontend/e2e/fixtures/ingest',
]
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
MARKER = b'OPMARK'


def pattern_bytes(size: int) -> bytes:
    return bytes(((i * 2654435761) >> 13 ^ (i >> 16)) & 0xFF for i in range(size))


def hex_of(data: bytes) -> str:
    digest: str = imohash.hashfileobject(io.BytesIO(data)).hex()
    return digest


def marker_offsets(size: int) -> list[int]:
    return [0, size // 2, size - SAMPLE]


def sparse_hash(size: int) -> str:
    """imohash of a sparse file of ``size`` bytes carrying ``MARKER`` + index at each offset."""
    with tempfile.TemporaryDirectory(prefix='imohash-sparse-') as tmp:
        path = Path(tmp) / 'sparse.bin'
        with path.open('wb') as handle:
            handle.truncate(size)
            for index, offset in enumerate(marker_offsets(size)):
                handle.seek(offset)
                handle.write(MARKER + bytes([index]))
        digest: str = imohash.hashfile(str(path), hexdigest=True)
        return digest


def committed_images() -> list[str]:
    return [
        path.relative_to(ROOT).as_posix()
        for directory in IMAGE_DIRS
        for path in sorted((ROOT / directory).rglob('*'))
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]


def build() -> dict[str, Any]:
    synthetic = [{'size': n, 'hex': hex_of(pattern_bytes(n))} for n in SYNTHETIC_SIZES]
    tiled = []
    for name, copies in TILED:
        data = (ROOT / name).read_bytes() * copies
        tiled.append({'file': name, 'copies': copies, 'size': len(data), 'hex': hex_of(data)})
    sparse = [{'size': n, 'hex': sparse_hash(n)} for n in SPARSE_SIZES]
    files = []
    for name in committed_images():
        data = (ROOT / name).read_bytes()
        files.append({'file': name, 'size': len(data), 'hex': hex_of(data)})
    return {
        'source': f'imohash {imohash_version()} hashfileobject / hashfile, hexdigest',
        'threshold': THRESHOLD,
        'sample_size': SAMPLE,
        'pattern': PATTERN,
        'marker': MARKER.decode(),
        'synthetic': synthetic,
        'tiled': tiled,
        'sparse': sparse,
        'files': files,
    }


def imohash_version() -> str:
    from importlib.metadata import version  # only for the provenance line

    return version('imohash')


def render() -> str:
    return json.dumps(build(), indent=1) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='compare with the committed file')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding='utf-8') != text:
            sys.stderr.write(f'{args.out} is stale: run scripts/codegen/imohash_goldens.py\n')
            return 1
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding='utf-8')
    sys.stdout.write(f'wrote {args.out} ({len(text)} bytes)\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
