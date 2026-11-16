#!/usr/bin/env python3
"""Goldens for the native frame engine's colour step (plan 17 section 5 N1a).

Runs the rule both decode routes follow, ``src/utils/icc_srgb.py`` (Little CMS through Pillow
13.0.0), on public and generated ICC profiles and writes ``native/frame_engine/tests/golden/
colour.json``: per profile, its bytes (base64, or a fixture's path), the verdict of its own tags
(``tag_verdict``), the largest colorant distance from sRGB's that verdict starts from, the kind
(``classify``) and, for ``convert``, the SHA-256 of the 48 MiB conversion table. The Rust port must give the same verdicts and a table equal to the byte.

The profiles: the three CC0 / MIT fixtures of ``tests/fixtures/icc/`` (named by path) and
matrix/TRC profiles written here byte by byte (ICC.1:2022 sections 7 and 10: header, tag table, ``XYZType``,
``curveType``, ``parametricCurveType``), each aimed at one branch of the rule, plus a grey
profile and bytes that are no profile. ``--corpus DIR`` writes the same record for every
``*.icc`` in a directory to ``--out`` instead (the COCO profile corpus, kept out of the tree).

The data is committed; the Rust test reads it. This script is deleted with the Python engine at
the switch (plan 17 N3). ``--check`` regenerates into memory and compares with the file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import io
import json
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import ImageCms


if TYPE_CHECKING:
    from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def _rule() -> ModuleType:
    """``src.utils.icc_srgb``, with the repository made importable."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module('src.utils.icc_srgb')


OUT = ROOT / 'native' / 'frame_engine' / 'tests' / 'golden' / 'colour.json'
FIXTURES = ROOT / 'tests' / 'fixtures' / 'icc'
D50 = (0.9642, 1.0, 0.8249)
SRGB_PARA = (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045)


def _s15(v: float) -> bytes:
    return struct.pack('>i', round(v * 65536))


def _xyz(x: float, y: float, z: float) -> bytes:
    return b'XYZ \0\0\0\0' + _s15(x) + _s15(y) + _s15(z)


def _gamma(g: float) -> bytes:
    return b'curv\0\0\0\0' + struct.pack('>IH', 1, round(g * 256)) + b'\0\0'


def _table(values: list[float]) -> bytes:
    body = b''.join(struct.pack('>H', round(v * 65535)) for v in values)
    pad = b'\0' * (-len(body) % 4)
    return b'curv\0\0\0\0' + struct.pack('>I', len(values)) + body + pad


def _para(kind: int, params: tuple[float, ...]) -> bytes:
    return b'para\0\0\0\0' + struct.pack('>H', kind) + b'\0\0' + b''.join(map(_s15, params))


def _profile(space: bytes, tags: list[tuple[bytes, bytes]]) -> bytes:
    """A version 2.1 display profile: the 128-byte header, the tag table, the tag data."""
    table = struct.pack('>I', len(tags))
    data = b''
    offset = 128 + 4 + 12 * len(tags)
    for sig, body in tags:
        table += sig + struct.pack('>II', offset + len(data), len(body))
        data += body + b'\0' * (-len(body) % 4)
    size = offset + len(data)
    header = (
        struct.pack('>I', size) + b'lcms' + bytes([2, 0x10, 0, 0]) + b'mntr' + space + b'XYZ '
        + bytes(12) + b'acsp' + bytes(4) + bytes(4) + bytes(4) + bytes(4) + bytes(8)
        + bytes(4) + _s15(D50[0]) + _s15(D50[1]) + _s15(D50[2]) + bytes(4) + bytes(16)
    )  # fmt: skip
    header += bytes(128 - len(header))
    return header + table + data


def _srgb_colorants() -> list[tuple[float, float, float]]:
    prof = ImageCms.createProfile('sRGB')
    found = (prof.red_colorant, prof.green_colorant, prof.blue_colorant)
    return [c[0] for c in found if c is not None]


def _rgb(colorants: list[tuple[float, float, float]], curve: bytes) -> bytes:
    sigs = (b'rXYZ', b'gXYZ', b'bXYZ')
    tags = [(b'wtpt', _xyz(*D50))]
    tags += [(s, _xyz(*c)) for s, c in zip(sigs, colorants, strict=True)]
    tags += [(s, curve) for s in (b'rTRC', b'gTRC', b'bTRC')]
    return _profile(b'RGB ', tags)


def _srgb_curve(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def generated() -> dict[str, bytes]:
    srgb = _srgb_colorants()
    shifted = [(srgb[0][0] + 0.005, srgb[0][1], srgb[0][2]), srgb[1], srgb[2]]
    nudged = [(srgb[0][0] + 0.0022, srgb[0][1], srgb[0][2]), srgb[1], srgb[2]]
    adobe = [(0.6097, 0.3111, 0.0195), (0.2053, 0.6257, 0.0609), (0.1492, 0.0632, 0.7446)]
    grid = [_srgb_curve(k / 1023) for k in range(1024)]
    return {
        'gen-srgb-para': _rgb(srgb, _para(3, SRGB_PARA)),
        'gen-srgb-table1024': _rgb(srgb, _table(grid)),
        'gen-srgb-gamma22': _rgb(srgb, _gamma(2.2)),
        'gen-srgb-linear': _rgb(srgb, b'curv\0\0\0\0' + struct.pack('>I', 0)),
        'gen-srgb-para0-24': _rgb(srgb, _para(0, (2.4,))),
        'gen-srgb-para4': _rgb(srgb, _para(4, (*SRGB_PARA, 0.0, 0.0))),
        'gen-shifted-red-para': _rgb(shifted, _para(3, SRGB_PARA)),
        'gen-nudged-red-para': _rgb(nudged, _para(3, SRGB_PARA)),
        'gen-srgb-para-243': _rgb(srgb, _para(3, (2.43, *SRGB_PARA[1:]))),
        'gen-adobe-gamma22': _rgb(adobe, _gamma(563 / 256)),
        'gen-adobe-para1': _rgb(adobe, _para(1, (2.2, 1.0, 0.0))),
        'gen-grey': _profile(b'GRAY', [(b'wtpt', _xyz(*D50)), (b'kTRC', _gamma(2.2))]),
        'gen-not-a-profile': b'\0' * 64 + b'not an ICC profile' * 4,
    }


def _distance(icc: bytes) -> float | None:
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc)).profile
    except (OSError, ImageCms.PyCMSError, ValueError, TypeError):
        return None
    if prof.xcolor_space != 'RGB ':
        return None
    if not prof.is_matrix_shaper:
        return None
    found = (prof.red_colorant, prof.green_colorant, prof.blue_colorant)
    if any(c is None for c in found):
        return None
    mine = np.array([c[0] for c in found if c is not None])
    return float(np.abs(mine - np.array(_srgb_colorants())).max())


def record(name: str, icc: bytes, *, inline: bool) -> dict[str, Any]:
    icc_srgb = _rule()
    kind, table = icc_srgb.classify(icc)
    out: dict[str, Any] = {'name': name}
    if inline:
        out['icc'] = base64.b64encode(icc).decode()
    else:
        out['sha256'] = hashlib.sha256(icc).hexdigest()
    out.update({
        'tag_verdict': icc_srgb.tag_verdict(icc),
        'colorant_distance': _distance(icc),
        'kind': kind,
        'table_sha256': None if table is None else hashlib.sha256(table.tobytes()).hexdigest(),
    })  # fmt: skip
    return out


def goldens() -> dict[str, Any]:
    profiles = [
        {
            'path': str(p.relative_to(ROOT)),
            **record(f'fixture-{p.stem}', p.read_bytes(), inline=False),
        }
        for p in sorted(FIXTURES.glob('*.icc'))
    ]
    profiles += [record(name, icc, inline=True) for name, icc in generated().items()]
    return {
        'generated_by': 'scripts/codegen/frame_engine_colour_goldens.py',
        'littlecms': ImageCms.core.littlecms_version,
        'profiles': profiles,
    }


def corpus(directory: Path) -> dict[str, Any]:
    return {
        'littlecms': ImageCms.core.littlecms_version,
        'profiles': [
            record(p.stem, p.read_bytes(), inline=False) for p in sorted(directory.glob('*.icc'))
        ],
    }


def render() -> dict[str, str]:
    """The file this script writes, by name in the golden directory."""
    return {OUT.name: json.dumps(goldens(), indent=1) + '\n'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='compare with the committed file')
    parser.add_argument('--corpus', type=Path, help='a directory of .icc files to record instead')
    parser.add_argument('--out', type=Path, help='where --corpus writes its record')
    args = parser.parse_args()
    if args.corpus is not None:
        if args.out is None:
            parser.error('--corpus needs --out')
        args.out.write_text(json.dumps(corpus(args.corpus), indent=1) + '\n')
        print(f'wrote {args.out}')
        return 0
    text = render()[OUT.name]
    if args.check:
        same = OUT.read_text() == text
        print('colour goldens up to date' if same else f'{OUT} is stale')
        return 0 if same else 1
    OUT.write_text(text)
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
