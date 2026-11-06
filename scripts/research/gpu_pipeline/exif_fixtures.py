#!/usr/bin/env python3
"""Generate the decoder-coverage fixtures of gpu:PR-1 from public images.

Orientation tags 1-8, progressive, truncated, CMYK, grey, 4:4:4 / 4:2:2 / 4:2:0 and restart
markers. Inputs are JPEG files given as arguments; nothing private is read.

    exif_fixtures.py --src-dir DIR --out DIR [--n-sources 3]
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image


ORIENTATION_TAG = 0x0112


def _jpeg(img: Image.Image, **kw: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=92, **kw)
    return buf.getvalue()


def make_fixtures(src: Image.Image) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    base = src.convert('RGB')
    for tag in range(1, 9):
        exif = Image.Exif()
        exif[ORIENTATION_TAG] = tag
        out[f'orient{tag}'] = _jpeg(base, exif=exif)
    plain = _jpeg(base)
    out['baseline'] = plain
    out['progressive'] = _jpeg(base, progressive=True)
    out['truncated'] = plain[: int(len(plain) * 0.6)]
    out['cmyk'] = _jpeg(base.convert('CMYK'))
    out['grey'] = _jpeg(base.convert('L'))
    for name, sub in (('444', 0), ('422', 1), ('420', 2)):
        out[f'chroma{name}'] = _jpeg(base, subsampling=sub)
    out['restart'] = _jpeg(base, restart_marker_blocks=4)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-sources', type=int, default=3)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(sorted(args.src_dir.glob('*.jpg'))[: args.n_sources]):
        with Image.open(path) as img:
            img.load()
            # tags are tested on a non-square frame so a wrong rotation changes the shape
            for name, data in make_fixtures(img).items():
                (args.out / f'src{i}_{name}.jpg').write_bytes(data)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
