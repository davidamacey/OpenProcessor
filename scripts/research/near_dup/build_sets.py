"""Constructed copy set for the near-duplicate prototypes (nd:P1, nd:P2), plus the Copydays reader.

A seeded sample of originals (COCO val2017 and Open Images validation images from local roots given as
arguments) and factor-isolated transforms of each. Nothing is downloaded and no image is committed: the
output directory and the manifest live under ``artifacts_local/`` or a scratch root.

Run ``build_sets.py --help``. Every dataset root is an argument; this file holds no private path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


# name -> (encoder format, saved extension)
TRANSFORMS = (
    'T-q60',
    'T-q30',
    'T-half',
    'T-quarter',
    'T-crop90',
    'T-crop80',
    'T-bright',
    'T-rot',
    'C-social',
    'CB-q60',
    'CB-half',
    'CB-crop90',
    'CB-bright',
)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
HEX64 = frozenset('0123456789abcdef')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pinned_hashes(manifest_dir: Path) -> set[str]:
    """Every 64-hex string found anywhere in the JSON manifests of ``manifest_dir``."""
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str) and len(node) == 64 and set(node) <= HEX64:
            found.add(node)

    for p in sorted(manifest_dir.glob('*.json')):
        walk(json.loads(p.read_text()))
    return found


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)


def sample_originals(
    sources: dict[str, list[Path]], per_source: int, seed: int, exclude: set[str]
) -> list[dict[str, str]]:
    """Seeded sample of ``per_source`` originals from each source, skipping any SHA-256 in ``exclude``."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for name in sorted(sources):
        files = sorted(sources[name])
        rng.shuffle(files)
        kept = 0
        for path in files:
            if kept == per_source:
                break
            digest = sha256_file(path)
            if digest in exclude:
                continue
            rows.append({'source': name, 'path': str(path), 'sha256': digest})
            kept += 1
    return rows


def _decode(path: str) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert('RGB')


def _jpeg(img: Image.Image, quality: int) -> tuple[bytes, str]:
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=quality)
    return buf.getvalue(), 'jpg'


def _png(img: Image.Image) -> tuple[bytes, str]:
    buf = io.BytesIO()
    img.save(buf, 'PNG', compress_level=1)
    return buf.getvalue(), 'png'


def _centre_crop(img: Image.Image, keep: float) -> Image.Image:
    w, h = img.size
    dx, dy = round(w * (1 - keep) / 2), round(h * (1 - keep) / 2)
    return img.crop((dx, dy, w - dx, h - dy))


def inscribed_rect(w: int, h: int, degrees: float) -> tuple[int, int]:
    """Largest axis-aligned rectangle inside a ``w`` x ``h`` rectangle rotated by ``degrees``."""
    angle = math.radians(abs(degrees))
    if w <= 0 or h <= 0:
        return 0, 0
    long_side, short_side = (w, h) if w >= h else (h, w)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if short_side <= 2.0 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short_side
        wr, hr = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return int(wr), int(hr)


def _rotate(img: Image.Image, degrees: float) -> Image.Image:
    w, h = img.size
    rotated = img.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=False)
    wr, hr = inscribed_rect(w, h, degrees)
    left, top = (w - wr) // 2, (h - hr) // 2
    return rotated.crop((left, top, left + wr, top + hr))


def _social(img: Image.Image) -> tuple[bytes, str]:
    w, h = img.size
    return _jpeg(img.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS), 75)


def _scaled(img: Image.Image, divisor: int) -> Image.Image:
    w, h = img.size
    return img.resize((max(1, w // divisor), max(1, h // divisor)), Image.Resampling.LANCZOS)


_SIMPLE = {
    'T-q60': lambda im: _jpeg(im, 60),
    'T-q30': lambda im: _jpeg(im, 30),
    'T-half': lambda im: _png(_scaled(im, 2)),
    'T-quarter': lambda im: _png(_scaled(im, 4)),
    'T-crop90': lambda im: _png(_centre_crop(im, 0.9)),
    'T-crop80': lambda im: _png(_centre_crop(im, 0.8)),
    'T-bright': lambda im: _png(ImageEnhance.Brightness(im).enhance(1.15)),
    'T-rot': lambda im: _png(_rotate(im, 3.0)),
    'C-social': _social,
}


def apply_transform(name: str, img: Image.Image) -> tuple[bytes, str]:
    """Encoded bytes and file extension of ``img`` under the transform ``name``."""
    if name in _SIMPLE:
        return _SIMPLE[name](img)
    if name.startswith('CB-'):
        return _baseline_variant(name, img)
    raise ValueError(f'unknown transform {name!r}')


def _baseline_variant(name: str, img: Image.Image) -> tuple[bytes, str]:
    """The four variants of ``scripts/bench/retrieval_eval_data.py``: each saved as JPEG q90."""
    w, h = img.size
    quality = 90
    if name == 'CB-q60':
        quality = 60
    elif name == 'CB-half':
        img = img.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS)
    elif name == 'CB-crop90':
        dx, dy = int(w * 0.05), int(h * 0.05)
        img = img.crop((dx, dy, w - dx, h - dy))
    elif name == 'CB-bright':
        img = ImageEnhance.Brightness(img).enhance(1.15)
    else:
        raise ValueError(name)
    return _jpeg(img, quality)


def _build_one(task: tuple[int, str, str, tuple[str, ...]]) -> list[dict[str, str | int]]:
    idx, src, out_dir, names = task
    img = _decode(src)
    rows: list[dict[str, str | int]] = []
    for name in names:
        data, ext = apply_transform(name, img)
        dest = Path(out_dir) / f'{idx:05d}__{name}.{ext}'
        dest.write_bytes(data)
        rows.append(
            {
                'family': idx,
                'transform': name,
                'path': str(dest),
                'sha256': hashlib.sha256(data).hexdigest(),
                'bytes': len(data),
            }
        )
    return rows


def build_copies(
    originals: list[dict[str, str]], out_dir: Path, names: tuple[str, ...], workers: int
) -> list[dict[str, str | int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(i, o['path'], str(out_dir), names) for i, o in enumerate(originals)]
    rows: list[dict[str, str | int]] = []
    if workers <= 1:
        for task in tasks:
            rows.extend(_build_one(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(_build_one, tasks, chunksize=8):
                rows.extend(part)
    return rows


def read_copydays(root: Path) -> list[dict[str, str | int]]:
    """Copydays items as pool rows. Files are ``XXXXYY.jpg``: ``XXXX`` names the image, ``YY`` the variant.

    ``root`` holds one directory per kind (``original``, ``strong``, optionally ``jpeg``, ``crop``).
    Families are numbered from the sorted image ids; an item whose id has no original is dropped.
    """
    originals = {p.stem[:4]: p for p in list_images(root / 'original')}
    family_of = {k: i for i, k in enumerate(sorted(originals))}
    rows: list[dict[str, str | int]] = [
        {'family': family_of[k], 'transform': 'orig', 'path': str(p), 'sha256': ''}
        for k, p in sorted(originals.items())
    ]
    for kind in ('strong', 'jpeg', 'crop'):
        kind_dir = root / kind
        if not kind_dir.is_dir():
            continue
        for path in list_images(kind_dir):
            key = path.stem[:4]
            if key in family_of and path.stem != originals[key].stem:
                rows.append(
                    {
                        'family': family_of[key],
                        'transform': f'copydays_{kind}',
                        'path': str(path),
                        'sha256': '',
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--coco-dir', type=Path, required=True, help='directory of COCO val2017 images')
    ap.add_argument(
        '--oi-dir',
        type=Path,
        action='append',
        required=True,
        help='Open Images image directory (repeatable)',
    )
    ap.add_argument('--manifest-dir', type=Path, required=True, help='scripts/datasets/manifests')
    ap.add_argument('--out-dir', type=Path, required=True, help='where copies are written')
    ap.add_argument('--manifest', type=Path, required=True, help='output manifest JSON')
    ap.add_argument('--per-source', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20261104)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--transforms', nargs='*', default=list(TRANSFORMS))
    args = ap.parse_args(argv)

    oi_files = [p for d in args.oi_dir for p in list_images(d)]
    originals = sample_originals(
        {'coco': list_images(args.coco_dir), 'openimages': oi_files},
        args.per_source,
        args.seed,
        pinned_hashes(args.manifest_dir),
    )
    rows = build_copies(originals, args.out_dir, tuple(args.transforms), args.workers)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                'seed': args.seed,
                'originals': originals,
                'copies': rows,
                'transforms': args.transforms,
            },
            indent=1,
        )
    )
    sys.stdout.write(f'{len(originals)} originals, {len(rows)} copies -> {args.manifest}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
