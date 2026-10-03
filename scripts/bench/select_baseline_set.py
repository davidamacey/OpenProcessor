"""Select a fixed, seeded baseline image set and write a checksummed manifest.

Two modes, one manifest format:

* ``local``: draw N decodable JPEGs from one or more root directories
  (min side >= 320 px, <= 40 MB). Candidates are taken round-robin across the
  first-level subfolders of every root, so an archive sorted by day or event
  is sampled evenly instead of from whichever folder sorts first.
* ``coco``: draw N images from a COCO 2017 annotation file (a local file or a
  URL), from a local image directory or by downloading, and record every
  image's license in a CSV sidecar so the public set is reproducible and
  license-checked.

The manifest is a text file: ``#`` header lines (seed, count, bytes,
per-source counts, date, sha256) and then one image path per line. The sha256
covers the path lines only, so the header date never changes the checksum.
Paths are arguments; nothing here has a default location.

    select_baseline_set.py local --root DIR [--root DIR ...] --count 4000 --out manifest.txt
    select_baseline_set.py coco --annotations FILE_OR_URL --images DIR_OR_URL \\
        --count 4000 --out manifest.txt [--download-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets._common import download, seeded_sample


MIN_SIDE = 320
MAX_BYTES = 40 * 1024 * 1024
JPEG_SUFFIXES = frozenset({'.jpg', '.jpeg'})
COCO_INSTANCES_MEMBER = 'annotations/instances_val2017.json'
LICENSE_COLUMNS = ('path', 'coco_id', 'license_id', 'license_name', 'license_url')


class ManifestError(ValueError):
    """The manifest is unreadable or its checksum does not match its paths."""


@dataclass(frozen=True)
class Candidate:
    path: Path
    size: int
    source: str
    coco_id: int | None = None
    license_id: int | None = None
    license_name: str = ''
    license_url: str = ''


def body_sha256(paths: Iterable[str]) -> str:
    """The one checksum of a manifest: sha256 of its newline-joined path lines."""
    return hashlib.sha256(''.join(f'{p}\n' for p in paths).encode('utf-8')).hexdigest()


def _decodable(path: Path, min_side: int) -> bool:
    try:
        with Image.open(path) as img:
            if min(img.size) < min_side:
                return False
            img.verify()
        return True
    except (OSError, SyntaxError, ValueError):
        return False


def _enumerate_groups(root: Path) -> dict[str, list[Path]]:
    """First-level subfolder name -> sorted JPEG paths below it ('' = files in the root)."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in JPEG_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(root)
        groups.setdefault(rel.parts[0] if len(rel.parts) > 1 else '', []).append(path)
    return groups


def select_images(
    roots: Sequence[Path],
    count: int,
    *,
    seed: int = 42,
    min_side: int = MIN_SIDE,
    max_bytes: int = MAX_BYTES,
) -> list[Candidate]:
    """Seeded round-robin draw of up to ``count`` valid JPEGs across ``roots``.

    Every (root, first-level subfolder) is a queue in a seeded shuffle order;
    the queues are visited in sorted order, one valid image per visit, until
    ``count`` is reached or every queue is empty. Validity is checked lazily,
    so a 40,000-image archive is not decoded to pick 4,000.
    """
    queues: list[tuple[str, list[Path]]] = []
    for root in roots:
        for name, files in sorted(_enumerate_groups(root).items()):
            order = list(files)
            random.Random(f'{seed}:{root.name}:{name}').shuffle(order)
            queues.append((root.name, order))
    picked: list[Candidate] = []
    cursors = [0] * len(queues)
    while len(picked) < count:
        progressed = False
        for i, (source, order) in enumerate(queues):
            while cursors[i] < len(order):
                path = order[cursors[i]]
                cursors[i] += 1
                size = path.stat().st_size
                if size <= max_bytes and _decodable(path, min_side):
                    picked.append(Candidate(path, size, source))
                    progressed = True
                    break
            if len(picked) >= count:
                break
        if not progressed:
            break
    return picked


def per_source_counts(picked: Iterable[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cand in picked:
        counts[cand.source] = counts.get(cand.source, 0) + 1
    return counts


def write_manifest(
    out: Path, picked: Sequence[Candidate], *, seed: int, today: str | None = None
) -> str:
    """Write the manifest and return its sha256 (also recorded in the header)."""
    paths = [str(c.path) for c in picked]
    digest = body_sha256(paths)
    sources = ','.join(f'{k}={v}' for k, v in sorted(per_source_counts(picked).items()))
    header = [
        f'# seed: {seed}',
        f'# count: {len(picked)}',
        f'# bytes: {sum(c.size for c in picked)}',
        f'# sources: {sources}',
        f'# date: {today or datetime.now(UTC).date().isoformat()}',
        f'# sha256: {digest}',
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(header + paths) + '\n', encoding='utf-8')
    return digest


def read_manifest(path: Path) -> list[str]:
    """The manifest's image paths; raises :class:`ManifestError` when the checksum disagrees."""
    lines = path.read_text(encoding='utf-8').splitlines()
    paths = [line for line in lines if line and not line.startswith('#')]
    recorded = next((ln.split(': ', 1)[1] for ln in lines if ln.startswith('# sha256: ')), None)
    if recorded is not None and recorded != body_sha256(paths):
        raise ManifestError(f'{path}: checksum does not match the listed paths')
    return paths


def _load_json_or_zip(
    source: str, cache_dir: Path, downloader: Callable[..., Path]
) -> dict[str, Any]:
    if source.startswith(('http://', 'https://')):
        local = downloader(source, cache_dir / Path(source).name)
    else:
        local = Path(source)
    if local.suffix == '.zip':
        with zipfile.ZipFile(local) as zf:
            return json.loads(zf.read(COCO_INSTANCES_MEMBER))
    return json.loads(local.read_text(encoding='utf-8'))


def select_coco(
    annotations: Path | dict[str, Any],
    images_dir: Path,
    count: int,
    *,
    seed: int = 42,
    allowed_licenses: set[str] | None = None,
    image_url_template: str | None = None,
    downloader: Callable[..., Path] = download,
    min_side: int = MIN_SIDE,
) -> list[Candidate]:
    """Seeded uniform draw from a COCO annotation file, with each image's license.

    With ``image_url_template`` (containing ``{file_name}``) the chosen images
    are downloaded into ``images_dir``; without it only images already on disk
    in ``images_dir`` are eligible.
    """
    doc = (
        annotations
        if isinstance(annotations, dict)
        else json.loads(annotations.read_text(encoding='utf-8'))
    )
    licenses = {lic['id']: lic for lic in doc.get('licenses', [])}
    pool = []
    for img in sorted(doc['images'], key=lambda i: i['id']):
        lic = licenses.get(img.get('license'), {})
        if allowed_licenses is not None and lic.get('name') not in allowed_licenses:
            continue
        if min(img.get('width', 0), img.get('height', 0)) < min_side:
            continue
        if image_url_template is None and not (images_dir / Path(img['file_name']).name).is_file():
            continue
        pool.append((img, lic))
    picked = []
    for img, lic in seeded_sample(pool, count, seed):
        dest = images_dir / Path(img['file_name']).name
        if image_url_template is not None:
            downloader(image_url_template.format(file_name=img['file_name']), dest)
        picked.append(
            Candidate(
                dest,
                dest.stat().st_size,
                'coco',
                coco_id=img['id'],
                license_id=lic.get('id'),
                license_name=lic.get('name', ''),
                license_url=lic.get('url', ''),
            )
        )
    return picked


def write_license_sidecar(manifest: Path, picked: Sequence[Candidate]) -> Path:
    """``<manifest>.licenses.csv``: one row per manifest line, with the COCO license."""
    sidecar = manifest.with_name(manifest.name + '.licenses.csv')
    with sidecar.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LICENSE_COLUMNS)
        writer.writeheader()
        for cand in picked:
            writer.writerow(
                {
                    'path': str(cand.path),
                    'coco_id': cand.coco_id,
                    'license_id': cand.license_id,
                    'license_name': cand.license_name,
                    'license_url': cand.license_url,
                }
            )
    return sidecar


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('local', 'coco'):
        p = sub.add_parser(name)
        p.add_argument('--count', type=int, required=True)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--out', type=Path, required=True, help='manifest path to write')
    local = sub.choices['local']
    local.add_argument('--root', type=Path, action='append', required=True)
    local.add_argument('--min-side', type=int, default=MIN_SIDE)
    local.add_argument('--max-mb', type=int, default=MAX_BYTES // (1024 * 1024))
    coco = sub.choices['coco']
    coco.add_argument(
        '--annotations', required=True, help='instances_*.json, a .zip, or a URL of either'
    )
    coco.add_argument(
        '--images', required=True, help='local image dir, or a URL (base or with {file_name})'
    )
    coco.add_argument('--download-dir', type=Path, help='where URL images/annotations are saved')
    coco.add_argument(
        '--licenses', nargs='*', help='keep only these license names (default: all, recorded)'
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == 'local':
        picked = select_images(
            args.root,
            args.count,
            seed=args.seed,
            min_side=args.min_side,
            max_bytes=args.max_mb * 1024 * 1024,
        )
    else:
        is_url = args.images.startswith(('http://', 'https://'))
        if (
            is_url or args.annotations.startswith(('http://', 'https://'))
        ) and args.download_dir is None:
            sys.exit('--download-dir is required when --annotations or --images is a URL')
        doc = _load_json_or_zip(args.annotations, args.download_dir or Path(), download)
        template = None
        if is_url:
            template = (
                args.images
                if '{file_name}' in args.images
                else args.images.rstrip('/') + '/{file_name}'
            )
        picked = select_coco(
            doc,
            args.download_dir if is_url else Path(args.images),
            args.count,
            seed=args.seed,
            allowed_licenses=set(args.licenses) if args.licenses else None,
            image_url_template=template,
        )
    digest = write_manifest(args.out, picked, seed=args.seed)
    if args.mode == 'coco':
        write_license_sidecar(args.out, picked)
    print(f'{len(picked)} images, sha256 {digest}, manifest {args.out}')
    if len(picked) < args.count:
        print(f'warning: only {len(picked)} of {args.count} images were eligible', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
