#!/usr/bin/env python3
"""Unpack a downloaded labelled benchmark into ``images/`` + ``labels.jsonl`` + ``manifest.json``.

    prepare_labelled_benchmark.py prepare <name> <dir>
    prepare_labelled_benchmark.py verify <dir> [--hash]

``<name>`` is one of ``stanford_cars``, ``stanford_online_products``, ``cub200_2011``,
``vmmrdb``, ``calfw``, ``cplfw``. ``<dir>`` already holds the download (``parquet/`` for the
two parquet mirrors, the archive for the others, see ``labelled_benchmarks.md`` for the source
URLs); the archive or parquet stays in place, nothing outside ``<dir>`` is touched.

``verify`` re-reads ``manifest.json`` and checks that the files on disk are exactly the files it
lists (count and size, SHA-256 with ``--hash``) and that every ``labels.jsonl`` row names an
existing file. ``pyarrow`` (parquet sets only) is imported when needed. Run with ``python -I``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.datasets._common import sha256_file  # noqa: E402  needs the sys.path entry above


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

FETCHED = '2026-10-29'
CARS_CLASSES = 196
CARS_NAMES_URL = 'https://huggingface.co/api/datasets/tanganke/stanford_cars'
CUB_MD5 = '97eceeb196236b17998738112f37df78'
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')
SKIP_IN_MANIFEST = frozenset({'manifest.json', 'dl.log', 'DOWNLOAD_DONE.marker'})

META = {
    'stanford_cars': {
        'source': 'https://huggingface.co/datasets/tanganke/stanford_cars (community parquet mirror of Stanford Cars, Krause et al. 2013)',
        'licence': 'no licence stated on the mirror; original Stanford Cars: non-commercial research (unverified). Local evaluation only, do not redistribute.',
    },
    'stanford_online_products': {
        'source': 'https://huggingface.co/datasets/nyris/stanford-online-products-v1 (parquet of the original Stanford_Online_Products.zip)',
        'licence': 'MIT (LICENSE file of the original archive, Stanford, 2016). Item photos come from a marketplace and carry their sellers rights.',
    },
    'cub200_2011': {
        'source': f'https://data.caltech.edu/records/65de6-vp158, CUB_200_2011.tgz, MD5 {CUB_MD5}',
        'licence': 'the CaltechDATA record lists CC-BY; images are Flickr photos, see the per-image terms in the archive README',
    },
    'vmmrdb': {
        'source': 'https://github.com/faezetta/VMMRdb (zip)',
        'licence': 'MIT (repo LICENSE, Tafazzoli et al. 2017). Photos come from classified ads; labels are the ad make, model and year and are noisy.',
    },
    'calfw': {
        'source': 'http://whdeng.cn/CALFW/ (official zip)',
        'licence': 'none stated; derived from LFW news photographs. Local evaluation only.',
    },
    'cplfw': {
        'source': 'http://www.whdeng.cn/CPLFW/ (official zip)',
        'licence': 'none stated; derived from LFW news photographs. Local evaluation only.',
    },
}


class PrepareError(RuntimeError):
    """An archive member or dataset field that must stop the preparation."""


def safe(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')


def check_member(name: str) -> None:
    """Refuse an archive member or dataset path that could leave the target directory."""
    parts = Path(name).parts
    if name.startswith('/') or '..' in parts or '\\' in name:
        raise PrepareError(f'unsafe archive member {name!r}')


def extract_zip(zpath: Path, dest: Path, keep: Callable[[str], bool] = lambda _name: True) -> int:
    n = 0
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            raise PrepareError(f'corrupt member {bad}')
        for info in z.infolist():
            if info.is_dir() or not keep(info.filename):
                continue
            check_member(info.filename)
            out = dest / info.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, out.open('wb') as dst:
                while block := src.read(1 << 20):
                    dst.write(block)
            n += 1
    return n


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')


def listed_files(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob('*')) if p.is_file() and p.name not in SKIP_IN_MANIFEST]


def write_manifest(name: str, root: Path, extra: dict[str, Any]) -> int:
    files = [
        {'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
        for p in listed_files(root)
    ]
    manifest = {
        'name': name,
        'fetched': FETCHED,
        **META[name],
        **extra,
        'file_count': len(files),
        'files': files,
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return len(files)


# parquet sets -------------------------------------------------------------------------------


def read_parquet(root: Path, split: str) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for pf in sorted((root / 'parquet').glob(f'{split}-*.parquet')):
        yield from pq.read_table(pf).to_pylist()


def fetch_cars_class_names() -> list[str]:
    with urllib.request.urlopen(CARS_NAMES_URL, timeout=30) as resp:  # nosec B310  fixed https URL
        info = json.load(resp)
    names = info['cardData']['dataset_info']['features'][1]['dtype']['class_label']['names']
    return [names[str(i)] for i in range(CARS_CLASSES)]


def cars_rows(
    root: Path,
    names: list[str],
    records: Callable[[str], Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Write the CARS196 images and return their label rows.

    The DML zero-shot split is by class: classes 0 to 97 are ``train``, 98 to 195 ``test``,
    independent of the mirror's own image split (kept as ``orig_split``).
    """
    rows: list[dict[str, Any]] = []
    for split in ('train', 'test'):
        for r in records(split):
            label = int(r['label'])
            rel = f'images/{split}/{label:03d}_{safe(names[label])}/{len(rows):05d}.jpg'
            out = root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(r['image']['bytes'])
            rows.append(
                {
                    'path': rel,
                    'class_id': label,
                    'class_name': names[label],
                    'orig_split': split,
                    'dml_split': 'train' if label < CARS_CLASSES // 2 else 'test',
                }
            )
    return rows


def sop_rows(
    root: Path, records: Callable[[str], Iterable[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ('train', 'test'):
        for r in records(split):
            orig = r['image_path']
            check_member(orig)
            rel = 'images/' + safe(orig.replace('/', '__'))[:200]
            out = root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(r['image']['bytes'])
            rows.append(
                {
                    'path': rel,
                    'orig_path': orig,
                    'item_id': r['item_id'],
                    'category': r['category'],
                    'split': split,
                }
            )
    return rows


# archive sets -------------------------------------------------------------------------------


def md5_file(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _id_map(path: Path, cast: type = str) -> dict[int, Any]:
    return {int(a): cast(b) for a, b in (ln.split() for ln in path.read_text().splitlines())}


def cub_rows(base: Path) -> list[dict[str, Any]]:
    """Label rows from the extracted ``CUB_200_2011`` directory (1-based class ids)."""
    cls, imgs = _id_map(base / 'classes.txt'), _id_map(base / 'images.txt')
    lab = _id_map(base / 'image_class_labels.txt', int)
    is_train = _id_map(base / 'train_test_split.txt', int)
    return [
        {
            'path': f'extracted/CUB_200_2011/images/{imgs[i]}',
            'class_id': lab[i],
            'class_name': cls[lab[i]],
            'orig_split': 'train' if is_train[i] else 'test',
            'dml_split': 'train' if lab[i] <= 100 else 'test',
        }
        for i in sorted(imgs)
    ]


def vmmr_row(path: Path, root: Path) -> dict[str, Any]:
    """A VMMRdb row: the folder is ``make_model_year`` and the model may itself contain ``_``."""
    cname = path.parent.name
    parts = cname.split('_')
    return {
        'path': str(path.relative_to(root)),
        'class_name': cname,
        'make': parts[0],
        'model': '_'.join(parts[1:-1]),
        'year': parts[-1],
    }


def lfw_variant_rows(images_dir: Path, root: Path) -> list[dict[str, Any]]:
    return [
        {'path': str(p.relative_to(root)), 'identity': re.sub(r'_\d+$', '', p.stem)}
        for p in sorted(images_dir.glob('*.jpg'))
    ]


# drivers ------------------------------------------------------------------------------------


def prepare(name: str, root: Path) -> int:
    if name not in META:
        raise PrepareError(f'unknown benchmark {name!r}; one of {sorted(META)}')
    if name == 'stanford_cars':
        rows = cars_rows(root, fetch_cars_class_names(), lambda s: read_parquet(root, s))
        extra: dict[str, Any] = {
            'classes': CARS_CLASSES,
            'protocol': 'DML zero-shot: classes 0-97 train, 98-195 test; evaluate on all images of the test classes (8,131)',
        }
    elif name == 'stanford_online_products':
        rows = sop_rows(root, lambda s: read_parquet(root, s))
        extra = {
            'classes_per_split': {
                s: len({r['item_id'] for r in rows if r['split'] == s}) for s in ('train', 'test')
            },
            'protocol': 'train = first 11,318 products (59,551 images), test = remaining 11,316 products (60,502 images); query = gallery = test',
        }
    elif name == 'cub200_2011':
        tgz = root / 'CUB_200_2011.tgz'
        if md5_file(tgz) != CUB_MD5:
            raise PrepareError('CUB archive MD5 does not match the published value')
        out = root / 'extracted'
        out.mkdir(exist_ok=True)
        with tarfile.open(tgz) as t:
            for m in t:
                check_member(m.name)
                if m.isfile():
                    t.extract(m, out, filter='data')
        rows = cub_rows(out / 'CUB_200_2011')
        extra = {
            'classes': 200,
            'archive_md5_verified': True,
            'protocol': 'DML zero-shot: classes 1-100 train (5,864 images), 101-200 test (5,924)',
        }
    elif name == 'vmmrdb':
        out = root / 'extracted'
        n = extract_zip(root / 'VMMRdb.zip', out, lambda x: x.lower().endswith(IMAGE_SUFFIXES))
        rows = [vmmr_row(p, root) for p in sorted(out.rglob('*')) if p.is_file()]
        extra = {'classes': len({r['class_name'] for r in rows}), 'extracted_members': n}
    else:
        n = extract_zip(root / f'{name}.zip', root / 'extracted')
        rows = lfw_variant_rows(root / 'extracted' / name / 'aligned images', root)
        extra = {
            'identities': len({r['identity'] for r in rows}),
            'extracted_members': n,
            'protocol': f'6,000 pairs, 10-fold verification accuracy; pairs file pairs_{name.upper()}.txt',
        }
    write_jsonl(rows, root / 'labels.jsonl')
    write_manifest(name, root, {'images': len(rows), **extra})
    return len(rows)


def verify(root: Path, *, check_hash: bool = False) -> list[str]:
    """Problems found between ``manifest.json``, ``labels.jsonl`` and the files on disk (empty = ok)."""
    manifest = json.loads((root / 'manifest.json').read_text())
    listed = {f['path']: f for f in manifest['files']}
    problems: list[str] = []
    if manifest['file_count'] != len(listed):
        problems.append(f'file_count {manifest["file_count"]} != {len(listed)} listed files')
    on_disk = {str(p.relative_to(root)) for p in listed_files(root)}
    problems += [f'missing: {p}' for p in sorted(set(listed) - on_disk)[:20]]
    problems += [f'unlisted: {p}' for p in sorted(on_disk - set(listed))[:20]]
    for rel in sorted(set(listed) & on_disk):
        f = listed[rel]
        p = root / rel
        if p.stat().st_size != f['bytes']:
            problems.append(f'size differs: {rel}')
        elif check_hash and sha256_file(p) != f['sha256']:
            problems.append(f'sha256 differs: {rel}')
    rows = [json.loads(x) for x in (root / 'labels.jsonl').read_text().splitlines() if x]
    if manifest.get('images') is not None and manifest['images'] != len(rows):
        problems.append(f'manifest images {manifest["images"]} != {len(rows)} label rows')
    problems += [f'label row without file: {r["path"]}' for r in rows if r['path'] not in listed][
        :20
    ]
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('name')
    p.add_argument('dir', type=Path)
    v = sub.add_parser('verify')
    v.add_argument('dir', type=Path)
    v.add_argument('--hash', action='store_true')
    args = ap.parse_args()
    if args.cmd == 'prepare':
        print(f'{args.name}: {prepare(args.name, args.dir)} images')
        return 0
    problems = verify(args.dir, check_hash=args.hash)
    print('\n'.join(problems) if problems else 'ok')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
