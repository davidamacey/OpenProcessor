#!/usr/bin/env python3
"""Fetch the public face proxies for real-data cluster evaluation.

    fetch_face_proxy.py lfw --out <dir> [--dry-run]
    fetch_face_proxy.py openimages --out <dir> --cache-dir <dir> [--n 1500] [--dry-run]
    fetch_face_proxy.py verify <dir>

``lfw``: Labeled Faces in the Wild (13,233 images, 5,749 identities) from the figshare copy
that scikit-learn uses (the original host no longer resolves). The archive hashes are pinned
below. LFW publishes no formal licence, so it is local only and nothing from it is committed.
``openimages``: V7 validation images with a "Human face" box, CC BY 2.0 images with CC BY 4.0
annotations, no identities. CPLFW is not fetched: its only distribution is a Google Drive or
Baidu link with no stated licence.

Each dataset directory gets ``manifest.json`` (every file: size and SHA-256), ``labels.jsonl``
and, for Open Images, ``ATTRIBUTION.csv``. ``--pin`` writes the small committed pin. Run
with ``python -I`` on the host; use ``nice`` for the downloads.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets import proxy_openimages as oi
from scripts.datasets._common import FetchError, download
from scripts.datasets.proxy_manifest import (
    Licence,
    build_manifest,
    file_entry,
    label_row,
    verify_dataset_dir,
    write_dataset,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fetch_face_proxy')

FIGSHARE = 'https://ndownloader.figshare.com/files/{}'
LFW_SOURCES = {
    'lfw.tgz': (
        FIGSHARE.format(5976018),
        '055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0',
    ),
    'pairs.txt': (
        FIGSHARE.format(5976006),
        'ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592',
    ),
    'pairsDevTrain.txt': (
        FIGSHARE.format(5976012),
        '1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa',
    ),
    'pairsDevTest.txt': (
        FIGSHARE.format(5976009),
        '7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c',
    ),
}
LFW_IMAGES = 13_233
LFW_PEOPLE = 5_749
LFW_LICENCE = Licence(
    name='no formal licence published',
    url='https://www.cs.umass.edu/~elm/papers/lfw.pdf',
    redistributable=False,
    note='Web-crawled news photographs; copyright stays with the original owners. '
    'Local evaluation only; aggregates are published only after owner review (plan O2).',
)
OI_LICENCE = Licence(
    name=f'images CC BY 2.0, annotations {oi.ANNOTATION_LICENCE}',
    url=oi.LICENCE_URL,
    redistributable=True,
    note='Attribution required; ATTRIBUTION.csv lists author and source per image.',
)
OI_MIN_AREA = 0.002


def _lfw_member_ok(name: str) -> bool:
    parts = name.split('/')
    return (
        len(parts) == 3
        and parts[0] == 'lfw'
        and parts[2].endswith('.jpg')
        and parts[2].startswith(parts[1] + '_')
    )


def extract_lfw(archive: Path, images_dir: Path) -> int:
    """Extract only ``lfw/<Person>/<Person>_NNNN.jpg`` members, stripping the ``lfw/`` prefix."""
    n = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if member.isdir():
                continue
            if not member.isreg() or not _lfw_member_ok(member.name):
                raise FetchError(f'{archive.name}: unexpected member {member.name!r}')
            dest = images_dir / member.name.removeprefix('lfw/')
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                raise FetchError(f'{archive.name}: cannot read {member.name!r}')
            with src, dest.open('wb') as out:
                out.write(src.read())
            n += 1
    return n


def cmd_lfw(args: argparse.Namespace) -> int:
    out: Path = args.out
    if args.dry_run:
        print(f'lfw: {len(LFW_SOURCES)} source files from figshare, expect {LFW_IMAGES} images')
        print(f'  identities {LFW_PEOPLE}, licence: {LFW_LICENCE.name} (local only)')
        return 0
    source = out / 'source'
    for name, (url, digest) in LFW_SOURCES.items():
        download(url, source / name, expected_sha256=digest, retries=3)
    images_dir = out / 'images'
    if images_dir.exists():
        raise FetchError(f'{images_dir} exists; verify it or remove it yourself')
    n = extract_lfw(source / 'lfw.tgz', images_dir)
    people = sorted(p.name for p in images_dir.iterdir())
    if n != LFW_IMAGES or len(people) != LFW_PEOPLE:
        raise FetchError(
            f'expected {LFW_IMAGES} images / {LFW_PEOPLE} people, got {n} / {len(people)}'
        )
    rels = [p.relative_to(out).as_posix() for p in sorted(out.rglob('*.*')) if p.is_file()]
    rels = [r for r in rels if r.startswith(('images/', 'source/'))]
    files = [file_entry(out, r) for r in rels]
    manifest = build_manifest('lfw', LFW_SOURCES['lfw.tgz'][0], LFW_LICENCE, files)
    labels = [
        label_row(r, 'face', ['face'], identity=r.split('/')[1])
        for r in rels
        if r.startswith('images/')
    ]
    write_dataset(out, manifest, labels, args.pin)
    logger.info('lfw: %d images, %d identities -> %s', n, len(people), out)
    return 0


def cmd_openimages(args: argparse.Namespace) -> int:
    cache: Path = args.cache_dir
    classes = download(oi.CLASSES_URL, cache / 'oidv7-class-descriptions-boxable.csv')
    wanted = oi.class_ids(classes.read_text(encoding='utf-8'), (oi.FACE_CLASS,))
    bbox = download(
        oi.BBOX_URL.format(split=args.split), cache / f'{args.split}-annotations-bbox.csv'
    )
    imgs = download(oi.IMAGES_URL.format(split=args.split), cache / f'{args.split}-images.csv')
    boxes = oi.qualifying_boxes(bbox.read_text(encoding='utf-8'), wanted, OI_MIN_AREA)
    eligible = oi.eligible_image_ids(imgs.read_text(encoding='utf-8'))
    selection = oi.select_images(boxes, eligible, args.n, args.seed)
    print(oi.summarize(selection))
    if args.dry_run:
        return 0
    got = oi.fetch_images(selection, args.split, args.out / 'images')
    oi.write_dataset(args.out, 'openimages_faces', OI_LICENCE, got, args.split, [], args.pin)
    logger.info('openimages faces: %d images -> %s', len(got), args.out)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    report = verify_dataset_dir(args.directory)
    for line in report.problems()[:20]:
        logger.error(line)
    logger.info('verify %s: %s', args.directory, 'ok' if report.ok else 'FAILED')
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    lfw = sub.add_parser('lfw')
    lfw.add_argument('--out', type=Path, required=True)
    lfw.add_argument('--pin', type=Path)
    lfw.add_argument('--dry-run', action='store_true')
    lfw.set_defaults(func=cmd_lfw)
    o = sub.add_parser('openimages')
    o.add_argument('--out', type=Path, required=True)
    o.add_argument('--cache-dir', type=Path, required=True)
    o.add_argument('--split', choices=oi.SPLITS, default='validation')
    o.add_argument('--n', type=int, default=1500)
    o.add_argument('--seed', type=int, default=20261029)
    o.add_argument('--pin', type=Path)
    o.add_argument('--dry-run', action='store_true')
    o.set_defaults(func=cmd_openimages)
    v = sub.add_parser('verify')
    v.add_argument('directory', type=Path)
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except FetchError as exc:
        logger.error('%s', exc)
        return 1


if __name__ == '__main__':
    sys.exit(main())
