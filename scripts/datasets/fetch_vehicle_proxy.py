#!/usr/bin/env python3
"""Fetch or index the vehicle make/model proxies for real-data cluster evaluation.

    fetch_vehicle_proxy.py commons --out <dir> --categories <json> --manifest <json> [--dry-run]
    fetch_vehicle_proxy.py openimages --out <dir> --cache-dir <dir> [--per-class 200] [--dry-run]
    fetch_vehicle_proxy.py stanford --root <dir>
    fetch_vehicle_proxy.py compcars --root <dir> --make-names <txt> --model-names <txt>
    fetch_vehicle_proxy.py verify <dir>

``commons``: set C, per-model Wikimedia Commons categories, licence recorded per file; the
only vehicle set whose images may appear in committed docs. The full manifest (with author and
licence per file) is committed; a re-run with an existing ``--manifest`` downloads exactly the
pinned files.
``openimages``: V7 validation vehicle boxes (CC BY 2.0 images, CC BY 4.0 annotations).
``stanford`` and ``compcars`` only index a copy the operator already holds, in place: the
dataset directory itself receives ``manifest.json`` and ``labels.jsonl``. Stanford Cars has no
working official download (the host is gone and the torchvision download is broken) and the
Kaggle mirrors give no licence; CompCars requires a signed release agreement. Both stay local.
Stanford layout: ``<root>/**/<class name>/*.jpg``. CompCars layout: ``<root>/image/<make_id>/
<model_id>/<year>/*.jpg`` with ``<root>/label/...txt``; the name files list one name per line,
line N for id N.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets import proxy_commons as commons, proxy_openimages as oi
from scripts.datasets._common import FetchError, download, write_csv, write_json
from scripts.datasets.proxy_manifest import (
    Licence,
    build_manifest,
    file_entry,
    label_row,
    load_manifest,
    verify_dataset_dir,
    verify_manifest,
    write_dataset,
)
from scripts.datasets.vehicle_labels import (
    compcars_names,
    parse_compcars_image,
    parse_compcars_view,
    parse_stanford_class,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fetch_vehicle_proxy')

COMMONS_LICENCE = Licence(
    name='per file: CC0, public domain, CC BY, CC BY-SA',
    url='https://commons.wikimedia.org/wiki/Commons:Licensing',
    redistributable=True,
    note='Attribution and share-alike per file; the manifest lists author, licence and the '
    'description page for every image. Labels come from the category and are noisy.',
)
OI_LICENCE = Licence(
    name=f'images CC BY 2.0, annotations {oi.ANNOTATION_LICENCE}',
    url=oi.LICENCE_URL,
    redistributable=True,
    note='Attribution required; ATTRIBUTION.csv lists author and source per image.',
)
STANFORD_LICENCE = Licence(
    name='unknown',
    url='https://github.com/pytorch/vision/issues/7545',
    redistributable=False,
    note='Original host is gone; mirrors state no licence. Local evaluation only.',
)
COMPCARS_LICENCE = Licence(
    name='non-commercial research, signed release agreement',
    url='http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html',
    redistributable=False,
    note='No reproduction or distribution of images or derived data. Local only.',
)
OI_VEHICLE_MIN_AREA = 0.05
COMMONS_IMAGE_THREADS = 2
REVIEW_FRACTION = 0.10


def _fetch_commons_file(cand: dict[str, Any], out: Path) -> dict[str, Any]:
    from PIL import Image

    rel = f'images/{cand["slug"]}/{cand["page_id"]}.jpg'
    dest = out / rel
    download(cand['url'], dest, expected_sha256=cand.get('sha256'), retries=4)
    with Image.open(dest) as im:
        im.verify()
    return file_entry(
        out,
        rel,
        page_id=cand['page_id'],
        title=cand['title'],
        author=cand['author'],
        license=cand['license'],
        description_url=cand['description_url'],
        source_url=cand['url'],
        category=cand['category'],
        make=cand['make'],
        model=cand['model'],
        generation=cand['generation'],
    )


def _commons_labels(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        label_row(
            f['path'],
            f'{f["make"]} {f["model"]}',
            [f['make'], f'{f["make"]} {f["model"]}'],
            generation=f['generation'],
        )
        for f in files
    ]


def cmd_commons(args: argparse.Namespace) -> int:
    out: Path = args.out
    if args.manifest.is_file():
        pinned = load_manifest(args.manifest)
        todo = [
            {**f, 'url': f['source_url'], 'slug': f['path'].split('/')[1]} for f in pinned['files']
        ]
        with ThreadPoolExecutor(COMMONS_IMAGE_THREADS) as pool:
            list(pool.map(lambda c: _fetch_commons_file(c, out), todo))
        report = verify_manifest(pinned, out)
        if not report.ok:
            raise FetchError('; '.join(report.problems()[:5]))
        write_dataset(out, pinned, _commons_labels(pinned['files']), None)
        logger.info('commons: %d pinned files verified', pinned['count'])
        return 0
    rows = commons.load_categories(args.categories)
    per_model: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, str]] = {}
    for row in rows:
        key = f'{row["make"]}/{row["model"]}/{row["generation"]}'
        per_model[key] = commons.collect(row, args.min_side)
        meta[key] = row
        logger.info('%s: %d candidates', key, len(per_model[key]))
    chosen, dropped = commons.choose(
        commons.drop_ambiguous(per_model), args.n, args.min_per_model, args.seed
    )
    print(
        {
            'models': len(chosen),
            'images': sum(len(v) for v in chosen.values()),
            'dropped_models': dropped,
            'per_make': commons.counts_by_make(chosen),
            'per_licence': dict(Counter(c['license'] for v in chosen.values() for c in v)),
        }
    )
    if args.dry_run:
        return 0
    todo = [
        {**cand, **meta[key], 'slug': commons.slug(meta[key])}
        for key, cands in chosen.items()
        for cand in cands
    ]
    with ThreadPoolExecutor(COMMONS_IMAGE_THREADS) as pool:
        files = list(pool.map(lambda c: _fetch_commons_file(c, out), todo))
    manifest = build_manifest(
        'commons_set_c', commons.API_URL, COMMONS_LICENCE, files, dropped_models=dropped
    )
    write_dataset(out, manifest, _commons_labels(files), None)
    write_json(args.manifest, manifest)
    sample = commons.review_sample([f['path'] for f in files], REVIEW_FRACTION, args.seed)
    write_csv(
        out / 'review_sample.csv',
        [{'path': p, 'label_correct': ''} for p in sample],
        ['path', 'label_correct'],
    )
    logger.info('commons: %d images, %d models, %d to review', len(files), len(chosen), len(sample))
    return 0


def cmd_openimages(args: argparse.Namespace) -> int:
    cache: Path = args.cache_dir
    classes = download(oi.CLASSES_URL, cache / 'oidv7-class-descriptions-boxable.csv')
    wanted = oi.class_ids(classes.read_text(encoding='utf-8'), oi.VEHICLE_CLASSES)
    bbox = download(
        oi.BBOX_URL.format(split=args.split), cache / f'{args.split}-annotations-bbox.csv'
    )
    imgs = download(oi.IMAGES_URL.format(split=args.split), cache / f'{args.split}-images.csv')
    boxes = oi.qualifying_boxes(bbox.read_text(encoding='utf-8'), wanted, OI_VEHICLE_MIN_AREA)
    eligible = oi.eligible_image_ids(imgs.read_text(encoding='utf-8'))
    selection = oi.select_images(boxes, eligible, args.per_class, args.seed)
    print(oi.summarize(selection))
    if args.dry_run:
        return 0
    got = oi.fetch_images(selection, args.split, args.out / 'images')
    oi.write_dataset(
        args.out, 'openimages_vehicles', OI_LICENCE, got, args.split, ['Vehicle'], args.pin
    )
    logger.info('openimages vehicles: %d images -> %s', len(got), args.out)
    return 0


def _index_in_place(
    root: Path, dataset: str, source_url: str, licence: Licence, labelled: list[dict[str, Any]]
) -> None:
    if not labelled:
        raise FetchError(f'{root}: no images found')
    files = [file_entry(root, r['path']) for r in labelled]
    manifest = build_manifest(dataset, source_url, licence, files)
    write_dataset(root, manifest, labelled, None)
    logger.info('%s: indexed %d images in %s', dataset, len(files), root)


def cmd_stanford(args: argparse.Namespace) -> int:
    root: Path = args.root
    rows, classes = [], set()
    for img in sorted(root.rglob('*.jpg')):
        cls = parse_stanford_class(img.parent.name)
        classes.add(img.parent.name)
        rows.append(
            label_row(
                img.relative_to(root).as_posix(),
                cls.class_name,
                cls.taxonomy_path,
                body=cls.body,
                year=cls.year,
            )
        )
    if len(classes) != 196:
        logger.warning('expected 196 classes, found %d', len(classes))
    _index_in_place(root, 'stanford_cars', STANFORD_LICENCE.url, STANFORD_LICENCE, rows)
    return 0


def _read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def cmd_compcars(args: argparse.Namespace) -> int:
    root: Path = args.root
    makes, models = _read_names(args.make_names), _read_names(args.model_names)
    rows = []
    for img in sorted((root / 'image').rglob('*.jpg')):
        rel = img.relative_to(root).as_posix()
        ref = parse_compcars_image(rel)
        make, model = compcars_names(ref, makes, models)
        label_file = root / ref.label_path
        view = (
            parse_compcars_view(label_file.read_text(encoding='utf-8'))
            if label_file.is_file()
            else None
        )
        year = None if ref.year == 'unknown' else int(ref.year)
        rows.append(
            label_row(
                rel,
                f'{make} {model}',
                [make, f'{make} {model}'],
                view=view,
                year=year,
            )
        )
    _index_in_place(root, 'compcars_web_nature', COMPCARS_LICENCE.url, COMPCARS_LICENCE, rows)
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
    c = sub.add_parser('commons')
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--categories', type=Path)
    c.add_argument('--manifest', type=Path, required=True)
    c.add_argument('--n', type=int, default=40)
    c.add_argument('--min-per-model', type=int, default=15)
    c.add_argument('--min-side', type=int, default=1000)
    c.add_argument('--seed', type=int, default=20261029)
    c.add_argument('--dry-run', action='store_true')
    c.set_defaults(func=cmd_commons)
    o = sub.add_parser('openimages')
    o.add_argument('--out', type=Path, required=True)
    o.add_argument('--cache-dir', type=Path, required=True)
    o.add_argument('--split', choices=oi.SPLITS, default='validation')
    o.add_argument('--per-class', type=int, default=200)
    o.add_argument('--seed', type=int, default=20261029)
    o.add_argument('--pin', type=Path)
    o.add_argument('--dry-run', action='store_true')
    o.set_defaults(func=cmd_openimages)
    s = sub.add_parser('stanford')
    s.add_argument('--root', type=Path, required=True)
    s.set_defaults(func=cmd_stanford)
    cc = sub.add_parser('compcars')
    cc.add_argument('--root', type=Path, required=True)
    cc.add_argument('--make-names', type=Path, required=True)
    cc.add_argument('--model-names', type=Path, required=True)
    cc.set_defaults(func=cmd_compcars)
    v = sub.add_parser('verify')
    v.add_argument('directory', type=Path)
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == 'commons' and not args.manifest.is_file() and args.categories is None:
        raise SystemExit('commons needs --categories unless --manifest already exists')
    try:
        return int(args.func(args))
    except FetchError as exc:
        logger.error('%s', exc)
        return 1


if __name__ == '__main__':
    sys.exit(main())
