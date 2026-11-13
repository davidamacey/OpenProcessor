#!/usr/bin/env python3
"""Wikimedia Commons face-identity set: one class per public figure.

Builds on ``proxy_commons`` (API paging, licence filter, thumbnails, ambiguity drop, review
sample) and ``proxy_manifest`` (manifest, labels, verification). Two steps:

    commons_faces.py identities --out manifests/commons_faces_identities.json [--n 150]
    commons_faces.py fetch --identities <json> --out <dir> --cache <json> --pin <json>

``identities`` asks Wikidata for humans with a Commons category (P373) and many sitelinks, born
1925 to 2000 so the photographs are real, counts the Commons files whose structured data
depicts each person (P180), keeps those with enough files and picks a spread over gender and
country. ``fetch`` lists the depicting files per person, drops files that depict two chosen
people, takes up to ``--n`` per person and writes images, ``labels.jsonl``, ``manifest.json``
and a seeded ``review_sample.csv``. Group photos stay in until ``filter_one_face.py`` runs
against the live face API.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.datasets import proxy_commons as pc
from scripts.datasets._common import USER_AGENT, FetchError, download, write_csv, write_json
from scripts.datasets.proxy_manifest import (
    Licence,
    build_manifest,
    file_entry,
    label_row,
    pin_of,
    write_dataset,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('commons_faces')

SPARQL_URL = 'https://query.wikidata.org/sparql'
GENDERS = {'Q6581072': 'female', 'Q6581097': 'male'}
IDENTITY_KEYS = ('identity', 'wikidata_qid', 'category')
THUMB_WIDTH = 1600
MIN_SIDE = 600
IMAGE_THREADS = 2
IMAGE_PAUSE_S = 0.3
REVIEW_FRACTION = 0.10
ARTWORK_TITLE = re.compile(
    r'\b(?:painting|drawing|sketch|caricature|cartoon|statue|sculpture|stamp|poster|coin|'
    r'banknote|wax|mural|illustration|oil on canvas)\b',
    re.IGNORECASE,
)
LICENCE = Licence(
    name='per file: CC0, public domain, CC BY, CC BY-SA',
    url='https://commons.wikimedia.org/wiki/Commons:Licensing',
    redistributable=True,
    note='Attribution and share-alike per file; the manifest lists author, licence and the '
    'description page for every image. The identity comes from the structured-data depicts '
    'statement and is noisy (group photos); review_sample.csv is a seeded 10 % check.',
)


def sparql_query(min_sitelinks: int, limit: int) -> str:
    return (
        'SELECT ?p ?label ?cat ?gender ?countryLabel ?born WHERE { '
        '?p wdt:P31 wd:Q5; wdt:P373 ?cat; wdt:P569 ?born; wikibase:sitelinks ?sl . '
        f'FILTER(?sl >= {int(min_sitelinks)}) FILTER(YEAR(?born) >= 1925 && YEAR(?born) <= 2000) '
        'OPTIONAL { ?p wdt:P21 ?gender } OPTIONAL { ?p wdt:P27 ?country } '
        'SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . '
        '?p rdfs:label ?label . ?country rdfs:label ?countryLabel } '
        f'}} LIMIT {int(limit)}'
    )


def run_sparql(query: str, attempts: int = 5) -> dict[str, Any]:
    url = f'{SPARQL_URL}?{urllib.parse.urlencode({"query": query})}'
    headers = {'User-Agent': USER_AGENT, 'Accept': 'application/sparql-results+json'}
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(  # nosec B310  fixed public Wikidata endpoint
                urllib.request.Request(url, headers=headers), timeout=180
            ) as resp:
                result: dict[str, Any] = json.load(resp)
                return result
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 503, 504} or attempt == attempts:
                raise
            time.sleep(float(exc.headers.get('Retry-After') or 30 * attempt))
    raise FetchError('unreachable')


def parse_sparql(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per person (first binding wins), sorted by Wikidata number."""
    people: dict[str, dict[str, Any]] = {}
    for b in doc.get('results', {}).get('bindings', []):
        qid = b['p']['value'].rsplit('/', 1)[-1]
        if qid in people or 'label' not in b or 'cat' not in b:
            continue
        gender = GENDERS.get(b.get('gender', {}).get('value', '').rsplit('/', 1)[-1], 'unknown')
        born = b.get('born', {}).get('value', '')[:4]
        people[qid] = {
            'identity': b['label']['value'],
            'wikidata_qid': qid,
            'category': b['cat']['value'],
            'gender': gender,
            'country': b.get('countryLabel', {}).get('value', 'unknown'),
            'born': int(born) if born.isdigit() else None,
        }
    return [people[q] for q in sorted(people, key=lambda q: int(q[1:]))]


def pick_identities(
    pool: list[dict[str, Any]], n: int, *, min_hits: int, female_share: float, seed: int
) -> list[dict[str, Any]]:
    """Seeded spread: ``female_share`` women, and at each step the person from the least-used
    country (then birth decade), so no region dominates."""
    eligible = [r for r in pool if r['hits'] >= min_hits]
    random.Random(seed).shuffle(eligible)
    want_f = round(n * female_share)
    quota = {'female': want_f, 'other': n - want_f}
    used_country: Counter[str] = Counter()
    used_decade: Counter[Any] = Counter()
    chosen: list[dict[str, Any]] = []
    left = list(eligible)
    while len(chosen) < n and left:
        group = 'female' if quota['female'] > 0 else 'other'
        options = [r for r in left if (r['gender'] == 'female') == (group == 'female')]
        if not options:
            quota[group] = 0
            if not any(quota.values()):
                break
            continue
        best = min(
            options,
            key=lambda r: (used_country[r['country']], used_decade[(r['born'] or 0) // 10]),
        )
        left.remove(best)
        quota[group] -= 1
        used_country[best['country']] += 1
        used_decade[(best['born'] or 0) // 10] += 1
        chosen.append(best)
    return sorted(chosen, key=lambda r: int(r['wikidata_qid'][1:]))


def load_identities(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise FetchError(f'{path}: expected a non-empty JSON list')
    for row in rows:
        if set(IDENTITY_KEYS) - set(row):
            raise FetchError(f'{path}: a row lacks one of {IDENTITY_KEYS}: {row}')
    if len({r['wikidata_qid'] for r in rows}) != len(rows):
        raise FetchError(f'{path}: duplicate wikidata_qid')
    return rows


def collect_identity(
    row: dict[str, Any], min_side: int, thumb_width: int, max_files: int
) -> list[dict[str, Any]]:
    cands = pc.collect_depicted(
        row['wikidata_qid'], min_side, thumb_width=thumb_width, max_files=max_files
    )
    return [c for c in cands if not ARTWORK_TITLE.search(c['title'])]


def _slug(row: dict[str, Any]) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', f'{row["identity"]}_{row["wikidata_qid"]}').strip('_')


def _fetch_file(cand: dict[str, Any], out: Path) -> dict[str, Any]:
    from PIL import Image

    rel = f'images/{cand["slug"]}/{cand["page_id"]}.jpg'
    dest = out / rel
    fresh = not dest.is_file()
    download(cand['url'], dest, retries=5)
    if fresh:
        time.sleep(IMAGE_PAUSE_S)
    with Image.open(dest) as im:
        im.verify()
    return file_entry(
        out, rel,
        page_id=cand['page_id'], title=cand['title'], author=cand['author'],
        license=cand['license'], license_url=cand['license_url'],
        description_url=cand['description_url'], source_url=cand['url'],
        identity=cand['identity'], wikidata_qid=cand['wikidata_qid'], category=cand['category'],
    )  # fmt: skip


def _label(f: dict[str, Any]) -> dict[str, Any]:
    return label_row(
        f['path'], f['identity'], [f['identity']], identity=f['identity'],
        wikidata_qid=f['wikidata_qid'], file_page=f['description_url'], author=f['author'],
        licence=f['license'], licence_url=f['license_url'], sha256=f['sha256'],
    )  # fmt: skip


def build_dataset(
    identities: list[dict[str, Any]],
    per_qid: dict[str, list[dict[str, Any]]],
    out: Path,
    *,
    n: int,
    min_per_identity: int,
    seed: int,
    review_fraction: float = REVIEW_FRACTION,
    pin_path: Path | None = None,
) -> dict[str, Any]:
    """Select, download (existing files are kept), and write manifest, labels and review sample."""
    by_qid = {r['wikidata_qid']: r for r in identities}
    before = sum(len(v) for v in per_qid.values())
    kept = pc.drop_ambiguous(per_qid)
    ambiguous = before - sum(len(v) for v in kept.values())
    chosen, dropped = pc.choose(kept, n, min_per_identity, seed)
    todo = [
        {**c, **by_qid[q], 'slug': _slug(by_qid[q])} for q, cands in chosen.items() for c in cands
    ]
    with ThreadPoolExecutor(IMAGE_THREADS) as pool:
        files = list(pool.map(lambda c: _fetch_file(c, out), todo))
    manifest = build_manifest(
        'commons_faces', pc.API_URL, LICENCE, files,
        dropped_identities=[f'{by_qid[d.split(" (")[0]]["identity"]} {d[d.rindex("("):]}' for d in dropped],
        dropped_ambiguous=ambiguous, identities=len(chosen),
    )  # fmt: skip
    write_dataset(out, manifest, [_label(f) for f in files], pin_path)
    sample = pc.review_sample([f['path'] for f in files], review_fraction, seed)
    write_csv(
        out / 'review_sample.csv',
        [{'path': p, 'label_correct': ''} for p in sample],
        ['path', 'label_correct'],
    )
    return manifest


def cmd_identities(args: argparse.Namespace) -> int:
    pool = parse_sparql(run_sparql(sparql_query(args.min_sitelinks, args.limit)))
    logger.info('wikidata: %d people', len(pool))
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.pool]
    for row in pool:
        row['hits'] = pc.depicts_count(row['wikidata_qid'])
    chosen = pick_identities(
        pool, args.n, min_hits=args.min_hits, female_share=args.female_share, seed=args.seed
    )
    rows = [
        {k: v for k, v in r.items() if k != 'hits'} | {'depicts_files': r['hits']} for r in chosen
    ]
    write_json(args.out, rows)
    print(
        {
            'identities': len(rows),
            'female': sum(r['gender'] == 'female' for r in rows),
            'countries': len({r['country'] for r in rows}),
        }
    )
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    identities = load_identities(args.identities)
    per_qid: dict[str, list[dict[str, Any]]] = {}
    for row in identities:
        qid = row['wikidata_qid']
        per_qid[qid] = pc.cached(
            qid,
            args.cache,
            partial(collect_identity, row, args.min_side, args.thumb_width, args.max_candidates),
        )
        logger.info('%s: %d candidates', row['identity'], len(per_qid[qid]))
    manifest = build_dataset(
        identities, per_qid, args.out, n=args.n, min_per_identity=args.min_per_identity,
        seed=args.seed, pin_path=args.pin,
    )  # fmt: skip
    print(
        {
            'identities': manifest['identities'],
            'images': manifest['count'],
            'bytes': manifest['total_bytes'],
            'dropped_ambiguous': manifest['dropped_ambiguous'],
            'dropped_identities': manifest['dropped_identities'],
            'per_licence': dict(Counter(f['license'] for f in manifest['files'])),
            'manifest_sha256': manifest['manifest_sha256'],
        }
    )
    if args.pin is not None:
        write_json(args.pin, pin_of(manifest))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('identities')
    i.add_argument('--out', type=Path, required=True)
    i.add_argument('--n', type=int, default=150)
    i.add_argument('--min-sitelinks', type=int, default=40)
    i.add_argument('--limit', type=int, default=4000)
    i.add_argument('--pool', type=int, default=700)
    i.add_argument('--min-hits', type=int, default=80)
    i.add_argument('--female-share', type=float, default=0.4)
    i.add_argument('--seed', type=int, default=20261113)
    i.set_defaults(func=cmd_identities)
    f = sub.add_parser('fetch')
    f.add_argument('--identities', type=Path, required=True)
    f.add_argument('--out', type=Path, required=True)
    f.add_argument('--cache', type=Path, help='candidate listing cache outside --out')
    f.add_argument('--pin', type=Path)
    f.add_argument('--n', type=int, default=40)
    f.add_argument('--min-per-identity', type=int, default=15)
    f.add_argument('--min-side', type=int, default=MIN_SIDE)
    f.add_argument('--thumb-width', type=int, default=THUMB_WIDTH)
    f.add_argument('--max-candidates', type=int, default=200)
    f.add_argument('--seed', type=int, default=20261113)
    f.set_defaults(func=cmd_fetch)
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
