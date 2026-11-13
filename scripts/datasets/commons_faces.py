#!/usr/bin/env python3
"""Wikimedia Commons face-identity set: one class per public figure.

Builds on ``proxy_commons`` (API paging, licence filter, thumbnails, ambiguity drop, review
sample) and ``proxy_manifest`` (manifest, labels, verification). Two steps:

    commons_faces.py identities --out manifests/commons_faces_identities.json [--n 150]
    commons_faces.py fetch --identities <json> --out <dir> --cache <json> --pin <json>

``identities`` asks Wikidata for humans with a Commons category (P373) and many sitelinks, born
1925 to 2000 so the photographs are real, counts the Commons files whose structured data
depicts each person (P180), or the files of the person's category tree when depicts statements
are sparse (``source`` in the identities file), keeps those with enough files and picks a spread
over gender and country. ``fetch`` lists those files per person, drops files that depict two chosen
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
from scripts.datasets.fetch_hires_set import api_get
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
WIKIDATA_API = 'https://www.wikidata.org/w/api.php'
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
    """Humans with a Commons category born 1925 to 2000; the sitelink range hint keeps it fast."""
    return (
        'SELECT ?p WHERE { ?p wikibase:sitelinks ?sl . hint:Prior hint:rangeSafe true . '
        f'FILTER(?sl >= {int(min_sitelinks)}) '
        '?p wdt:P31 wd:Q5 ; wdt:P373 [] ; wdt:P569 ?born . '
        'FILTER(?born >= "1925-01-01T00:00:00Z"^^xsd:dateTime && '
        '?born < "2001-01-01T00:00:00Z"^^xsd:dateTime) '
        f'}} LIMIT {int(limit)}'
    )


def run_sparql(query: str, attempts: int = 5) -> dict[str, Any]:
    url = f'{SPARQL_URL}?{urllib.parse.urlencode({"query": query})}'
    headers = {'User-Agent': USER_AGENT, 'Accept': 'application/sparql-results+json'}
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(  # nosec B310  fixed public Wikidata endpoint
                urllib.request.Request(url, headers=headers), timeout=120
            ) as resp:
                result: dict[str, Any] = json.loads(resp.read(), strict=False)
                return result
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 502, 503, 504} or attempt == attempts:
                raise
            time.sleep(float(exc.headers.get('Retry-After') or 30 * attempt))
        except ValueError:  # the endpoint cut a slow answer short
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)
    raise FetchError('unreachable')


def parse_qids(doc: dict[str, Any]) -> list[str]:
    found = {b['p']['value'].rsplit('/', 1)[-1] for b in doc.get('results', {}).get('bindings', [])}
    return sorted(found, key=lambda q: int(q[1:]))


def _claim_qids(entity: dict[str, Any], prop: str) -> list[str]:
    out = []
    for claim in entity.get('claims', {}).get(prop, []):
        value = claim.get('mainsnak', {}).get('datavalue', {}).get('value')
        if isinstance(value, dict) and 'id' in value:
            out.append(value['id'])
    return out


def _claim_text(entity: dict[str, Any], prop: str) -> str | None:
    for claim in entity.get('claims', {}).get(prop, []):
        value = claim.get('mainsnak', {}).get('datavalue', {}).get('value')
        if isinstance(value, str | dict):
            return value if isinstance(value, str) else value.get('time')
    return None


def person_row(entity: dict[str, Any], country_labels: dict[str, str]) -> dict[str, Any] | None:
    """A person row from a Wikidata entity, or ``None`` without an English label or category."""
    label = entity.get('labels', {}).get('en', {}).get('value')
    category = _claim_text(entity, 'P373')
    if not label or not category:
        return None
    genders = _claim_qids(entity, 'P21')
    countries = _claim_qids(entity, 'P27')
    born = (_claim_text(entity, 'P569') or '').lstrip('+')[:4]
    return {
        'identity': label,
        'wikidata_qid': entity['id'],
        'category': category,
        'gender': GENDERS.get(genders[0], 'unknown') if genders else 'unknown',
        'country': country_labels.get(countries[0], countries[0]) if countries else 'unknown',
        'born': int(born) if born.isdigit() else None,
    }


def _wikidata_get(**params: str) -> dict[str, Any]:
    time.sleep(pc.API_PAUSE_S)
    url = f'{WIKIDATA_API}?{urllib.parse.urlencode({"format": "json", **params})}'
    return api_get(url)


def fetch_people(qids: list[str]) -> list[dict[str, Any]]:
    """Person rows for ``qids``, via ``wbgetentities`` (50 per request)."""
    entities: dict[str, dict[str, Any]] = {}
    for i in range(0, len(qids), 50):
        doc = _wikidata_get(
            action='wbgetentities', ids='|'.join(qids[i : i + 50]), props='labels|claims',
            languages='en',
        )  # fmt: skip
        entities.update(doc.get('entities', {}))
    country_ids = sorted({c for e in entities.values() for c in _claim_qids(e, 'P27')[:1]})
    labels: dict[str, str] = {}
    for i in range(0, len(country_ids), 50):
        doc = _wikidata_get(
            action='wbgetentities', ids='|'.join(country_ids[i : i + 50]), props='labels',
            languages='en',
        )  # fmt: skip
        for qid, ent in doc.get('entities', {}).items():
            if 'en' in ent.get('labels', {}):
                labels[qid] = ent['labels']['en']['value']
    rows = (person_row(entities[q], labels) for q in qids if q in entities)
    return [r for r in rows if r is not None]


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
            other = 'other' if group == 'female' else 'female'
            quota[other] += quota[group]
            quota[group] = 0
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
    expression = (
        pc.deepcat_expr(row['category'])
        if row.get('source') == 'deepcategory'
        else pc.depicts_expr(row['wikidata_qid'])
    )
    cands = pc.collect_search(expression, min_side, thumb_width=thumb_width, max_files=max_files)
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
    qids = parse_qids(run_sparql(sparql_query(args.min_sitelinks, args.limit)))
    random.Random(args.seed).shuffle(qids)
    pool = fetch_people(qids[: args.pool])
    logger.info('wikidata: %d people, %d in the pool', len(qids), len(pool))
    for row in pool:
        depicts = pc.depicts_count(row['wikidata_qid'])
        row['source'] = 'depicts' if depicts >= args.min_hits else 'deepcategory'
        row['hits'] = (
            depicts
            if row['source'] == 'depicts'
            else pc.search_count(pc.deepcat_expr(row['category']))
        )
    chosen = pick_identities(
        pool, args.n, min_hits=args.min_hits, female_share=args.female_share, seed=args.seed
    )
    rows = [{k: v for k, v in r.items() if k != 'hits'} | {'files': r['hits']} for r in chosen]
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
    i.add_argument('--min-sitelinks', type=int, default=50)
    i.add_argument('--limit', type=int, default=5000)
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
