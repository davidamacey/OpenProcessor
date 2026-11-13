#!/usr/bin/env python3
"""Fetch a licence-filtered, pinned set of geotagged street photos from Wikimedia Commons.

COCO images carry no GPS, so a COCO project has an empty map. This builds the test set the map is
judged on: real camera photos with a real position, a licence that allows reuse and a capture time.

For each fixed site (a few city centres, two of them a few kilometres apart so the map has cells
to merge) the fetcher asks Commons for the files near the point (``list=geosearch`` in the File
namespace, https://www.mediawiki.org/wiki/Extension:GeoData), reads each file's licence and credit
from ``prop=imageinfo&iiprop=extmetadata``
(https://www.mediawiki.org/wiki/Extension:CommonsMetadata) and its EXIF from ``iiprop=metadata``,
and keeps a file when all of these hold:

* the MIME type is JPEG, the longest side is at least ``--min-side`` pixels and the file at most
  ``--max-bytes`` bytes;
* ``LicenseShortName`` matches :data:`ALLOWED_LICENSE` (CC0, public domain, CC BY, CC BY-SA);
* the EXIF block carries ``GPSLatitude`` and ``GPSLongitude`` (the camera's own position, not a
  position typed onto the file page) within ``--max-offset-deg`` of the position Commons lists;
* the title or description names a street scene (:data:`SCENE_WORDS`) and the title does not name
  something else (:data:`OFF_TOPIC`), so the COCO detector finds objects in it.

At most ``--per-site`` files are kept per site and at most two per author, so a single photographer's
burst does not fill a site. The first run fetches the originals (EXIF is dropped from resized
thumbnails), writes the pin and ``CREDITS.csv``; a later run downloads exactly the pinned files and
checks every SHA-256, so the set does not depend on what Commons lists that day.

    fetch_geotagged_commons.py --out data/samples/commons_geotagged_60 \\
        --manifest scripts/datasets/manifests/commons_geotagged_60.json

Requests carry the project's identified User-Agent (``scripts.datasets._common.USER_AGENT``):
the image servers answer a client that does not identify itself with HTTP 429 and ``Retry-After:
600``, an identified one with the file.

Run on the host. Images are never committed; the pin is.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.datasets._common import (
    FetchError,
    download,
    require_basename,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.datasets.fetch_hires_set import ALLOWED_LICENSE, API_URL, api_get


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fetch_geotagged_commons')

SCHEMA = 1

#: (name, latitude, longitude): a busy street or square in each city. The two Berlin sites are about
#: 6 km apart, so a map at city zoom shows two cells and at street zoom several.
SITES: tuple[tuple[str, float, float], ...] = (
    ('berlin_alexanderplatz', 52.5219, 13.4132),
    ('berlin_kurfuerstendamm', 52.5048, 13.3320),
    ('paris_republique', 48.8675, 2.3636),
    ('london_piccadilly', 51.5100, -0.1347),
    ('amsterdam_dam', 52.3731, 4.8926),
    ('vienna_stephansplatz', 48.2086, 16.3725),
    ('prague_wenceslas', 50.0813, 14.4280),
    ('rome_piazza_venezia', 41.8956, 12.4823),
    ('barcelona_placa_catalunya', 41.3870, 2.1700),
    ('zurich_paradeplatz', 47.3699, 8.5391),
    ('new_york_times_square', 40.7580, -73.9855),
    ('tokyo_shibuya', 35.6595, 139.7005),
)
SEARCH_RADIUS_M = 1_000  # a point of the street, not the city centre Commons files stray photos on
SEARCH_LIMIT = 150
INFO_BATCH = 50  # titles per imageinfo request (the API maximum for a normal client)
SCENE_WORDS = re.compile(
    r'street|road|traffic|avenue|boulevard|strasse|straße|platz|square|crossing|junction|'
    r'tram|bus\b|taxi|bicycle|bike|cars?\b|parking|market|station|bridge|sidewalk|pedestrian|'
    r'rue\b|via\b|calle|plaza|piazza|gasse|allee|city|downtown|urban|police|polizei|shibuya|times square',
    re.IGNORECASE,
)
#: Titles that name something other than a street scene, though they also match SCENE_WORDS
#: (a memorial plaque on a street, a city map, a metro corridor).
OFF_TOPIC = re.compile(
    r'gedenktafel|stolperstein|plaque|\bmap\b|\bkarte\b|panneau|couloir|corridor|sortie|'
    r'interior|inside|portrait|audience',
    re.IGNORECASE,
)
MAX_PER_AUTHOR_PER_SITE = 2
DOWNLOAD_PAUSE_S = 1.5
INFO_PROPS = 'url|size|sha1|mime|extmetadata|metadata'
META_FILTER = 'LicenseShortName|LicenseUrl|Artist|Credit|DateTimeOriginal|ImageDescription'


def geosearch_url(lat: float, lon: float, limit: int = SEARCH_LIMIT) -> str:
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'geosearch',
        'gscoord': f'{lat}|{lon}',
        'gsradius': str(SEARCH_RADIUS_M),
        'gslimit': str(limit),
        'gsnamespace': '6',
        'gsprimary': 'all',
    }
    return f'{API_URL}?{urllib.parse.urlencode(params)}'


def imageinfo_url(titles: list[str]) -> str:
    params = {
        'action': 'query',
        'format': 'json',
        'titles': '|'.join(titles),
        'prop': 'imageinfo',
        'iiprop': INFO_PROPS,
        'iiextmetadatafilter': META_FILTER,
    }
    return f'{API_URL}?{urllib.parse.urlencode(params)}'


def exif_position(metadata: list[dict[str, Any]]) -> tuple[float, float] | None:
    """The camera's own GPS position from an imageinfo ``metadata`` list, or ``None``."""
    values = {m.get('name'): m.get('value') for m in metadata or []}
    lat, lon = values.get('GPSLatitude'), values.get('GPSLongitude')
    numbers = (isinstance(v, int | float) and not isinstance(v, bool) for v in (lat, lon))
    if not all(numbers):
        return None
    lat, lon = float(lat), float(lon)  # type: ignore[arg-type]
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat, lon) == (0.0, 0.0):
        return None
    return lat, lon


def exif_taken(metadata: list[dict[str, Any]]) -> str | None:
    """``DateTimeOriginal`` ('2016:12:26 15:31:33') as ISO 8601 without a zone, or ``None``."""
    for m in metadata or []:
        if m.get('name') == 'DateTimeOriginal':
            match = re.fullmatch(r'(\d{4}):(\d\d):(\d\d) (\d\d:\d\d:\d\d)', str(m.get('value')))
            if match:
                y, mo, d, t = match.groups()
                return f'{y}-{mo}-{d}T{t}'
    return None


def _text(value: str) -> str:
    return html.unescape(re.sub(r'<[^>]+>', '', value)).strip()


def scene_score(title: str, description: str) -> int:
    """How many street-scene words the title and description hold (0 = not a street scene)."""
    return len(SCENE_WORDS.findall(f'{title} {description}'))


def site_candidates(
    site: str,
    hits: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    *,
    min_side: int,
    max_offset_deg: float,
    max_bytes: int,
) -> list[dict[str, Any]]:
    """The files of one site that pass every filter, best street-scene score first.

    ``hits`` is the geosearch result, ``pages`` the imageinfo pages of those titles."""
    listed = {h['title']: (float(h['lat']), float(h['lon'])) for h in hits}
    out: list[dict[str, Any]] = []
    for page in pages:
        title = page.get('title', '')
        info = (page.get('imageinfo') or [{}])[0]
        meta = info.get('extmetadata') or {}
        licence = (meta.get('LicenseShortName') or {}).get('value', '')
        if info.get('mime') != 'image/jpeg' or not ALLOWED_LICENSE.match(licence):
            continue
        if max(info.get('width', 0), info.get('height', 0)) < min_side:
            continue
        if info.get('size', 0) > max_bytes:
            continue
        exif = exif_position(info.get('metadata') or [])
        near = listed.get(title)
        if exif is None or near is None:
            continue
        if abs(exif[0] - near[0]) > max_offset_deg or abs(exif[1] - near[1]) > max_offset_deg:
            continue
        description = _text((meta.get('ImageDescription') or {}).get('value', ''))
        score = scene_score(title, description)
        if score == 0 or OFF_TOPIC.search(title):
            continue
        out.append(
            {
                'site': site,
                'page_id': page['pageid'],
                'title': title.removeprefix('File:'),
                'author': _text((meta.get('Artist') or {}).get('value', '')),
                'license': licence,
                'license_url': (meta.get('LicenseUrl') or {}).get('value', ''),
                'description_url': info.get('descriptionurl', ''),
                'url': info['url'].split('?', 1)[0],
                'commons_sha1': info.get('sha1', ''),
                'width': info['width'],
                'height': info['height'],
                'lat': round(exif[0], 6),
                'lon': round(exif[1], 6),
                'taken': exif_taken(info.get('metadata') or []),
                'scene_score': score,
            }
        )
    out.sort(key=lambda c: (-c['scene_score'], c['page_id']))
    return out


def choose(candidates: list[dict[str, Any]], per_site: int) -> list[dict[str, Any]]:
    """The best ``per_site`` candidates of one site: no repeated file, at most
    :data:`MAX_PER_AUTHOR_PER_SITE` per author."""
    chosen: list[dict[str, Any]] = []
    authors: Counter[str] = Counter()
    seen: set[str] = set()
    for cand in candidates:
        if cand['commons_sha1'] in seen or authors[cand['author']] >= MAX_PER_AUTHOR_PER_SITE:
            continue
        seen.add(cand['commons_sha1'])
        authors[cand['author']] += 1
        chosen.append(cand)
        if len(chosen) == per_site:
            break
    return chosen


def api_json(url: str, attempts: int = 4) -> dict[str, Any]:
    """One API request. Commons answers a throttled or failed request with an ``error`` document
    and HTTP 200, so that is checked here, with a wait before each retry."""
    for attempt in range(1, attempts + 1):
        doc = api_get(url)
        if 'error' not in doc:
            return doc
        if attempt == attempts:
            raise FetchError(f'Commons API error: {doc["error"]}')
        time.sleep(20.0 * attempt)
    raise FetchError('unreachable')


def site_listing(site: str, lat: float, lon: float, cache_dir: Path) -> dict[str, Any]:
    """The geosearch hits and the imageinfo pages of one site. The raw answers are cached under
    ``cache_dir`` so a rerun (or a crash after three sites) does not ask the API again."""
    cached = cache_dir / f'{site}.json'
    if cached.is_file():
        listing: dict[str, Any] = json.loads(cached.read_text(encoding='utf-8'))
        return listing
    hits = api_json(geosearch_url(lat, lon))['query']['geosearch']
    titles = [h['title'] for h in hits]
    pages: list[dict[str, Any]] = []
    for start in range(0, len(titles), INFO_BATCH):
        doc = api_json(imageinfo_url(titles[start : start + INFO_BATCH]))
        pages.extend(doc.get('query', {}).get('pages', {}).values())
        time.sleep(1.0)
    listing = {'hits': hits, 'pages': pages}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(listing), encoding='utf-8')
    return listing


def fetch_site(
    site: str, lat: float, lon: float, args: argparse.Namespace, cache_dir: Path
) -> list[dict[str, Any]]:
    listing = site_listing(site, lat, lon, cache_dir)
    found = site_candidates(
        site,
        listing['hits'],
        listing['pages'],
        min_side=args.min_side,
        max_offset_deg=args.max_offset_deg,
        max_bytes=args.max_bytes,
    )
    chosen = choose(found, args.per_site)
    logger.info(
        '%s: %d listed, %d pass, %d kept', site, len(listing['hits']), len(found), len(chosen)
    )
    return chosen


def download_row(cand: dict[str, Any], images_dir: Path) -> dict[str, Any]:
    name = require_basename(f'{cand["page_id"]}.jpg')
    dest = images_dir / name
    download(cand['url'], dest, retries=5)
    time.sleep(DOWNLOAD_PAUSE_S)  # Commons asks bots for a polite request rate
    with Image.open(dest) as im:
        # Commons lists the size as shown (an EXIF rotation applied), PIL decodes it as stored.
        listed = (cand['width'], cand['height'])
        if im.size not in (listed, listed[::-1]):
            raise FetchError(f'{name}: decoded size {im.size} differs from the listing {listed}')
    return {
        **{k: v for k, v in cand.items() if k != 'scene_score'},
        'file_name': name,
        'bytes': dest.stat().st_size,
        'sha256': sha256_file(dest),
    }


def build_manifest(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    import hashlib

    body = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    return {
        'schema': SCHEMA,
        'source': 'Wikimedia Commons, geosearch in the File namespace, original files',
        'sites': [{'name': n, 'lat': la, 'lon': lo} for n, la, lo in SITES],
        'per_site': args.per_site,
        'filters': {
            'mime': 'image/jpeg',
            'license': ALLOWED_LICENSE.pattern,
            'min_side': args.min_side,
            'max_bytes': args.max_bytes,
            'exif_gps_within_deg': args.max_offset_deg,
        },
        'count': len(rows),
        'manifest_sha256': hashlib.sha256(body).hexdigest(),
        'images': rows,
    }


def verify(manifest: dict[str, Any], images_dir: Path) -> list[str]:
    problems = []
    for row in manifest['images']:
        path = images_dir / require_basename(row['file_name'])
        if not path.is_file():
            problems.append(f'{row["file_name"]}: missing')
        elif sha256_file(path) != row['sha256']:
            problems.append(f'{row["file_name"]}: sha256 differs from the pin')
    return problems


def run(args: argparse.Namespace) -> dict[str, Any]:
    images_dir = args.out / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest.is_file():
        manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding='utf-8'))
        for row in manifest['images']:
            download(
                row['url'],
                images_dir / require_basename(row['file_name']),
                expected_sha256=row['sha256'],
                retries=5,
            )
        problems = verify(manifest, images_dir)
        if problems:
            raise FetchError('; '.join(problems[:5]))
        return manifest
    chosen: list[dict[str, Any]] = []
    for site, lat, lon in SITES:
        chosen.extend(fetch_site(site, lat, lon, args, args.out / 'api_cache'))
    rows = [download_row(c, images_dir) for c in sorted(chosen, key=lambda c: c['page_id'])]
    manifest = build_manifest(rows, args)
    write_json(args.manifest, manifest)
    write_csv(
        args.out / 'CREDITS.csv',
        rows,
        (
            'file_name',
            'title',
            'author',
            'license',
            'license_url',
            'description_url',
            'lat',
            'lon',
            'taken',
        ),
        extrasaction='ignore',
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--per-site', type=int, default=5)
    p.add_argument('--min-side', type=int, default=1200)
    p.add_argument('--max-offset-deg', type=float, default=0.02)
    p.add_argument('--max-bytes', type=int, default=8_000_000)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        manifest = run(build_parser().parse_args(argv))
    except FetchError as exc:
        logger.error('%s', exc)
        return 1
    logger.info(
        'pinned %d images, manifest sha256 %s', manifest['count'], manifest['manifest_sha256']
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
