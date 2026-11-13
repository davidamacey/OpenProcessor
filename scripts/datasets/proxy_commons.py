"""Wikimedia Commons vehicle-model photos ("set C"), drawn from per-model categories.

The category list is a JSON file (make, model, generation, category). For each category the
direct files and the files of its subcategories (one level, interior/engine/detail
subcategories excluded) are candidates. A file qualifies when it is a JPEG whose long side is
at least ``min_side`` and whose licence short name is CC0, public domain, CC BY or CC BY-SA.
A file that sits in more than one listed model category is dropped from all of them, because
its label would be ambiguous. Labels come from the category, so they are noisy: the fetcher
writes a seeded 10 % ``review_sample.csv`` for a person to check.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from scripts.datasets._common import FetchError, seeded_sample
from scripts.datasets.fetch_hires_set import ALLOWED_LICENSE, api_get


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

API_URL = 'https://commons.wikimedia.org/w/api.php'
THUMB_WIDTH = 1280
MAX_SUBCATS = 12
API_PAUSE_S = 1.0
EXCLUDED_SUBCAT = re.compile(
    r'\b(?:interior|engine|wheel|detail|seat|dashboard|cockpit|logo|badge|emblem|headlight|'
    r'taillight|tail light|rear light|steering|trunk|boot|diagram|drawing|model car|toy|'
    r'scale model|wreck|crash|accident|tuning|paint|speedometer|gear|mirror|chassis|'
    r'undercarriage|brochure|advert|sign|plate)s?\b',
    re.IGNORECASE,
)
CATEGORY_KEYS = ('make', 'model', 'generation', 'category')


def load_categories(path: Path) -> list[dict[str, str]]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise FetchError(f'{path}: expected a non-empty JSON list')
    for row in rows:
        if set(CATEGORY_KEYS) - set(row):
            raise FetchError(f'{path}: a row lacks one of {CATEGORY_KEYS}: {row}')
    if len({r['category'] for r in rows}) != len(rows):
        raise FetchError(f'{path}: duplicate categories')
    return rows


def slug(row: dict[str, str]) -> str:
    text = f'{row["make"]}_{row["model"]}_{row["generation"]}'
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')


def keep_subcategory(title: str) -> bool:
    return EXCLUDED_SUBCAT.search(title.removeprefix('Category:')) is None


def _get(**params: str) -> dict[str, Any]:
    time.sleep(API_PAUSE_S)
    return api_get(f'{API_URL}?{urllib.parse.urlencode({"format": "json", **params})}')


def list_subcategories(category: str, max_subcats: int = MAX_SUBCATS) -> list[str]:
    doc = _get(
        action='query', list='categorymembers', cmtitle=f'Category:{category}',
        cmtype='subcat', cmlimit='100',
    )  # fmt: skip
    titles = sorted(m['title'] for m in doc.get('query', {}).get('categorymembers', []))
    return [t for t in titles if keep_subcategory(t)][:max_subcats]


_IMAGEINFO = {
    'prop': 'imageinfo', 'iiprop': 'url|size|mime|extmetadata',
    'iiextmetadatafilter': 'LicenseShortName|LicenseUrl|Artist',
}  # fmt: skip


def _paged(params: dict[str, str], max_pages: int | None = None) -> list[dict[str, Any]]:
    """Follow the API ``continue`` token, stopping once ``max_pages`` pages are in hand."""
    pages: list[dict[str, Any]] = []
    while True:
        doc = _get(**params)
        pages.extend(doc.get('query', {}).get('pages', {}).values())
        if 'continue' not in doc or (max_pages is not None and len(pages) >= max_pages):
            return pages
        params.update(doc['continue'])


def list_files(category_title: str, thumb_width: int = THUMB_WIDTH) -> list[dict[str, Any]]:
    """Image pages of one category title with size, mime, licence and a thumbnail URL."""
    return _paged({
        'action': 'query', 'generator': 'categorymembers', 'gcmtitle': category_title,
        'gcmtype': 'file', 'gcmlimit': '50', 'iiurlwidth': str(thumb_width), **_IMAGEINFO,
    })  # fmt: skip


def depicts_expr(qid: str) -> str:
    """Search expression for files whose structured data depicts the Wikidata item (P180)."""
    return f'haswbstatement:P180={qid} filetype:bitmap'


def deepcat_expr(category: str) -> str:
    """Search expression for files in a category or any of its subcategories."""
    return f'deepcategory:"{category}" filetype:bitmap'


def search_count(expression: str) -> int:
    doc = _get(
        action='query', list='search', srnamespace='6', srlimit='1', srprop='', srsearch=expression
    )
    return int(doc.get('query', {}).get('searchinfo', {}).get('totalhits', 0))


def depicts_count(qid: str) -> int:
    return search_count(depicts_expr(qid))


def collect_search(
    expression: str, min_side: int, *, thumb_width: int = THUMB_WIDTH, max_files: int = 200
) -> list[dict[str, Any]]:
    """Qualifying candidates among the first ``max_files`` search hits, sorted by page id."""
    pages = _paged({
        'action': 'query', 'generator': 'search', 'gsrnamespace': '6', 'gsrlimit': '50',
        'gsrsearch': expression, 'iiurlwidth': str(thumb_width), **_IMAGEINFO,
    }, max_files)  # fmt: skip
    found: dict[int, dict[str, Any]] = {}
    for page in pages:
        cand = candidate(page, min_side)
        if cand is not None:
            found[cand['page_id']] = cand
    return [found[k] for k in sorted(found)]


def cached(
    key: str, cache_path: Path | None, fn: Callable[[], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """``fn()`` once per ``key``: the listing is kept in ``cache_path`` so a resumed run skips the API."""
    store: dict[str, list[dict[str, Any]]] = {}
    if cache_path is not None and cache_path.is_file():
        store = json.loads(cache_path.read_text(encoding='utf-8'))
    if key not in store:
        store[key] = fn()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(store), encoding='utf-8')
            tmp.replace(cache_path)
    return store[key]


def candidate(page: dict[str, Any], min_side: int) -> dict[str, Any] | None:
    info = (page.get('imageinfo') or [{}])[0]
    meta = info.get('extmetadata', {})
    licence = meta.get('LicenseShortName', {}).get('value', '')
    if info.get('mime') != 'image/jpeg' or not ALLOWED_LICENSE.match(licence):
        return None
    if max(info.get('width', 0), info.get('height', 0)) < min_side or not info.get('thumburl'):
        return None
    return {
        'page_id': page['pageid'],
        'title': page['title'].removeprefix('File:'),
        'url': info['thumburl'],
        'description_url': info.get('descriptionurl', ''),
        'license': licence,
        'license_url': meta.get('LicenseUrl', {}).get('value', ''),
        'author': html.unescape(re.sub(r'<[^>]+>', '', meta.get('Artist', {}).get('value', ''))),
    }


def collect(
    row: dict[str, str],
    min_side: int,
    *,
    thumb_width: int = THUMB_WIDTH,
    max_subcats: int = MAX_SUBCATS,
) -> list[dict[str, Any]]:
    """Qualifying candidates of one model category, sorted by page id."""
    titles = [f'Category:{row["category"]}', *list_subcategories(row['category'], max_subcats)]
    found: dict[int, dict[str, Any]] = {}
    for title in titles:
        for page in list_files(title, thumb_width):
            cand = candidate(page, min_side)
            if cand is not None:
                found[cand['page_id']] = cand
    return [found[k] for k in sorted(found)]


def drop_ambiguous(per_model: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Remove files that appear under more than one model category."""
    seen: Counter[int] = Counter(c['page_id'] for cands in per_model.values() for c in cands)
    return {key: [c for c in cands if seen[c['page_id']] == 1] for key, cands in per_model.items()}


def choose(
    per_model: dict[str, list[dict[str, Any]]], per_model_n: int, min_per_model: int, seed: int
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Seeded sample per model; models with fewer than ``min_per_model`` are dropped and named."""
    chosen: dict[str, list[dict[str, Any]]] = {}
    dropped: list[str] = []
    for key in sorted(per_model):
        cands = per_model[key]
        if len(cands) < min_per_model:
            dropped.append(f'{key} ({len(cands)})')
            continue
        chosen[key] = sorted(seeded_sample(cands, per_model_n, seed), key=lambda c: c['page_id'])
    return chosen, dropped


def review_sample(paths: list[str], fraction: float, seed: int) -> list[str]:
    """A seeded sample of file paths for a person to check label correctness."""
    n = max(1, round(len(paths) * fraction))
    return sorted(seeded_sample(sorted(paths), n, seed))


def counts_by_make(chosen: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for key, cands in chosen.items():
        out[key.split('/')[0]] += len(cands)
    return dict(out)
