"""Tests for scripts/datasets/fetch_geotagged_commons.py: the licence, GPS and street-scene filters
over a recorded Commons API answer, the pin, and the committed manifest. No network."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scripts.datasets import fetch_geotagged_commons as f
from scripts.datasets._common import sha256_file


RECORDED = json.loads(
    (Path(__file__).parent / 'golden' / 'commons_geotagged_listing.json').read_text()
)
MANIFEST = (
    Path(__file__).resolve().parents[2]
    / 'scripts'
    / 'datasets'
    / 'manifests'
    / 'commons_geotagged_60.json'
)
# Pages of the recording that must never be kept, and why.
NOT_A_STREET_SCENE = {129875184, 129875176}
TOO_BIG = 152272131
NO_GPS = 152272135


def _found(**over: Any) -> list[dict[str, Any]]:
    args: dict[str, Any] = {'min_side': 1200, 'max_offset_deg': 0.02, 'max_bytes': 8_000_000}
    return f.site_candidates('berlin', RECORDED['hits'], RECORDED['pages'], **{**args, **over})


def _variant(**mutate: Any) -> list[dict[str, Any]]:
    """The recorded page 72833695 (kept as it is) after ``mutate`` changed its imageinfo."""
    page = copy.deepcopy(next(p for p in RECORDED['pages'] if p['pageid'] == 72833695))
    info = page['imageinfo'][0]
    for key, value in mutate.items():
        if key == 'licence':
            info['extmetadata']['LicenseShortName']['value'] = value
        elif key == 'gps':
            info['metadata'] = [m for m in info['metadata'] if not m['name'].startswith('GPS')]
            info['metadata'] += value
        else:
            info[key] = value
    return f.site_candidates(
        'berlin',
        RECORDED['hits'],
        [page],
        min_side=1200,
        max_offset_deg=0.02,
        max_bytes=8_000_000,
    )


def test_the_recording_keeps_street_scenes_with_real_gps_and_an_allowed_licence() -> None:
    ids = {c['page_id'] for c in _found()}
    assert 72833695 in ids
    assert not ids & NOT_A_STREET_SCENE
    assert TOO_BIG not in ids
    assert NO_GPS not in ids
    for cand in _found():
        assert f.ALLOWED_LICENSE.match(cand['license'])
        assert -90 <= cand['lat'] <= 90
        assert -180 <= cand['lon'] <= 180
        assert cand['url'].startswith('https://upload.wikimedia.org/')
        assert '?' not in cand['url']


def test_a_title_naming_something_else_is_not_a_street_scene() -> None:
    page = copy.deepcopy(next(p for p in RECORDED['pages'] if p['pageid'] == 72833695))
    page['title'] = 'File:Stolperstein Joachimsthaler Strasse 5 Berlin.jpg'
    hits = [{**h, 'title': page['title']} for h in RECORDED['hits'] if h['pageid'] == 72833695]
    kept = f.site_candidates(
        'berlin', hits, [page], min_side=1200, max_offset_deg=0.02, max_bytes=8_000_000
    )
    assert kept == []
    assert f.OFF_TOPIC.search('Textured city map for the blind')


def test_candidates_are_ordered_by_street_scene_score_then_page_id() -> None:
    found = _found()
    assert [c['scene_score'] for c in found] == sorted(
        (c['scene_score'] for c in found), reverse=True
    )
    equal = [c['page_id'] for c in found if c['scene_score'] == found[1]['scene_score']]
    assert equal == sorted(equal)


def test_the_unchanged_page_passes_so_each_variant_below_fails_for_its_own_reason() -> None:
    assert [c['page_id'] for c in _variant()] == [72833695]


@pytest.mark.parametrize(
    'mutate',
    [
        {'mime': 'image/png'},
        {'licence': 'CC BY-NC 2.0'},
        {'licence': 'All rights reserved'},
        {'width': 800, 'height': 600},
        {'size': 9_000_000},
        {'gps': []},
        {'gps': [{'name': 'GPSLatitude', 'value': 0}, {'name': 'GPSLongitude', 'value': 0}]},
        {'gps': [{'name': 'GPSLatitude', 'value': 48.85}, {'name': 'GPSLongitude', 'value': 2.35}]},
        {'gps': [{'name': 'GPSLatitude', 'value': 'n/a'}, {'name': 'GPSLongitude', 'value': 1}]},
    ],
    ids=[
        'not_jpeg',
        'non_commercial',
        'all_rights_reserved',
        'too_small',
        'too_large_file',
        'no_gps',
        'null_island',
        'gps_far_from_the_listed_position',
        'gps_not_a_number',
    ],
)
def test_a_page_failing_one_filter_is_dropped(mutate: dict[str, Any]) -> None:
    assert _variant(**mutate) == []


def test_exif_position_and_time() -> None:
    meta: list[dict[str, Any]] = [
        {'name': 'GPSLatitude', 'value': 52.52},
        {'name': 'GPSLongitude', 'value': 13.405},
        {'name': 'DateTimeOriginal', 'value': '2016:12:26 15:31:33'},
    ]
    assert f.exif_position(meta) == (52.52, 13.405)
    assert f.exif_taken(meta) == '2016-12-26T15:31:33'
    assert f.exif_position([]) is None
    assert f.exif_taken([{'name': 'DateTimeOriginal', 'value': 'unknown'}]) is None


def test_choose_keeps_at_most_two_photos_per_author_and_each_file_once() -> None:
    chosen = f.choose(_found(), per_site=5)
    authors = [c['author'] for c in chosen]
    assert len(chosen) == 5
    assert max(authors.count(a) for a in set(authors)) <= f.MAX_PER_AUTHOR_PER_SITE
    assert len({c['commons_sha1'] for c in chosen}) == len(chosen)
    duplicate = _found()
    duplicate.append({**duplicate[0], 'page_id': 1, 'author': 'someone else'})
    assert 1 not in [c['page_id'] for c in f.choose(duplicate, per_site=99)]


def test_the_search_urls_use_the_documented_geosearch_parameters() -> None:
    url = f.geosearch_url(52.5, 13.4)
    for part in ('list=geosearch', 'gscoord=52.5%7C13.4', 'gsnamespace=6', 'gsradius=1000'):
        assert part in url
    assert 'iiprop=url%7Csize%7Csha1%7Cmime%7Cextmetadata%7Cmetadata' in f.imageinfo_url(
        ['File:A.jpg']
    )


# --- a whole run, with the network replaced ----------------------------------------------------


@pytest.fixture
def run_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    out = tmp_path / 'out'
    (out / 'api_cache').mkdir(parents=True)
    (out / 'api_cache' / 'berlin_test.json').write_text(json.dumps(RECORDED))
    monkeypatch.setattr(f, 'SITES', (('berlin_test', 52.5219, 13.4132),))
    sizes = {c['url']: (c['width'], c['height']) for c in _found()}
    downloads: list[str] = []

    def fake_download(
        url: str, dest: Path, *, expected_sha256: str | None = None, **_kw: Any
    ) -> Path:
        downloads.append(url)
        if dest.is_file() and (expected_sha256 is None or sha256_file(dest) == expected_sha256):
            return dest
        Image.new('RGB', sizes[url], (len(url) % 255, 90, 30)).save(dest, quality=50)
        return dest

    monkeypatch.setattr(f, 'download', fake_download)
    monkeypatch.setattr(f.time, 'sleep', lambda _seconds: None)
    return {'out': out, 'manifest': tmp_path / 'pin.json', 'downloads': downloads}


def _main(env: dict[str, Any]) -> int:
    return f.main(['--out', str(env['out']), '--manifest', str(env['manifest']), '--per-site', '4'])


def test_a_run_writes_the_pin_and_the_credits(run_env: dict[str, Any]) -> None:
    assert _main(run_env) == 0
    pin = json.loads(run_env['manifest'].read_text())
    assert pin['count'] == len(pin['images']) == 4
    for row in pin['images']:
        assert f.ALLOWED_LICENSE.match(row['license'])
        assert row['sha256'] == sha256_file(run_env['out'] / 'images' / row['file_name'])
        assert {'title', 'author', 'license_url', 'description_url', 'lat', 'lon', 'taken'} <= set(
            row
        )
    with (run_env['out'] / 'CREDITS.csv').open(encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    assert [c['file_name'] for c in rows] == [r['file_name'] for r in pin['images']]


def test_a_rerun_downloads_exactly_the_pinned_files_and_checks_them(
    run_env: dict[str, Any],
) -> None:
    assert _main(run_env) == 0
    pinned = {r['url'] for r in json.loads(run_env['manifest'].read_text())['images']}
    run_env['downloads'].clear()
    assert _main(run_env) == 0
    assert set(run_env['downloads']) == pinned


def test_a_changed_image_fails_the_rerun(run_env: dict[str, Any]) -> None:
    assert _main(run_env) == 0
    row = json.loads(run_env['manifest'].read_text())['images'][0]
    path = run_env['out'] / 'images' / row['file_name']
    Image.new('RGB', (row['width'], row['height']), (1, 2, 3)).save(path)
    pin = json.loads(run_env['manifest'].read_text())
    assert f.verify(pin, run_env['out'] / 'images') == [
        f'{row["file_name"]}: sha256 differs from the pin'
    ]


# --- the committed pin -------------------------------------------------------------------------


def test_the_committed_manifest_is_at_least_fifty_licensed_geotagged_photos() -> None:
    pin = json.loads(MANIFEST.read_text())
    rows = pin['images']
    assert pin['count'] == len(rows) >= 50
    assert len({r['sha256'] for r in rows}) == len(rows)
    for row in rows:
        assert f.ALLOWED_LICENSE.match(row['license']), row['title']
        assert row['license_url'].startswith('http')
        assert row['description_url'].startswith('https://commons.wikimedia.org/')
        assert -90 <= row['lat'] <= 90
        assert -180 <= row['lon'] <= 180
        assert (row['lat'], row['lon']) != (0, 0)
        assert row['taken'] is None or len(row['taken']) == 19
        assert len(row['sha256']) == 64


def test_the_committed_manifest_covers_many_sites_and_two_that_are_close() -> None:
    import math

    rows = json.loads(MANIFEST.read_text())['images']
    assert len({r['site'] for r in rows}) >= 10
    berlin = [r for r in rows if r['site'].startswith('berlin')]
    near: list[float] = []
    far: list[float] = []
    for a in berlin:
        for b in berlin:
            if a['site'] < b['site']:
                km = 111 * math.hypot(
                    a['lat'] - b['lat'], (a['lon'] - b['lon']) * math.cos(math.radians(a['lat']))
                )
                (near if km < 15 else far).append(km)
    assert near, 'two sites a few kilometres apart are needed for the map to merge cells'
