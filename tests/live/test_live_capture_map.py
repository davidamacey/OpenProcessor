"""Live: ExifTool capture and the map, end to end, under each location policy.

The images enter the way the 0.6.0 upgrade carries stored images in (``tests/live/test_live_ingest.py``
covers the ingest routes themselves): a JPEG on the
source mount and an images document without a ``capture`` block, then the ``source_backfill``
job's ``capture`` step reads the file through the API container's own ExifTool process and stores
the block under the project's ``location_policy``. That is the same ``read_capture_file`` ->
``apply_location_policy`` -> ``capture_image_block`` chain ingest uses
(``src/services/source_backfill/capture.py``), against a real OpenSearch.

The pictures are generated solid-colour JPEGs (no photograph, no private data). The tags are
written on the host with PyExifTool's ``set_tags`` (https://sylikc.github.io/pyexiftool/); tag
names and the ``-n`` numeric form follow https://exiftool.org/TagNames/EXIF.html and
https://exiftool.org/TagNames/GPS.html (``GPSLatitudeRef``/``GPSLongitudeRef`` carry the sign, the
number is the magnitude). Order is load-bearing: the policy is switched ``store`` ->
``store_coarse`` -> ``drop`` on one project.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from exiftool import ExifToolHelper
from PIL import Image

from .conftest import API_PREFIX, VERIFY_DATA_DIR, create_project, refresh, wait_until


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


pytestmark = pytest.mark.live

# A deleted slug is retired for good, so each run takes a fresh one.
PROJECT = f'capmap{uuid.uuid4().hex[:6]}'
PREFIX = f'{API_PREFIX}/projects/{PROJECT}'
IMAGES = f'verify_prj_{PROJECT}__images'
TERMINAL = ('completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted')

GPS_ID, PLAIN_ID = 'cap_gps', 'cap_plain'
LATITUDE, LONGITUDE = 40.689247, -74.044502
COARSE = (40.69, -74.04)
TAKEN = '2021-06-05T14:22:11'
MAKE, MODEL = 'OpenProcessorCam', 'Harness-1'
ZOOM = 10


def _write_jpeg(path: Path, tags: dict[str, Any] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (96, 96), (30, 120, 200)).save(path, format='JPEG', quality=80)
    if tags:
        with ExifToolHelper() as et:
            et.set_tags(str(path), tags, params=['-overwrite_original', '-n'])


def _wait_job(api: Any, job_id: str, timeout: float = 120.0) -> dict[str, Any]:
    def fetch() -> dict[str, Any]:
        resp = api.get(f'{PREFIX}/jobs/{job_id}')
        assert resp.status_code == 200, resp.text
        return resp.json()

    deadline = time.monotonic() + timeout
    job = fetch()
    while job['status'] not in TERMINAL and time.monotonic() < deadline:
        time.sleep(1.0)
        job = fetch()
    return job


def _capture(opensearch: Any, image_id: str) -> dict[str, Any]:
    resp = opensearch.get(f'/{IMAGES}/_doc/{image_id}')
    resp.raise_for_status()
    return resp.json()['_source'].get('capture') or {}


def _cells(api: Any) -> list[dict[str, Any]]:
    resp = api.get(f'{PREFIX}/map/cells', params={'zoom': ZOOM})
    assert resp.status_code == 200, resp.text
    assert resp.headers['content-type'].startswith('application/geo+json')
    body = resp.json()
    assert body['type'] == 'FeatureCollection'
    return body['features']


def _set_policy(api: Any, policy: str) -> dict[str, Any]:
    current = api.get(f'{PREFIX}/location_policy')
    assert current.status_code == 200, current.text
    resp = api.put(
        f'{PREFIX}/location_policy',
        json={'location_policy': policy, 'expected_revision': current.json()['revision']},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture(scope='module')
def capture_project(api: Any, opensearch: Any, seeded: dict[str, Any]) -> Iterator[dict[str, Any]]:
    create_project(api, PROJECT, 'Capture map live')
    assert wait_until(lambda: opensearch.head(f'/{IMAGES}').status_code == 200, timeout=60)

    # The host path under the harness data mount is /verify-data/images in the API container.
    folder = f'capture_map_{PROJECT}'
    host_dir = VERIFY_DATA_DIR / 'images' / folder
    paths = {GPS_ID: host_dir / 'gps.jpg', PLAIN_ID: host_dir / 'plain.jpg'}
    _write_jpeg(
        paths[GPS_ID],
        {
            'EXIF:DateTimeOriginal': TAKEN.replace('-', ':').replace('T', ' '),
            'EXIF:OffsetTimeOriginal': '+00:00',
            'EXIF:Make': MAKE,
            'EXIF:Model': MODEL,
            'EXIF:GPSLatitudeRef': 'N',
            'EXIF:GPSLatitude': abs(LATITUDE),
            'EXIF:GPSLongitudeRef': 'W',
            'EXIF:GPSLongitude': abs(LONGITUDE),
        },
    )
    _write_jpeg(paths[PLAIN_ID], None)
    now = datetime.now(UTC).isoformat()
    for index, (image_id, path) in enumerate(paths.items()):
        doc = {
            'image_id': image_id,
            'image_path': f'/verify-data/images/{folder}/{path.name}',
            'source': 'capture_map_live',
            'width': 96,
            'height': 96,
            'imohash': f'{index + 1:032x}',
            'indexed_at': now,
        }
        resp = opensearch.put(f'/{IMAGES}/_doc/{image_id}', json=doc)
        resp.raise_for_status()
    refresh(opensearch, IMAGES)
    started = api.post(f'{PREFIX}/source_backfill/runs', json={'steps': ['capture']})
    assert started.status_code == 202, started.text
    job = _wait_job(api, started.json()['job_id'])
    assert job['status'] == 'completed', job
    refresh(opensearch, IMAGES)
    yield {'job': job}
    deleted = api.delete(f'{API_PREFIX}/projects/{PROJECT}', params={'confirm': PROJECT})
    assert deleted.status_code == 202, deleted.text
    for path in paths.values():
        path.unlink(missing_ok=True)
    host_dir.rmdir()


def test_capture_job_counts_both_files(capture_project: dict[str, Any]) -> None:
    result = capture_project['job']['result'] or {}
    assert result.get('capture_recorded') == 2, capture_project['job']
    assert result.get('capture_unreadable', 0) == 0


def test_gps_image_stores_time_camera_and_exact_location(
    capture_project: dict[str, Any], opensearch: Any
) -> None:
    block = _capture(opensearch, GPS_ID)
    assert block['exif_status'] == 'ok'
    assert block['taken_local'] == TAKEN
    assert block['taken_utc'] == TAKEN
    assert block['camera_make'] == MAKE
    assert block['camera_model'] == MODEL
    assert block['has_location'] is True
    assert block['location_precision'] == 'exact'
    assert block['location']['lat'] == pytest.approx(LATITUDE, abs=1e-4)
    assert block['location']['lon'] == pytest.approx(LONGITUDE, abs=1e-4)


def test_image_without_gps_has_no_location(
    capture_project: dict[str, Any], opensearch: Any
) -> None:
    block = _capture(opensearch, PLAIN_ID)
    assert block['exif_status'] in ('absent', 'ok')
    assert block['has_location'] is False
    assert 'location' not in block
    assert 'camera_make' not in block


def test_map_serves_the_gps_point_and_only_it(api: Any, capture_project: dict[str, Any]) -> None:
    meta = api.get(f'{PREFIX}/map')
    assert meta.status_code == 200, meta.text
    assert meta.json()['available'] is True
    assert meta.json()['location_policy'] == 'store'
    assert meta.json()['n_located'] == 1

    (feature,) = _cells(api)
    lon, lat = feature['geometry']['coordinates']
    assert (lat, lon) == pytest.approx((LATITUDE, LONGITUDE), abs=1e-3)
    assert feature['properties']['count'] == 1

    tile = feature['properties']['tile']
    listed = api.get(f'{PREFIX}/map/images', params={'tile': tile})
    assert listed.status_code == 200, listed.text
    assert listed.json()['image_ids'] == [GPS_ID]


def _image_ids(api: Any, **params: Any) -> list[str]:
    resp = api.get(f'{PREFIX}/images', params=params)
    assert resp.status_code == 200, resp.text
    return [card['image_id'] for card in resp.json()['items']]


def test_area_filters_on_images_and_crops(api: Any, capture_project: dict[str, Any]) -> None:
    around = f'{LONGITUDE - 0.01},{LATITUDE - 0.01},{LONGITUDE + 0.01},{LATITUDE + 0.01}'
    elsewhere = '10,10,11,11'
    assert _image_ids(api, geo_bbox=around) == [GPS_ID]
    assert _image_ids(api, geo_bbox=elsewhere) == []
    ring = ';'.join(
        f'{LONGITUDE + dx},{LATITUDE + dy}' for dx, dy in ((-0.01, -0.01), (0.01, -0.01), (0, 0.01))
    )
    assert _image_ids(api, geo_polygon=ring) == [GPS_ID]
    assert _image_ids(api, has_location='true') == [GPS_ID]
    assert _image_ids(api, has_location='false') == [PLAIN_ID]

    # The project holds no items: the area filter takes the item-side join and matches none.
    crops = api.get(f'{PREFIX}/crops', params={'geo_bbox': around})
    assert crops.status_code == 200, crops.text
    assert crops.json()['total'] == 0

    meta = api.get(f'{PREFIX}/map').json()
    assert meta['n_located'] == 1
    assert meta['n_without_location'] == 1
    assert meta['bounds'] is not None
    # A single-photo pin names its image only at the deepest zoom, where cells are finest.
    deepest = api.get(f'{PREFIX}/map/cells', params={'zoom': meta['max_zoom']}).json()
    (feature,) = deepest['features']
    assert feature['properties']['image_id'] == GPS_ID


def test_store_coarse_rounds_the_stored_point(
    api: Any, opensearch: Any, capture_project: dict[str, Any]
) -> None:
    purge = _set_policy(api, 'store_coarse')['purge_job_id']
    assert purge, 'a stricter policy starts the purge'
    assert _wait_job(api, purge)['status'] == 'completed'
    refresh(opensearch, IMAGES)

    block = _capture(opensearch, GPS_ID)
    assert block['location_precision'] == 'coarse'
    assert (block['location']['lat'], block['location']['lon']) == pytest.approx(COARSE)

    def served() -> list[dict[str, Any]]:
        # The API may serve the previous policy until its policy cache expires.
        if api.get(f'{PREFIX}/map').json()['location_policy'] != 'store_coarse':
            return []
        return _cells(api)

    (feature,) = wait_until(served, timeout=30)
    lon, lat = feature['geometry']['coordinates']
    assert (lat, lon) == pytest.approx(COARSE, abs=0.011)


def test_drop_removes_the_point_everywhere(
    api: Any, opensearch: Any, capture_project: dict[str, Any]
) -> None:
    purge = _set_policy(api, 'drop')['purge_job_id']
    assert purge
    assert _wait_job(api, purge)['status'] == 'completed'
    refresh(opensearch, IMAGES)

    block = _capture(opensearch, GPS_ID)
    assert 'location' not in block
    assert block['camera_make'] == MAKE, 'drop clears coordinates only, not the camera'

    def gone() -> bool:
        return api.get(f'{PREFIX}/map').json()['available'] is False

    assert wait_until(gone, timeout=30)
    assert _cells(api) == []
