"""Every ingest path carries `capture`, and no coordinate leaves the images index (CM-2).

The location canary is a latitude/longitude pair no other test value contains; it may appear only
in the image document's `capture.location` under `store` (rounded under `store_coarse`, absent
under `drop`) and nowhere else: not on an item, not in the ingest result, not in a log record.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from src.domain.capture_metadata import CaptureLocation, CaptureMetadata
from src.services.curation.capture_read import read_capture


if TYPE_CHECKING:
    from pathlib import Path

CANARY_LAT, CANARY_LON = 47.123456, 8.987654
META = CaptureMetadata(
    exif_status='ok',
    taken_local='2024-03-05T14:22:11',
    taken_utc='2024-03-05T13:22:11',
    offset_minutes=60,
    time_source='exif_original',
    offset_source='exif_offset',
    camera_make='Acme',
    location=CaptureLocation(CANARY_LAT, CANARY_LON, altitude_m=410.0),
    has_location=True,
)


def _wire(svc: Any, policy: str = 'store', *, meta: CaptureMetadata = META) -> list[int]:
    """Give the service a reader that counts its calls and a fixed policy."""
    calls: list[int] = []

    async def reader(_data: bytes) -> CaptureMetadata:
        calls.append(1)
        return meta

    async def policy_source() -> str:
        return policy

    svc._capture_reader = reader
    svc._location_policy = policy_source
    return calls


def _leaks(value: Any) -> bool:
    text = json.dumps(value, default=str)
    # the canary lat/lon at any precision, as a whole number: not the tail of an unrelated one
    return re.search(r'(?<![\d.])(47\.12|8\.98|8\.99)', text) is not None


@pytest.mark.parametrize(
    'value',
    [{'ms': 18.987}, {'score': 0.998}, {'t': 147.1234}, {'text': 'took 18.987 ms'}, {'n': 8}],
)
def test_the_leak_check_ignores_unrelated_numbers(value: Any) -> None:
    assert not _leaks(value)


@pytest.mark.parametrize(
    'value',
    [
        {'lat': 47.123456},
        {'a': {'b': [8.987654]}},
        {'lat': 47.12},
        {'lon': 8.99},
        {'text': '47.123456,8.987654'},
        {'text': 'at 47.12 / 8.99'},
        {'lat': '47.1235'},
    ],
)
def test_the_leak_check_catches_the_coordinates_in_any_field(value: Any) -> None:
    assert _leaks(value)


@pytest.mark.asyncio
async def test_an_upload_ingest_stores_capture_on_the_image_and_two_times_on_the_item() -> None:
    svc, os_fake, _ = _make_service()
    _wire(svc)
    result = await svc.ingest_one(_jpeg_bytes(seed=1), '/tmp/a.jpg')
    assert result.status == 'success'

    [image] = os_fake.images.values()
    assert image['capture']['location'] == {'lat': CANARY_LAT, 'lon': CANARY_LON}
    assert image['capture']['taken_local'] == '2024-03-05T14:22:11'
    assert image['capture']['camera_make'] == 'Acme'
    assert image['capture']['has_location'] is True
    assert image['capture']['parser'] == 'capture@1'
    [item] = os_fake.items.values()
    assert item['capture_taken_local'] == '2024-03-05T14:22:11'
    assert item['capture_taken_utc'] == '2024-03-05T13:22:11'
    assert not _leaks(item)
    assert not _leaks(result.model_dump())


@pytest.mark.asyncio
async def test_the_batch_path_used_by_directory_and_bulk_ingest_carries_capture() -> None:
    svc, os_fake, _ = _make_service()
    _wire(svc)
    images = [_jpeg_bytes(seed=10 + s) for s in range(3)]
    batch = await svc.ingest_batch(images, [f'/tmp/b{s}.jpg' for s in range(3)])

    assert [r.status for r in batch.results] == ['success'] * 3
    assert len(os_fake.images) == 3
    assert all(
        img['capture']['taken_utc'] == '2024-03-05T13:22:11' for img in os_fake.images.values()
    )
    assert all(it['capture_taken_local'] == '2024-03-05T14:22:11' for it in os_fake.items.values())
    assert not _leaks(list(os_fake.items.values()))


@pytest.mark.asyncio
async def test_a_dataset_import_of_a_new_image_carries_capture_in_its_context() -> None:
    svc, _, _ = _make_service()
    _wire(svc)
    ctx = await svc.ingest_image(_jpeg_bytes(seed=20), '/tmp/imp.jpg', adopt_existing=True)
    assert ctx.created
    assert ctx.capture == META


@pytest.mark.asyncio
async def test_a_duplicate_is_not_read_again() -> None:
    svc, os_fake, _ = _make_service()
    calls = _wire(svc)
    data = _jpeg_bytes(seed=30)
    await svc.ingest_one(data, '/tmp/a.jpg')
    second = await svc.ingest_one(data, '/tmp/b.jpg')
    assert second.status == 'duplicate'
    assert len(calls) == 1
    assert len(os_fake.images) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('policy', 'expected'),
    [
        ('store', {'lat': CANARY_LAT, 'lon': CANARY_LON}),
        ('store_coarse', {'lat': 47.12, 'lon': 8.99}),
        ('drop', None),
    ],
)
async def test_the_policy_decides_what_reaches_the_images_index(
    policy: str, expected: dict[str, float] | None
) -> None:
    svc, os_fake, _ = _make_service()
    _wire(svc, policy)
    await svc.ingest_one(_jpeg_bytes(seed=40), '/tmp/a.jpg')

    [image] = os_fake.images.values()
    assert image['capture'].get('location') == expected
    assert image['capture']['has_location'] is True
    if policy != 'store':
        assert 'altitude_m' not in image['capture']
    [item] = os_fake.items.values()
    assert not _leaks(item)


@pytest.mark.asyncio
async def test_without_a_policy_source_no_coordinate_is_stored() -> None:
    svc, os_fake, _ = _make_service()
    calls = _wire(svc)
    svc._location_policy = None
    await svc.ingest_one(_jpeg_bytes(seed=50), '/tmp/a.jpg')
    assert calls
    [image] = os_fake.images.values()
    assert 'location' not in image['capture']


@pytest.mark.asyncio
async def test_without_a_reader_the_capture_is_unknown_and_unstamped() -> None:
    svc, os_fake, _ = _make_service()
    await svc.ingest_one(_jpeg_bytes(seed=60), '/tmp/a.jpg')
    [image] = os_fake.images.values()
    assert image['capture']['exif_status'] == 'unknown'
    assert 'parser' not in image['capture']
    [item] = os_fake.items.values()
    assert 'capture_taken_local' not in item


@pytest.mark.asyncio
async def test_no_log_record_carries_a_coordinate(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    svc, _, _ = _make_service()
    _wire(svc)
    await svc.ingest_one(_jpeg_bytes(seed=70), '/tmp/a.jpg')
    assert not _leaks(caplog.text)


def _with_exif(tmp_path: Path, data: bytes, *tags: str) -> bytes:
    path = tmp_path / 'f.jpg'
    path.write_bytes(data)
    subprocess.run(
        ['exiftool', '-overwrite_original', '-q', *tags, str(path)], check=True, timeout=60
    )
    return path.read_bytes()


@pytest.mark.asyncio
async def test_the_real_reader_end_to_end_on_a_generated_gps_jpeg(tmp_path: Path) -> None:
    data = _with_exif(
        tmp_path,
        _jpeg_bytes(seed=80),
        '-DateTimeOriginal=2024:03:05 14:22:11',
        '-OffsetTimeOriginal=+01:00',
        '-GPSLatitude=47.123456',
        '-GPSLatitudeRef=N',
        '-GPSLongitude=8.987654',
        '-GPSLongitudeRef=E',
    )
    svc, os_fake, _ = _make_service()
    svc._capture_reader = read_capture

    async def policy_source() -> str:
        return 'store'

    svc._location_policy = policy_source
    result = await svc.ingest_one(data, '/tmp/gps.jpg')

    assert result.status == 'success'
    [image] = os_fake.images.values()
    assert image['capture']['taken_utc'] == '2024-03-05T13:22:11'
    assert image['capture']['location'] == {
        'lat': pytest.approx(CANARY_LAT),
        'lon': pytest.approx(CANARY_LON),
    }
    [item] = os_fake.items.values()
    assert item['capture_taken_utc'] == '2024-03-05T13:22:11'
    assert not _leaks(item)
