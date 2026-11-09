"""The capture step of source_backfill for upgraded projects (CM-3)."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any, cast

import pytest

from curation.capture_backfill_fakes import BackfillCtx, FakeBackfillOpenSearch, bind_backfill
from curation.ingest_fakes import _jpeg_bytes
from src.clients import exiftool_client
from src.services.source_backfill import job
from src.tasks.runtime.context import JobResult


if TYPE_CHECKING:
    from pathlib import Path


def _photo(tmp_path: Path, name: str, *tags: str) -> str:
    path = tmp_path / name
    path.write_bytes(_jpeg_bytes(seed=len(name)))
    if tags:
        subprocess.run(
            ['exiftool', '-overwrite_original', '-q', *tags, str(path)], check=True, timeout=60
        )
    return str(path)


@pytest.fixture(autouse=True)
def _close_exiftool() -> Any:
    yield
    exiftool_client.get_exiftool_client().close()


@pytest.mark.asyncio
async def test_an_upgraded_project_gets_capture_on_images_and_items_and_a_rerun_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind_backfill(monkeypatch, 'store')
    gps = _photo(
        tmp_path,
        'gps.jpg',
        '-DateTimeOriginal=2024:03:05 14:22:11',
        '-OffsetTimeOriginal=+01:00',
        '-GPSLatitude=47.5',
        '-GPSLatitudeRef=N',
        '-GPSLongitude=8.25',
        '-GPSLongitudeRef=E',
    )
    plain = _photo(tmp_path, 'plain.jpg')
    images: dict[str, dict[str, Any]] = {
        'a': {'image_id': 'a', 'image_path': gps},
        'b': {'image_id': 'b', 'image_path': plain},
        'c': {'image_id': 'c', 'image_path': str(tmp_path / 'gone.jpg')},
    }
    items: dict[str, dict[str, Any]] = {
        'ia': {'image_id': 'a'},
        'ib': {'image_id': 'b'},
        'ic': {'image_id': 'c'},
    }
    client = FakeBackfillOpenSearch(images, items)

    first = await job.run(BackfillCtx(client, ['capture']))

    assert isinstance(first, JobResult)
    assert first.result is not None
    assert first.result['capture_recorded'] == 2
    assert first.result['capture_missing'] == 1
    assert images['a']['capture']['location'] == {'lat': 47.5, 'lon': 8.25}
    assert images['a']['capture']['taken_utc'] == '2024-03-05T13:22:11'
    assert images['b']['capture']['exif_status'] == 'absent'
    assert 'capture' not in images['c']
    assert items['ia']['capture_taken_utc'] == '2024-03-05T13:22:11'
    assert 'capture_taken_utc' not in items['ib']

    bulks = client.bulk_calls
    second = await job.run(BackfillCtx(client, ['capture']))
    assert isinstance(second, JobResult)
    assert second.result is not None
    assert second.result['capture_recorded'] == 0
    assert second.result['capture_already'] == 2
    assert client.bulk_calls == bulks


@pytest.mark.asyncio
async def test_a_drop_project_backfills_capture_without_any_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind_backfill(monkeypatch, 'drop')
    gps = _photo(
        tmp_path,
        'gps.jpg',
        '-GPSLatitude=47.5',
        '-GPSLatitudeRef=N',
        '-GPSLongitude=8.25',
        '-GPSLongitudeRef=E',
        '-Make=Acme',
    )
    images: dict[str, dict[str, Any]] = {'a': {'image_id': 'a', 'image_path': gps}}
    await job.run(BackfillCtx(FakeBackfillOpenSearch(images, {}), ['capture']))

    cap = images['a']['capture']
    assert cap['camera_make'] == 'Acme'
    assert cap['has_location'] is True
    assert 'location' not in cap


@pytest.mark.asyncio
async def test_an_image_whose_ingest_read_failed_is_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind_backfill(monkeypatch, 'store')
    path = _photo(tmp_path, 'a.jpg', '-Make=Acme')
    images: dict[str, dict[str, Any]] = {
        'a': {'image_id': 'a', 'image_path': path, 'capture': {'exif_status': 'unknown'}}
    }
    await job.run(BackfillCtx(FakeBackfillOpenSearch(images, {}), ['capture']))
    assert images['a']['capture']['camera_make'] == 'Acme'
    assert images['a']['capture']['parser'] == 'capture@1'


@pytest.mark.asyncio
async def test_an_exiftool_outage_stops_the_step_instead_of_stamping_images_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind_backfill(monkeypatch, 'store')
    monkeypatch.setattr(
        exiftool_client,
        'get_exiftool_client',
        lambda: exiftool_client.ExifToolClient(executable='/nonexistent/exiftool'),
    )
    monkeypatch.setattr(
        'src.services.curation.capture_read.get_exiftool_client',
        exiftool_client.get_exiftool_client,
    )
    images: dict[str, dict[str, Any]] = {
        f'i{n}': {'image_id': f'i{n}', 'image_path': _photo(tmp_path, f'p{n}.jpg')}
        for n in range(5)
    }
    with pytest.raises(exiftool_client.ExifToolError):
        await job.run(BackfillCtx(FakeBackfillOpenSearch(images, {}), ['capture']))
    assert all('capture' not in doc for doc in images.values())


@pytest.mark.asyncio
async def test_the_start_route_passes_the_chosen_steps_as_job_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uuid
    from types import SimpleNamespace

    from src.routers.curation import source_backfill as route
    from src.schemas.curation.source_backfill import SourceBackfillRequest

    seen: list[dict[str, Any]] = []

    async def start_job(
        _session: Any, _kind: str, _project: Any, *, params: dict[str, Any], **_kw: Any
    ) -> Any:
        seen.append(params)
        return SimpleNamespace(job=SimpleNamespace(id=uuid.uuid4()))

    monkeypatch.setattr(route, 'start_job', start_job)
    monkeypatch.setattr(route, 'current_project_id', uuid.uuid4)
    await route.start_source_backfill(cast('Any', None), None)
    await route.start_source_backfill(
        cast('Any', None), SourceBackfillRequest(steps=['location_purge'])
    )
    assert seen == [{}, {'steps': ['location_purge']}]
    with pytest.raises(ValueError, match='steps'):
        SourceBackfillRequest(steps=['frobnicate'])
