"""Two different files with one ``imohash`` are two images; a true copy is still a duplicate.

``imohash`` samples the file, so same-size files that differ outside its samples share it. The
offline tier forces the shared hash (real collisions need files over 128 KiB; the database tier
builds them): the second file must not be answered as a duplicate of the first, must not take its
image id, and a re-upload of either must find its own image.
"""

from __future__ import annotations

import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from src.services.near_duplicates import content_key
from src.services.near_duplicates.content_key import content_image_id, content_sha256


SHARED = 'c' * 32


@pytest.fixture(autouse=True)
def _every_file_has_one_imohash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(content_key, 'content_imohash', lambda _data: SHARED)
    monkeypatch.setattr('src.services.curation.ingest.content_imohash', lambda _data: SHARED)


@pytest.mark.asyncio
async def test_a_colliding_file_is_its_own_image_and_a_copy_finds_it(content_rows) -> None:
    first, second = _jpeg_bytes(seed=1), _jpeg_bytes(seed=2)
    svc, os_fake, _ = _make_service()

    a = await svc.ingest_one(first, '/a.jpg')
    b = await svc.ingest_one(second, '/b.jpg')

    assert (a.status, b.status) == ('success', 'success')
    assert a.image_id == content_image_id(SHARED)
    assert b.image_id == content_image_id(content_sha256(second))
    assert len(os_fake.images) == 2
    assert (await svc.ingest_one(second, '/b-copy.jpg')).image_id == b.image_id
    again = await svc.ingest_one(first, '/a-copy.jpg')
    assert (again.status, again.image_id) == ('duplicate', a.image_id)


@pytest.mark.asyncio
async def test_colliding_files_in_one_batch_are_stored_and_a_repeat_is_a_duplicate(
    content_rows,
) -> None:
    first, second = _jpeg_bytes(seed=1), _jpeg_bytes(seed=2)
    svc, os_fake, _ = _make_service()

    batch = await svc.ingest_batch([first, second, first], ['/a.jpg', '/b.jpg', '/a-copy.jpg'])

    assert [r.status for r in batch.results] == ['success', 'success', 'duplicate']
    assert batch.results[2].image_id == batch.results[0].image_id
    assert len({r.image_id for r in batch.results}) == 2
    assert len(os_fake.images) == 2
