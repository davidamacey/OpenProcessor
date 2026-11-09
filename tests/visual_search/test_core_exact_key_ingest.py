"""Two concurrent core ingests of one file store it once: the second is a duplicate of the first.

Before ND-C4 the core duplicate check was an index lookup by ``imohash`` (``refresh=False`` bulk
writes are invisible to it), so two requests in flight both passed it and indexed the file twice
(#358 shape). Now the exact content row is claimed under an advisory lock before the images
document is written.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from fixtures.job_rows import sql

from src.services.near_duplicates import content_key
from src.services.visual_search import VisualSearchService
from visual_search.core_fakes import INDEX, FakeCoreOpenSearch


pytestmark = [pytest.mark.db, pytest.mark.asyncio]

UNIT = [1.0, 0.0, 0.0]


@pytest.fixture
def opensearch(monkeypatch: pytest.MonkeyPatch) -> FakeCoreOpenSearch:
    fake = FakeCoreOpenSearch()

    def infer(images: list[bytes], **_kw: Any) -> list[dict[str, Any]]:
        return [{'num_dets': 0, 'num_faces': 0, 'global_embedding': UNIT} for _ in images]

    monkeypatch.setattr('src.services.visual_search.ingest_batch.infer_batch', infer)
    monkeypatch.setattr('src.clients.triton_client.get_sync_triton_client', lambda _url: object())
    return fake


def _service(fake: FakeCoreOpenSearch) -> VisualSearchService:
    service = VisualSearchService.__new__(VisualSearchService)
    service.opensearch = fake  # type: ignore[assignment]
    service.inference = None  # type: ignore[assignment]
    return service


async def _batch(service: VisualSearchService, data: bytes, image_id: str) -> dict[str, Any]:
    return await service.ingest_batch(
        [(data, image_id, None)],
        enable_ocr=False,
        enable_detection=False,
        enable_faces=False,
        detect_near_duplicates=False,
    )


async def test_two_requests_with_one_file_index_it_once(pg, bound_async_engine, opensearch) -> None:
    data = uuid.uuid4().bytes * 400  # unique per run: the module's database is shared
    service = _service(opensearch)
    ids = (f'core-{uuid.uuid4().hex[:6]}-1', f'core-{uuid.uuid4().hex[:6]}-2')

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(_batch(service, data, i)) for i in ids]
    results = [t.result() for t in tasks]

    assert sorted(r['processed'] for r in results) == [0, 1]
    assert sorted(r['duplicates'] for r in results) == [0, 1]
    assert len(opensearch.docs[INDEX]) == 1
    (stored,) = opensearch.docs[INDEX]
    (loser,) = [r for r in results if r['processed'] == 0]
    assert loser['duplicate_details'][0]['existing_image_id'] == stored
    assert sql(
        pg,
        'SELECT image_id FROM image_contents WHERE project_id IS NULL AND imohash = %s',
        content_key.content_imohash(data),
    ) == [(stored,)]


async def test_a_later_upload_of_the_same_file_is_a_duplicate_until_the_image_is_deleted(
    pg, bound_async_engine, opensearch
) -> None:
    from src.services.image_admin import delete_image_everywhere

    data = uuid.uuid4().bytes * 400
    service = _service(opensearch)
    first, again = f'core-{uuid.uuid4().hex[:6]}-a', f'core-{uuid.uuid4().hex[:6]}-b'
    await _batch(service, data, first)

    duplicate = await _batch(service, data, again)
    assert (duplicate['processed'], duplicate['duplicates']) == (0, 1)

    opensearch.delete = lambda **kw: _drop(opensearch, kw['index'], kw['id'])  # type: ignore[method-assign]
    opensearch.delete_by_query = _no_hits  # type: ignore[method-assign]
    await delete_image_everywhere(opensearch, first)  # type: ignore[arg-type]
    fresh = await _batch(service, data, again)
    assert (fresh['processed'], fresh['duplicates']) == (1, 0)


async def _drop(fake: FakeCoreOpenSearch, index: str, image_id: str) -> None:
    if index == INDEX:
        fake.docs[INDEX].pop(image_id)


async def _no_hits(**_kw: Any) -> dict[str, Any]:
    return {'deleted': 0}
