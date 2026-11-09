"""The ingest read queries moved out of the ingest router."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.services.curation.ingest_reads import ingest_summary, known_image_paths
from src.services.curation.region_drain import pending_region_counts


@pytest.mark.asyncio
async def test_known_paths_key_by_whichever_field_the_caller_asked_for() -> None:
    fake = AsyncMock()
    fake.search = AsyncMock(
        return_value={
            'hits': {
                'hits': [
                    {'_source': {'image_id': 'i1', 'image_path': '/srv/a.jpg'}},
                    # An upload: the server path is unrelated to the caller; its own identifier keys it.
                    {
                        '_source': {
                            'image_id': 'i2',
                            'image_path': '/srv/stored/xyz.jpg',
                            'source_identifier': 'client-b.jpg',
                        }
                    },
                    {'_source': {'image_path': '/srv/c.jpg'}},  # no image_id: skipped
                ]
            }
        }
    )
    found = await known_image_paths(fake, ['/srv/a.jpg', 'client-b.jpg', '/srv/c.jpg'])
    assert found == {'/srv/a.jpg': 'i1', 'client-b.jpg': 'i2'}


@pytest.mark.asyncio
async def test_known_paths_chunk_large_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.services.curation.ingest_reads._PATH_LOOKUP_CHUNK', 2)
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'hits': {'hits': []}})
    await known_image_paths(fake, ['a', 'b', 'c'])
    sizes = [call.kwargs['body']['size'] for call in fake.search.call_args_list]
    assert sizes == [2, 1]


@pytest.mark.asyncio
async def test_summary_scopes_to_a_run_and_defaults_to_empty_buckets() -> None:
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={})
    out = await ingest_summary(fake, 'run-1')
    assert out == {'total': 0, 'by_source': [], 'by_day': []}
    assert fake.search.call_args.kwargs['body']['query'] == {'term': {'ingest_run_id': 'run-1'}}
    await ingest_summary(fake, None)
    assert fake.search.call_args.kwargs['body']['query'] == {'match_all': {}}


@pytest.mark.asyncio
async def test_pending_region_counts_read_the_status_buckets_and_never_answer_zero_on_failure() -> (
    None
):
    fake = AsyncMock()
    fake.search = AsyncMock(
        return_value={
            'aggregations': {
                'by_status': {
                    'buckets': [
                        {'key': 'pending_detection', 'doc_count': 4},
                        {'key': 'pending_verification', 'doc_count': 2},
                        {'key': 'detected', 'doc_count': 90},
                    ]
                }
            }
        }
    )
    assert await pending_region_counts(fake) == (4, 2)

    fake.search = AsyncMock(side_effect=RuntimeError('connection refused'))
    with pytest.raises(RuntimeError):
        await pending_region_counts(fake)
