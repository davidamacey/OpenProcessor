"""Stats read-your-writes: an ingest call (single or batch) refreshes the
images and items indexes once at its end, never per item; a refresh failure
never fails the ingest."""

from __future__ import annotations

from typing import Any

import pytest

from curation.test_ingest_service import _jpeg_bytes, _make_service


class _Indices:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self._fail = fail

    async def refresh(self, index: str, **_kw: Any) -> dict[str, Any]:
        self.calls.append(index)
        if self._fail:
            raise RuntimeError('refresh unavailable')
        return {}


@pytest.mark.asyncio
async def test_a_batch_refreshes_once_covering_both_indexes() -> None:
    svc, os_fake, _ = _make_service()
    os_fake.indices = _Indices()
    await svc.ingest_batch(
        [_jpeg_bytes(seed=1), _jpeg_bytes(seed=2), _jpeg_bytes(seed=3)],
        ['/tmp/a.jpg', '/tmp/b.jpg', '/tmp/c.jpg'],
    )
    assert len(os_fake.indices.calls) == 1
    assert set(os_fake.indices.calls[0].split(',')) == {
        svc.config.images_index,
        svc.config.items_index,
    }


@pytest.mark.asyncio
async def test_a_failing_refresh_does_not_fail_the_batch() -> None:
    svc, os_fake, _ = _make_service()
    os_fake.indices = _Indices(fail=True)
    batch = await svc.ingest_batch([_jpeg_bytes(seed=1)], ['/tmp/a.jpg'])
    assert batch.status == 'success'
    assert len(os_fake.indices.calls) == 1
