"""The fetch gate bounds the encoded bytes held across concurrent batches."""

from __future__ import annotations

import asyncio
import time

from src.services.curation.bulk_ingest import object_fetch
from src.services.curation.bulk_ingest.object_fetch import held_fetch
from src.services.curation.bulk_ingest.object_store import ObjectRef, S3Config
from tests._fake_object_store import FakeObjectStore


MIB = 1024 * 1024


class _CountingStore(FakeObjectStore):
    def __init__(self) -> None:
        super().__init__()
        self.live = 0
        self.peak = 0

    def get(self, ref: ObjectRef):
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
        time.sleep(0.05)
        try:
            return super().get(ref)
        finally:
            with self.lock:
                self.live -= 1


def test_two_batches_that_each_fill_the_capacity_never_download_together(monkeypatch) -> None:
    monkeypatch.setattr(object_fetch, '_gates', {})
    monkeypatch.setattr(object_fetch, '_limits', {})
    store = _CountingStore()
    batches = []
    for b in range(2):
        refs = []
        for i in range(2):
            ref = store.put('bkt', f'b{b}/{i}.jpg', b'x' * 10)
            refs.append(ObjectRef(ref.bucket, ref.key, ref.etag, None, MIB))
        batches.append(refs)
    config = S3Config(allowed=(('bkt', ''),), inflight_mb=2, fetch_workers=8)

    async def one(refs: list[ObjectRef]) -> None:
        async with held_fetch(store, refs, config) as outcomes:
            assert all(o.data for o in outcomes)
            await asyncio.sleep(0.05)  # the batch is still held while it is ingested

    async def both() -> None:
        await asyncio.gather(*(one(refs) for refs in batches))

    asyncio.run(both())
    assert store.peak == 2, 'only one 2 MiB batch may fetch at a time under a 2 MiB gate'
