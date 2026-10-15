"""Batched Triton inference and the crop-created events of the ingest service."""

from __future__ import annotations

from typing import Any

import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.occ_fakes import FakeIngestOpenSearch


class TestBatchedTritonInference:
    """Regression guards for G11 — ``ingest_batch`` must issue *batched*
    Triton calls, not N single-image calls behind a semaphore.

    The public API is output-identical either way, so nothing else in the
    suite would notice a silent regression back to per-image inference.
    These tests assert on the call count and the request batch dimension,
    which is the only observable difference.
    """

    @pytest.mark.asyncio
    async def test_batch_issues_one_triton_call_for_the_whole_batch(self) -> None:
        svc, _, triton = _make_service(batch_limit=8)
        images = [_jpeg_bytes(seed=s) for s in range(5)]
        paths = [f'/tmp/b{s}.jpg' for s in range(5)]

        result = await svc.ingest_batch(images, paths)

        assert result.summary.successful == 5
        # THE assertion: one Triton round-trip for 5 images, carrying a
        # batch of 5 — not 5 round-trips of batch 1.
        assert triton.calls == ['primary_end2end']
        assert triton.batch_sizes == [5]

    @pytest.mark.asyncio
    async def test_batch_chunks_at_profile_batch_limit(self) -> None:
        """Chunking honors ``DetectionProfile.batch_limit`` — the engine's
        configured ``max_batch_size`` — rather than one giant request."""
        svc, _, triton = _make_service(batch_limit=2)
        images = [_jpeg_bytes(seed=100 + s) for s in range(5)]
        paths = [f'/tmp/c{s}.jpg' for s in range(5)]

        result = await svc.ingest_batch(images, paths)

        assert result.summary.successful == 5
        assert triton.calls == ['primary_end2end'] * 3
        assert triton.batch_sizes == [2, 2, 1]

    @pytest.mark.asyncio
    async def test_batch_skips_duplicates_before_inference(self) -> None:
        """Duplicates never reach the GPU — the batched call carries only
        the non-duplicate images."""
        from src.services.curation.ingest_helpers import _imohash_bytes

        dup = _jpeg_bytes(seed=200)
        os_fake = FakeIngestOpenSearch(
            images={'existing': {'image_id': 'existing', 'imohash': _imohash_bytes(dup)}}
        )
        svc, _, triton = _make_service(opensearch=os_fake, batch_limit=8)
        images = [dup, _jpeg_bytes(seed=201), _jpeg_bytes(seed=202)]
        paths = ['/tmp/dup.jpg', '/tmp/d1.jpg', '/tmp/d2.jpg']

        result = await svc.ingest_batch(images, paths)

        assert result.summary.duplicates == 1
        assert result.summary.successful == 2
        assert triton.batch_sizes == [2]

    @pytest.mark.asyncio
    async def test_batch_output_matches_per_image_output(self) -> None:
        """Batched and per-image paths must produce identical documents —
        batching is a performance change, never a behavior change."""
        images = [_jpeg_bytes(seed=300 + s) for s in range(3)]
        paths = [f'/tmp/e{s}.jpg' for s in range(3)]

        svc_batch, os_batch, _ = _make_service()
        await svc_batch.ingest_batch(images, paths)

        svc_single, os_single, _ = _make_service()
        for data, path in zip(images, paths, strict=True):
            await svc_single.ingest_one(data, path, source='batch')

        def _comparable(store: dict) -> list[dict]:
            out = [
                {
                    k: v
                    for k, v in doc.items()
                    if k not in {'created_at', 'updated_at', 'class_labeled_at'}
                }
                for doc in store.values()
            ]
            return sorted(out, key=lambda d: d['crop_id'])

        assert _comparable(os_batch.items) == _comparable(os_single.items)

    @pytest.mark.asyncio
    async def test_batch_falls_back_to_per_image_when_batched_call_fails(self) -> None:
        """A detector that rejects multi-image requests must not fail the
        ingest — it falls back to the per-image path."""
        svc, os_fake, triton = _make_service(batch_limit=8, fail_on_batch_gt=1)
        images = [_jpeg_bytes(seed=400 + s) for s in range(3)]
        paths = [f'/tmp/f{s}.jpg' for s in range(3)]

        result = await svc.ingest_batch(images, paths)

        assert result.summary.successful == 3
        assert len(os_fake.items) == 3
        # The batched attempt raised (never recorded), then three
        # per-image calls of batch 1 went through.
        assert triton.batch_sizes == [1, 1, 1]

    @pytest.mark.asyncio
    async def test_ingest_one_honors_prefilled_items(self) -> None:
        """``prefilled_items`` is real plumbing, not a vestigial arg: when
        supplied, ``ingest_one`` issues no detector call at all."""
        from src.services.curation.item_doc import DetectedItem

        svc, os_fake, triton = _make_service()
        prefilled = [
            DetectedItem(
                bbox_pixel=(10.0, 10.0, 80.0, 60.0),
                score=0.9,
                class_id=1,
                class_name='widget',
                class_source='primary_model',
                proposal_name='widget',
            )
        ]
        result = await svc.ingest_one(
            _jpeg_bytes(seed=500), '/tmp/g.jpg', prefilled_items=prefilled
        )

        assert result.status == 'success'
        assert result.n_crops == 1
        assert triton.calls == []
        [doc] = list(os_fake.items.values())
        assert doc['class_id'] == 1

    @pytest.mark.asyncio
    async def test_ingest_one_honors_empty_prefilled_items(self) -> None:
        """An empty prefilled list means "the detector found nothing",
        not "not prefilled" — it must not trigger a detector call."""
        svc, os_fake, triton = _make_service()
        result = await svc.ingest_one(_jpeg_bytes(seed=501), '/tmp/h.jpg', prefilled_items=[])

        assert result.status == 'success'
        assert result.n_crops == 0
        assert triton.calls == []
        assert len(os_fake.images) == 1


class TestCropCreatedEvents:
    """N2 — ingest publishes ``crop.created`` for every item doc it newly
    writes, only after the write succeeded, and a publish failure can
    never fail the ingest."""

    @pytest.fixture(autouse=True)
    def _fresh_hub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.curation import event_hub

        monkeypatch.setattr(event_hub, '_HUB', None)

    @staticmethod
    async def _subscribe() -> Any:
        from src.services.curation.event_hub import get_event_hub

        return await get_event_hub().subscribe(project='default')

    @staticmethod
    def _drain(sub: Any) -> list[dict[str, Any]]:
        events = []
        while not sub.queue.empty():
            events.append(sub.queue.get_nowait())
        return events

    @pytest.mark.asyncio
    async def test_ingest_one_publishes_crop_created_per_new_item(self) -> None:
        sub = await self._subscribe()
        svc, os_fake, _ = _make_service(
            detections=[(0.05, 0.05, 0.4, 0.4, 0.9, 1), (0.5, 0.5, 0.9, 0.9, 0.9, 1)]
        )
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')
        assert result.crops_created == 2

        events = self._drain(sub)
        assert sorted(e['crop_id'] for e in events) == sorted(os_fake.items)
        for event in events:
            assert event['type'] == 'crop.created'
            assert event['topic'] == 'crop'
            assert event['image_path'] == '/tmp/photo.jpg'
            assert isinstance(event['ts'], float)

    @pytest.mark.asyncio
    async def test_reingest_update_is_not_announced_as_created(self) -> None:
        data = _jpeg_bytes()
        svc, os_fake, _ = _make_service()
        await svc.ingest_one(data, '/tmp/photo.jpg')
        os_fake.images.clear()

        sub = await self._subscribe()
        result = await svc.ingest_one(data, '/tmp/photo.jpg')
        assert result.crops_updated == 1
        assert self._drain(sub) == []

    @pytest.mark.asyncio
    async def test_failed_write_publishes_nothing(self) -> None:
        sub = await self._subscribe()
        svc, os_fake, _ = _make_service()

        async def _broken_mget(*, body: Any, index: str) -> Any:
            raise RuntimeError('opensearch down')

        os_fake.mget = _broken_mget  # type: ignore[method-assign]
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')
        assert result.status == 'failed'
        assert self._drain(sub) == []

    @pytest.mark.asyncio
    async def test_rejected_create_is_not_published(self) -> None:
        sub = await self._subscribe()
        svc, os_fake, _ = _make_service(
            detections=[(0.05, 0.05, 0.4, 0.4, 0.9, 1), (0.5, 0.5, 0.9, 0.9, 0.9, 1)]
        )
        real_bulk = os_fake.bulk
        rejected: list[str] = []

        async def _bulk_rejecting_first_create(*, body: Any, refresh: Any = False) -> Any:
            resp = await real_bulk(body=body, refresh=refresh)
            for item in resp['items']:
                if 'create' in item and not rejected:
                    rejected.append(item['create']['_id'])
                    item['create']['status'] = 400
                    os_fake.items.pop(item['create']['_id'])
            return resp

        os_fake.bulk = _bulk_rejecting_first_create  # type: ignore[method-assign]
        await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')

        published = [e['crop_id'] for e in self._drain(sub)]
        assert rejected
        assert rejected[0] not in published
        assert published == list(os_fake.items)

    @pytest.mark.asyncio
    async def test_publish_failure_never_fails_ingest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.services.curation import event_hub

        def _boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError('hub exploded')

        monkeypatch.setattr(event_hub.EventHub, 'publish', _boom)
        svc, os_fake, _ = _make_service()
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')
        assert result.status == 'success'
        assert result.crops_created == 1
        assert len(os_fake.items) == 1

    @pytest.mark.asyncio
    async def test_batch_ingest_publishes_for_every_image(self) -> None:
        sub = await self._subscribe()
        svc, os_fake, _ = _make_service()
        images = [_jpeg_bytes(seed=700 + s) for s in range(3)]
        paths = [f'/tmp/ev{s}.jpg' for s in range(3)]
        result = await svc.ingest_batch(images, paths)
        assert result.summary.successful == 3

        events = self._drain(sub)
        assert sorted(e['crop_id'] for e in events) == sorted(os_fake.items)
        assert sorted(e['image_path'] for e in events) == sorted(paths)
