"""Batched Triton inference and the crop-created events of the ingest service."""

from __future__ import annotations

from typing import Any

import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.occ_fakes import FakeIngestOpenSearch
from src.services.near_duplicates.content_key import content_imohash


class TestHostStageDetectorCalls:
    """The host stage (a service without the frame engine) runs the detector one photo at a time
    and lets the model server's dynamic batcher group the concurrent calls; the client no longer
    stacks a batch itself. The output is the same as for a photo ingested alone."""

    @pytest.mark.asyncio
    async def test_each_photo_is_one_single_image_detector_call(self) -> None:
        svc, _, triton = _make_service(batch_limit=8)
        images = [_jpeg_bytes(seed=s) for s in range(5)]
        paths = [f'/tmp/b{s}.jpg' for s in range(5)]

        result = await svc.ingest_batch(images, paths)

        assert result.summary.successful == 5
        assert triton.calls == ['primary_end2end'] * 5
        assert triton.batch_sizes == [1] * 5

    @pytest.mark.asyncio
    async def test_batch_skips_duplicates_before_inference(self, content_rows: Any) -> None:
        """Duplicates never reach the detector: only the new photos are staged."""
        dup = _jpeg_bytes(seed=200)
        os_fake = FakeIngestOpenSearch(images={'existing': {'image_id': 'existing'}})
        content_rows.rows[content_imohash(dup)] = 'existing'
        svc, _, triton = _make_service(opensearch=os_fake, batch_limit=8)
        images = [dup, _jpeg_bytes(seed=201), _jpeg_bytes(seed=202)]
        paths = ['/tmp/dup.jpg', '/tmp/d1.jpg', '/tmp/d2.jpg']

        result = await svc.ingest_batch(images, paths)

        assert result.summary.duplicates == 1
        assert result.summary.successful == 2
        assert triton.batch_sizes == [1, 1]

    @pytest.mark.asyncio
    async def test_batch_output_matches_per_image_output(self) -> None:
        """A request's flow and a photo ingested alone produce identical documents."""
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
    async def test_a_detector_that_fails_for_one_photo_fails_that_photo_only(self) -> None:
        svc, os_fake, triton = _make_service(batch_limit=8, fail_on_batch_gt=0)
        real_infer = triton.infer
        calls = [0]

        async def infer(*args: Any, **kwargs: Any) -> Any:
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError('model server refused this call')
            return await real_infer(*args, **kwargs)

        triton.fail_on_batch_gt = None
        triton.infer = infer
        images = [_jpeg_bytes(seed=400 + s) for s in range(3)]
        result = await svc.ingest_batch(images, [f'/tmp/f{s}.jpg' for s in range(3)])

        assert sorted(r.status for r in result.results) == ['failed', 'success', 'success']
        assert [r.error_kind for r in result.results if r.status == 'failed'] == ['detector_infer']
        assert len(os_fake.items) == 2


class TestCropCreatedEvents:
    """N2 — ingest publishes ``crop.created`` for every item doc it newly
    writes, only after the write succeeded, and a publish failure can
    never fail the ingest."""

    @pytest.fixture(autouse=True)
    def _capture(self, captured_events: list[dict[str, Any]]) -> None:
        self._seen = captured_events

    async def _subscribe(self) -> list[dict[str, Any]]:
        self._seen.clear()  # a subscriber only sees events published after it joined
        return self._seen

    @staticmethod
    def _drain(seen: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events = list(seen)
        seen.clear()
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

        os_fake.mget = _broken_mget
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

        os_fake.bulk = _bulk_rejecting_first_create
        await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')

        published = [e['crop_id'] for e in self._drain(sub)]
        assert rejected
        assert rejected[0] not in published
        assert published == list(os_fake.items)

    @pytest.mark.asyncio
    async def test_publish_failure_never_fails_ingest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.services.events import publish as publish_module

        def _boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError('publish exploded')

        monkeypatch.setattr(publish_module, 'publish', _boom)
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
