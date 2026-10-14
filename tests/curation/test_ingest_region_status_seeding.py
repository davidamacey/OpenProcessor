"""Region-status seeding: newly ingested items must reach the region worker."""

from __future__ import annotations

from typing import Any

import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service


# =============================================================================
# Region-status seeding — newly ingested items must reach the region worker
# =============================================================================


@pytest.fixture
def neutral_region_profile(monkeypatch: pytest.MonkeyPatch) -> Any:
    """No region profile configured (the OSS neutral default)."""
    from src.services.detection import profile_registry

    monkeypatch.delenv('OP_REGION_PROFILE', raising=False)
    profile_registry._reset_registry_for_tests()
    yield
    profile_registry._reset_registry_for_tests()


class TestRegionStatusSeeding:
    """With a region profile active, every newly created item is seeded
    ``pending_detection`` so the detection worker's pending query selects
    it; an existing status is never overwritten; neutral writes nothing."""

    @staticmethod
    def _status_field() -> str:
        from src.config.region_fields import get_region_fields

        return get_region_fields().status

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_new_items_are_seeded_pending_detection(self) -> None:
        from curation.query_fakes import matches
        from scripts.curation.worker.cascade import _build_pending_query
        from src.config import RegionStatus

        svc, os_fake, _ = _make_service(
            detections=[(0.05, 0.05, 0.4, 0.4, 0.9, 1), (0.5, 0.5, 0.9, 0.9, 0.9, 1)]
        )
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')

        assert result.n_region_queued == 2
        worker_query = _build_pending_query()
        for doc in os_fake.items.values():
            assert doc[self._status_field()] == RegionStatus.PENDING_DETECTION.value
            assert matches(doc, worker_query)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('neutral_region_profile')
    async def test_neutral_default_writes_no_region_status(self) -> None:
        svc, os_fake, _ = _make_service()
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')

        assert result.crops_created == 1
        assert result.n_region_queued == 0
        [doc] = list(os_fake.items.values())
        assert self._status_field() not in doc

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_reingest_never_overwrites_an_existing_region_status(self) -> None:
        from src.config import RegionStatus

        data = _jpeg_bytes()
        svc, os_fake, _ = _make_service()
        await svc.ingest_one(data, '/tmp/photo.jpg')
        [crop_id] = list(os_fake.items.keys())
        os_fake.items[crop_id][self._status_field()] = RegionStatus.DETECTED.value
        os_fake.images.clear()

        result = await svc.ingest_one(data, '/tmp/photo.jpg')

        assert result.crops_updated == 1
        assert result.n_region_queued == 0
        assert os_fake.items[crop_id][self._status_field()] == RegionStatus.DETECTED.value

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_reingest_seeds_an_existing_item_that_has_no_status(self) -> None:
        from src.config import RegionStatus

        data = _jpeg_bytes()
        svc, os_fake, _ = _make_service()
        await svc.ingest_one(data, '/tmp/photo.jpg')
        [crop_id] = list(os_fake.items.keys())
        del os_fake.items[crop_id][self._status_field()]
        os_fake.images.clear()

        result = await svc.ingest_one(data, '/tmp/photo.jpg')

        assert result.n_region_queued == 1
        assert os_fake.items[crop_id][self._status_field()] == RegionStatus.PENDING_DETECTION.value

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_batch_ingest_seeds_every_new_item(self) -> None:
        from src.config import RegionStatus

        svc, os_fake, _ = _make_service()
        images = [_jpeg_bytes(seed=800 + s) for s in range(3)]
        result = await svc.ingest_batch(images, [f'/tmp/seed{s}.jpg' for s in range(3)])

        assert [r.n_region_queued for r in result.results] == [1, 1, 1]
        assert {d[self._status_field()] for d in os_fake.items.values()} == {
            RegionStatus.PENDING_DETECTION.value
        }

    @pytest.mark.usefixtures('reference_region_profile')
    def test_ingest_response_reports_seeded_count(self) -> None:
        from src.routers.curation.ingest import _batch_response
        from src.services.curation.ingest_models import BatchIngestResult, IngestResult

        batch = BatchIngestResult(
            results=[IngestResult(image_path='/a.jpg', n_crops=3, n_region_queued=2)]
        )
        [resp] = _batch_response(batch, []).results
        assert resp.n_regions == 2
