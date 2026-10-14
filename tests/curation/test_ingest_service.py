"""Tests for src.services.curation.ingest.CurationIngestService."""

from __future__ import annotations

from typing import Any

import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.occ_fakes import FakeIngestOpenSearch
from src.services.curation.ingest import MAX_INGEST_CONCURRENCY


del MAX_INGEST_CONCURRENCY  # imported only to confirm the module exports it


class TestDuplicateDetection:
    @pytest.mark.asyncio
    async def test_single_check_duplicate_hit(self) -> None:
        os_fake = FakeIngestOpenSearch(images={'img1': {'image_id': 'img1', 'imohash': 'HASH'}})
        svc, _, _ = _make_service(opensearch=os_fake)
        assert await svc._check_duplicate('HASH') == 'img1'

    @pytest.mark.asyncio
    async def test_single_check_duplicate_miss(self) -> None:
        svc, _, _ = _make_service()
        assert await svc._check_duplicate('NOPE') is None

    @pytest.mark.asyncio
    async def test_msearch_batched_dedup(self) -> None:
        os_fake = FakeIngestOpenSearch(images={'img1': {'image_id': 'img1', 'imohash': 'H1'}})
        svc, _, _ = _make_service(opensearch=os_fake)
        result = await svc._check_duplicates_msearch(['H1', 'H2'])
        assert result == {'H1': 'img1', 'H2': None}

    @pytest.mark.asyncio
    async def test_msearch_failure_falls_back_to_per_image(self) -> None:
        os_fake = FakeIngestOpenSearch(images={'img1': {'image_id': 'img1', 'imohash': 'H1'}})

        async def _broken_msearch(*, body: Any) -> Any:
            raise RuntimeError('msearch unavailable')

        os_fake.msearch = _broken_msearch  # type: ignore[method-assign]
        svc, _, _ = _make_service(opensearch=os_fake)
        result = await svc._check_duplicates_msearch(['H1', 'H2'])
        assert result == {'H1': 'img1', 'H2': None}

    @pytest.mark.asyncio
    async def test_ingest_one_returns_duplicate_status(self) -> None:
        data = _jpeg_bytes()
        from src.services.curation.ingest import _imohash_bytes

        h = _imohash_bytes(data)
        os_fake = FakeIngestOpenSearch(images={'img1': {'image_id': 'img1', 'imohash': h}})
        svc, _, _ = _make_service(opensearch=os_fake)
        result = await svc.ingest_one(data, '/tmp/a.jpg')
        assert result.status == 'duplicate'
        assert result.image_id == 'img1'


class TestIngestOne:
    @pytest.mark.asyncio
    async def test_success_writes_quality_fields_and_embedding(self) -> None:
        svc, os_fake, triton = _make_service()
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg', source='unit_test')

        assert result.status == 'success'
        assert result.n_crops == 1
        assert result.crops_created == 1
        assert triton.calls == ['primary_end2end']

        [doc] = list(os_fake.items.values())
        assert doc['class_id'] == 1
        assert doc['class_name'] == 'widget'
        assert doc['class_source'] == 'primary_model'
        assert doc['crop_area_norm'] > 0
        assert doc['crop_rank_in_image'] == 1
        assert 'blur_lap_var' in doc
        assert 'blur_lap_ratio' in doc
        assert doc['pe_embedding'] == pytest.approx([1.0, 0.0, 0.0])

        [image_doc] = list(os_fake.images.values())
        assert image_doc['pe_embedding'] == pytest.approx([0.0, 1.0, 0.0])

    @pytest.mark.asyncio
    async def test_low_confidence_detection_is_unlabeled_proposal(self) -> None:
        svc, os_fake, _ = _make_service(
            detections=[(0.05, 0.05, 0.6, 0.6, 0.1, 1)], confidence_floor=0.5
        )
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')
        assert result.status == 'success'
        [doc] = list(os_fake.items.values())
        assert 'class_id' not in doc
        assert doc['class_source'] == 'primary_low_conf'
        # Diagnostic lineage field still recorded even when unlabeled.
        assert doc['proposal_name'] == 'widget'

    @pytest.mark.asyncio
    async def test_empty_bytes_fails_cleanly(self) -> None:
        svc, _, _ = _make_service()
        result = await svc.ingest_one(b'', '/tmp/empty.jpg')
        assert result.status == 'failed'
        assert result.error_kind == 'empty'

    @pytest.mark.asyncio
    async def test_undecodable_bytes_fails_cleanly(self) -> None:
        svc, _, _ = _make_service()
        result = await svc.ingest_one(b'not an image', '/tmp/bad.jpg')
        assert result.status == 'failed'
        # An unidentified image and a generic decode error serve one stable code and reason.
        assert result.error_kind == 'decode_failed'
        assert result.error == 'not a decodable image'

    @pytest.mark.asyncio
    async def test_no_detections_still_indexes_image_doc(self) -> None:
        svc, os_fake, _ = _make_service(detections=[])
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/empty_scene.jpg')
        assert result.status == 'success'
        assert result.n_crops == 0
        assert len(os_fake.images) == 1


class TestClassProvenance:
    """N1 — ingest is the first writer of every items doc, so it must stamp
    the same ``class_detector``/``class_detector_version``/``class_labeler``/
    ``class_labeled_at`` provenance every other class writer records."""

    @pytest.mark.asyncio
    async def test_confident_detection_records_primary_detector(self) -> None:
        svc, os_fake, _ = _make_service(detector_version='7')
        await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')

        [doc] = list(os_fake.items.values())
        assert doc['class_detector'] == 'primary_end2end'
        assert doc['class_detector_version'] == '7'
        assert doc['class_labeler'] == 'ingest'
        assert doc['class_labeled_at'] == doc['created_at']

    @pytest.mark.asyncio
    async def test_low_confidence_proposal_still_records_its_detector(self) -> None:
        """The unlabeled proposal's ``proposal_name`` came from this
        detector too — its provenance is recorded even with no class_id."""
        svc, os_fake, _ = _make_service(
            detections=[(0.05, 0.05, 0.6, 0.6, 0.1, 1)], confidence_floor=0.5
        )
        await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')

        [doc] = list(os_fake.items.values())
        assert 'class_id' not in doc
        assert doc['class_detector'] == 'primary_end2end'
        assert doc['class_detector_version'] == '1'

    @pytest.mark.asyncio
    async def test_reingest_never_overwrites_human_class_provenance(self) -> None:
        data = _jpeg_bytes()
        svc, os_fake, _ = _make_service()
        await svc.ingest_one(data, '/tmp/photo.jpg')
        [crop_id] = list(os_fake.items.keys())

        os_fake.items[crop_id].update(
            {
                'class_source': 'human',
                'label_source': 'human',
                'class_detector': 'human',
                'class_detector_version': 'h1',
                'class_labeler': 'human',
                'class_labeled_at': '2020-01-01T00:00:00+00:00',
            }
        )
        os_fake.images.clear()
        await svc.ingest_one(data, '/tmp/photo.jpg')

        doc = os_fake.items[crop_id]
        assert doc['class_detector'] == 'human'
        assert doc['class_detector_version'] == 'h1'
        assert doc['class_labeler'] == 'human'
        assert doc['class_labeled_at'] == '2020-01-01T00:00:00+00:00'

    @pytest.mark.asyncio
    async def test_reingest_does_not_invent_provenance_on_legacy_human_doc(self) -> None:
        """A human-owned doc written before class provenance existed must
        not gain the ingest detector's provenance on re-ingest — that would
        claim a detector produced a human's label."""
        data = _jpeg_bytes()
        svc, os_fake, _ = _make_service()
        await svc.ingest_one(data, '/tmp/photo.jpg')
        [crop_id] = list(os_fake.items.keys())
        doc = os_fake.items[crop_id]
        for key in ('class_detector', 'class_detector_version', 'class_labeler'):
            doc.pop(key, None)
        doc.pop('class_labeled_at', None)
        doc.update({'class_source': 'human', 'label_source': 'human'})
        os_fake.images.clear()

        await svc.ingest_one(data, '/tmp/photo.jpg')

        doc = os_fake.items[crop_id]
        assert 'class_detector' not in doc
        assert 'class_labeler' not in doc


class TestQualityGate:
    def test_ingest_passes_gate_no_active_gate(self) -> None:
        from src.services.curation.clustering.ivf_ingest import (
            ingest_passes_gate,
            reset_ivf_ingest_cache,
        )

        reset_ivf_ingest_cache()
        assert ingest_passes_gate(None, None) is True

    def test_ingest_fails_gate_on_rank(self) -> None:
        from src.services.curation.clustering import ivf_ingest

        ivf_ingest.reset_ivf_ingest_cache()
        ivf_ingest._ivf_ingest_cache['gate'] = {'max_rank': 3, 'min_blur_ratio': None}
        assert ivf_ingest.ingest_passes_gate(5, 1.0) is False
        assert ivf_ingest.ingest_passes_gate(2, 1.0) is True
        ivf_ingest.reset_ivf_ingest_cache()


class TestOccUpsertHumanGuards:
    @pytest.mark.asyncio
    async def test_reingest_preserves_human_class_source(self) -> None:
        """A crop that a human already labeled must not be clobbered by
        a re-ingest of the same image producing the same crop_id."""
        data = _jpeg_bytes()
        svc, os_fake, _ = _make_service(detections=[(0.05, 0.05, 0.6, 0.6, 0.9, 1)])

        first = await svc.ingest_one(data, '/tmp/photo.jpg')
        assert first.status == 'success'
        [crop_id] = list(os_fake.items.keys())

        # Simulate a human relabel, then an operator deleting the images-index
        # doc while the item survives (the exact scenario occ_upsert_bulk's
        # human-field guard exists for) so a re-ingest of the identical bytes
        # isn't short-circuited by dedup but still lands on the same crop_id.
        os_fake.items[crop_id]['class_source'] = 'human'
        os_fake.items[crop_id]['label_source'] = 'human'
        os_fake.items[crop_id]['class_id'] = 99
        os_fake.items[crop_id]['class_name'] = 'human_relabel'
        os_fake.images.clear()

        second = await svc.ingest_one(data, '/tmp/photo.jpg')
        assert second.status == 'success'
        assert second.crops_preserved_human >= 1
        # The guarded provenance fields survive the re-ingest untouched —
        # class_id/class_name are not companions of class_source (only
        # RegionFields.text is a companion of text_source), so this is the
        # documented scope of the human-field guard, not a full label freeze.
        assert os_fake.items[crop_id]['class_source'] == 'human'
        assert os_fake.items[crop_id]['label_source'] == 'human'


class TestBatchIngest:
    @pytest.mark.asyncio
    async def test_batch_reports_success_and_duplicate(self) -> None:
        data_a = _jpeg_bytes(seed=1)
        from src.services.curation.ingest import _imohash_bytes

        existing_hash = _imohash_bytes(data_a)
        os_fake = FakeIngestOpenSearch(
            images={'existing': {'image_id': 'existing', 'imohash': existing_hash}}
        )
        svc, _, _ = _make_service(opensearch=os_fake)

        data_b = _jpeg_bytes(seed=2)
        result = await svc.ingest_batch([data_a, data_b], ['/tmp/a.jpg', '/tmp/b.jpg'])

        assert result.summary.duplicates == 1
        assert result.summary.successful == 1
        assert result.status == 'success'
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_batch_dedups_byte_identical_files_within_the_same_request(self) -> None:
        """D3: two byte-identical files uploaded together must not both
        ingest as 'success' — the second is reported as a duplicate of the
        first (same image_id, same wire field the cross-batch duplicate
        path uses), and exactly one item is created."""
        data = _jpeg_bytes(seed=7)
        svc, os_fake, _ = _make_service()

        result = await svc.ingest_batch([data, data], ['/tmp/x.jpg', '/tmp/y.jpg'])

        assert result.summary.successful == 1
        assert result.summary.duplicates == 1
        first, second = result.results
        assert first.status == 'success'
        assert second.status == 'duplicate'
        assert second.image_id == first.image_id
        assert first.image_id
        # Only one image doc was ever created for the two identical uploads.
        assert len(os_fake.images) == 1

    @pytest.mark.asyncio
    async def test_in_batch_duplicate_keeps_its_own_source_identifier(self) -> None:
        """D3 follow-up: clients match results to request entries by
        source_identifier. The in-batch duplicate follower must carry its
        OWN source_identifier (not null, not the representative's) even
        though image_id points at the representative."""
        data = _jpeg_bytes(seed=8)
        svc, _, _ = _make_service()

        result = await svc.ingest_batch(
            [data, data],
            ['/tmp/a.jpg', '/tmp/b.jpg'],
            source_identifiers=['req-a', 'req-b'],
        )

        first, second = result.results
        assert first.status == 'success'
        assert first.source_identifier == 'req-a'
        assert second.status == 'duplicate'
        assert second.source_identifier == 'req-b'
        assert second.image_id == first.image_id

    @pytest.mark.asyncio
    async def test_cross_batch_duplicate_keeps_its_own_source_identifier(self) -> None:
        """Same requirement for the cross-batch dedup path (hash already
        committed to the index before this batch started)."""
        data = _jpeg_bytes(seed=9)
        from src.services.curation.ingest import _imohash_bytes

        existing_hash = _imohash_bytes(data)
        os_fake = FakeIngestOpenSearch(
            images={'existing': {'image_id': 'existing', 'imohash': existing_hash}}
        )
        svc, _, _ = _make_service(opensearch=os_fake)

        result = await svc.ingest_batch(
            [data],
            ['/tmp/c.jpg'],
            source_identifiers=['req-c'],
        )

        [only] = result.results
        assert only.status == 'duplicate'
        assert only.image_id == 'existing'
        assert only.source_identifier == 'req-c'
