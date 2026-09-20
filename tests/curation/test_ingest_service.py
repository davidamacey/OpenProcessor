"""Tests for src.services.curation.ingest.CurationIngestService."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from PIL import Image

from curation.occ_fakes import FakeIngestOpenSearch
from src.config import CurationConfig, DetectionProfile
from src.services.curation.ingest import MAX_INGEST_CONCURRENCY, CurationIngestService


del MAX_INGEST_CONCURRENCY  # imported only to confirm the module exports it


@dataclass
class _FakeClassEntry:
    class_id: int
    class_name: str
    deprecated: bool = False


@dataclass
class _FakeRegistryFile:
    classes: list[_FakeClassEntry] = field(default_factory=list)


class FakeClassRegistry:
    """Minimal stand-in for src.clients.curation_opensearch.ClassRegistry."""

    def __init__(self, entries: list[_FakeClassEntry] | None = None) -> None:
        self._file = _FakeRegistryFile(entries or [])

    def load(self) -> _FakeRegistryFile:
        return self._file

    def get(self, class_id: int) -> _FakeClassEntry | None:
        for c in self._file.classes:
            if c.class_id == class_id:
                return c
        return None


class FakeInferResult:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self._outputs[name]


class FakeTritonPool:
    """Returns pre-scripted end2end detections for the primary detector.

    ``detections`` is a list of ``(x1, y1, x2, y2, score, class_id)`` with
    box coordinates normalized to ``[0, 1]`` of the network input square —
    the same convention the real end2end TRT export uses (ingest multiplies
    by ``input_size`` before undoing the letterbox).
    """

    def __init__(self, detections: list[tuple[float, float, float, float, float, int]]) -> None:
        self.detections = detections
        self.calls: list[str] = []

    async def infer(self, model_name: str, inputs: list, outputs: list) -> FakeInferResult:  # noqa: ARG002
        self.calls.append(model_name)
        n = len(self.detections)
        boxes = np.array([[d[0], d[1], d[2], d[3]] for d in self.detections], dtype=np.float32)
        scores = np.array([d[4] for d in self.detections], dtype=np.float32)
        classes = np.array([d[5] for d in self.detections], dtype=np.float32)
        return FakeInferResult(
            {
                'num_dets': np.array([[n]], dtype=np.int32),
                'det_boxes': boxes.reshape(1, n, 4) if n else np.zeros((1, 0, 4), dtype=np.float32),
                'det_scores': scores.reshape(1, n),
                'det_classes': classes.reshape(1, n),
            }
        )


class FakePEEncoder:
    def __init__(self) -> None:
        self.embed_crops_calls: list[int] = []

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        self.embed_crops_calls.append(len(crops))
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(crops), 1))

    async def embed_whole_frame(self, path: str) -> np.ndarray | None:  # noqa: ARG002
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _jpeg_bytes(size: tuple[int, int] = (400, 300), seed: int = 0) -> bytes:
    """A textured (non-flat) RGB JPEG so Laplacian-variance blur metrics are non-zero."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def _make_service(
    *,
    detections: list[tuple[float, float, float, float, float, int]] | None = None,
    opensearch: FakeIngestOpenSearch | None = None,
    registry: FakeClassRegistry | None = None,
    confidence_floor: float = 0.5,
) -> tuple[CurationIngestService, FakeIngestOpenSearch, FakeTritonPool]:
    os_fake = opensearch or FakeIngestOpenSearch()
    triton = FakeTritonPool(
        detections if detections is not None else [(0.05, 0.05, 0.6, 0.6, 0.9, 1)]
    )
    reg = registry or FakeClassRegistry([_FakeClassEntry(1, 'widget')])
    profile = DetectionProfile(
        name='primary',
        detector_model='primary_end2end',
        input_size=320,
        confidence_floor=confidence_floor,
        batch_limit=8,
    )
    svc = CurationIngestService(
        opensearch=os_fake,
        triton_pool=triton,
        registry=reg,
        profile=profile,
        pe_encoder=FakePEEncoder(),
        config=CurationConfig(),
    )
    return svc, os_fake, triton


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
        assert doc['coco_proposal_name'] == 'widget'

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
        assert result.error_kind in {'unidentified_image', 'decode_error'}

    @pytest.mark.asyncio
    async def test_no_detections_still_indexes_image_doc(self) -> None:
        svc, os_fake, _ = _make_service(detections=[])
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/empty_scene.jpg')
        assert result.status == 'success'
        assert result.n_crops == 0
        assert len(os_fake.images) == 1


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
