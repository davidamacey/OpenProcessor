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

    The fake honors the *request's* batch dimension — it replies with one
    detection row per input image, exactly as Triton would. Tests assert
    on ``calls`` (one entry per Triton round-trip) and ``batch_sizes`` to
    prove the batch path issues one call for N images rather than N calls.
    """

    def __init__(
        self,
        detections: list[tuple[float, float, float, float, float, int]],
        *,
        fail_on_batch_gt: int | None = None,
    ) -> None:
        self.detections = detections
        self.calls: list[str] = []
        self.batch_sizes: list[int] = []
        self.fail_on_batch_gt = fail_on_batch_gt

    async def infer(self, model_name: str, inputs: list, outputs: list) -> FakeInferResult:  # noqa: ARG002
        batch = int(inputs[0].shape()[0]) if hasattr(inputs[0], 'shape') else 1
        if self.fail_on_batch_gt is not None and batch > self.fail_on_batch_gt:
            raise RuntimeError(f'model does not support batch={batch}')
        self.calls.append(model_name)
        self.batch_sizes.append(batch)
        n = len(self.detections)
        boxes = np.array([[d[0], d[1], d[2], d[3]] for d in self.detections], dtype=np.float32)
        scores = np.array([d[4] for d in self.detections], dtype=np.float32)
        classes = np.array([d[5] for d in self.detections], dtype=np.float32)
        return FakeInferResult(
            {
                'num_dets': np.tile(np.array([[n]], dtype=np.int32), (batch, 1)),
                'det_boxes': (
                    np.tile(boxes.reshape(1, n, 4), (batch, 1, 1))
                    if n
                    else np.zeros((batch, 0, 4), dtype=np.float32)
                ),
                'det_scores': np.tile(scores.reshape(1, n), (batch, 1)),
                'det_classes': np.tile(classes.reshape(1, n), (batch, 1)),
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


def _two_class_registry() -> FakeClassRegistry:
    """Class ids must be dense 0..N-1 — the YOLO label parser rejects
    ``cls_id >= len(registry.classes)``."""
    return FakeClassRegistry([_FakeClassEntry(0, 'gadget'), _FakeClassEntry(1, 'widget')])


def _make_service(
    *,
    detections: list[tuple[float, float, float, float, float, int]] | None = None,
    opensearch: FakeIngestOpenSearch | None = None,
    registry: FakeClassRegistry | None = None,
    confidence_floor: float = 0.5,
    batch_limit: int = 8,
    fail_on_batch_gt: int | None = None,
    detector_version: str = '1',
) -> tuple[CurationIngestService, FakeIngestOpenSearch, FakeTritonPool]:
    os_fake = opensearch or FakeIngestOpenSearch()
    triton = FakeTritonPool(
        detections if detections is not None else [(0.05, 0.05, 0.6, 0.6, 0.9, 1)],
        fail_on_batch_gt=fail_on_batch_gt,
    )
    reg = registry or FakeClassRegistry([_FakeClassEntry(1, 'widget')])
    profile = DetectionProfile(
        name='primary',
        detector_model='primary_end2end',
        detector_version=detector_version,
        input_size=320,
        confidence_floor=confidence_floor,
        batch_limit=batch_limit,
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
        """The unlabeled proposal's ``coco_proposal_name`` came from this
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
        from src.services.curation.ingest import _imohash_bytes

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


class TestBatchLabelImport:
    """Regression guards for G12 — ``ingest_batch`` accepts companion
    ground-truth labels again."""

    @pytest.mark.asyncio
    async def test_batch_imports_companion_labels(self, tmp_path: Any) -> None:
        svc, os_fake, _ = _make_service(registry=_two_class_registry())
        data = _jpeg_bytes(seed=600)
        image_path = tmp_path / 'img.jpg'
        image_path.write_bytes(data)
        label_path = tmp_path / 'img.txt'
        # One label covering the same region the fake detector proposes,
        # agreeing with the detected class.
        label_path.write_text('1 0.325 0.325 0.55 0.55\n')

        result = await svc.ingest_batch(
            [data],
            [str(image_path)],
            label_paths=[str(label_path)],
            label_source='ground_truth',
        )

        assert result.summary.successful == 1
        assert result.summary.labels_imported == 1
        assert any(d.get('label_source') == 'ground_truth' for d in os_fake.items.values())
        # The imported label replaced the detector's class, so the class
        # provenance must follow — not keep claiming the detector.
        [item] = list(os_fake.items.values())
        assert item['class_detector'] == 'ground_truth'
        assert item['class_labeler'] == 'label_import'

    @pytest.mark.asyncio
    async def test_batch_without_label_paths_imports_nothing(self, tmp_path: Any) -> None:
        svc, _, _ = _make_service()
        data = _jpeg_bytes(seed=601)
        image_path = tmp_path / 'img.jpg'
        image_path.write_bytes(data)

        result = await svc.ingest_batch([data], [str(image_path)])
        assert result.summary.labels_imported == 0

    @pytest.mark.asyncio
    async def test_batch_reports_label_vs_detector_mismatches(self, tmp_path: Any) -> None:
        """``detect_mismatches`` surfaces where the detector disagreed with
        ground truth — the report a re-ingest-and-verify pass needs."""
        svc, os_fake, _ = _make_service(registry=_two_class_registry())
        data = _jpeg_bytes(seed=602)
        image_path = tmp_path / 'img.jpg'
        image_path.write_bytes(data)
        label_path = tmp_path / 'img.txt'
        # Same box as the detector's proposal, but class 0 vs detected 1.
        label_path.write_text('0 0.325 0.325 0.55 0.55\n')

        result = await svc.ingest_batch(
            [data],
            [str(image_path)],
            label_paths=[str(label_path)],
            detect_mismatches=True,
        )

        assert result.summary.labels_imported == 1
        assert result.summary.mismatches == 1
        # The ground-truth label still wins; the mismatch is a report only.
        [item] = list(os_fake.items.values())
        assert item['class_id'] == 0

    @pytest.mark.asyncio
    async def test_mismatch_not_counted_when_flag_off(self, tmp_path: Any) -> None:
        svc, _, _ = _make_service(registry=_two_class_registry())
        data = _jpeg_bytes(seed=603)
        image_path = tmp_path / 'img.jpg'
        image_path.write_bytes(data)
        label_path = tmp_path / 'img.txt'
        label_path.write_text('0 0.325 0.325 0.55 0.55\n')

        result = await svc.ingest_batch(
            [data], [str(image_path)], label_paths=[str(label_path)], detect_mismatches=False
        )
        assert result.summary.mismatches == 0

    @pytest.mark.asyncio
    async def test_mismatched_label_paths_length_rejected(self) -> None:
        svc, _, _ = _make_service()
        with pytest.raises(ValueError, match='label_paths'):
            await svc.ingest_batch([b'x', b'y'], ['/a.jpg', '/b.jpg'], label_paths=[None])
