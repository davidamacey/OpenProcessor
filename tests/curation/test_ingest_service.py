"""Tests for src.services.curation.ingest.CurationIngestService."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from curation.occ_fakes import FakeIngestOpenSearch
from src.config import CurationConfig, DetectionProfile
from src.services.curation.ingest import MAX_INGEST_CONCURRENCY, CurationIngestService


del MAX_INGEST_CONCURRENCY  # imported only to confirm the module exports it

# The secondary detector's client-side NMS imports a YOLOv5 fork; point it at
# the test fixture fork (see tests/curation/test_ensemble_nms.py).
os.environ.setdefault(
    'DETECTION_YOLOV5_FORK',
    str(Path(__file__).resolve().parent.parent / 'fixtures' / 'yolov5_fork'),
)


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
    # These tests exercise a primary whose label space IS the registry.
    profile = DetectionProfile(
        name='primary',
        detector_model='primary_end2end',
        assigns_class=True,
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

        return await get_event_hub().subscribe()

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

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_batch_label_import_seeds_items_it_creates(self, tmp_path: Any) -> None:
        """A label the detector missed becomes a new item — it needs region
        detection as much as a detector-created one."""
        from src.config import RegionStatus

        svc, os_fake, _ = _make_service(detections=[], registry=_two_class_registry())
        data = _jpeg_bytes(seed=900)
        image_path = tmp_path / 'img.jpg'
        image_path.write_bytes(data)
        label_path = tmp_path / 'img.txt'
        label_path.write_text('1 0.5 0.5 0.2 0.2\n')

        result = await svc.ingest_batch([data], [str(image_path)], label_paths=[str(label_path)])

        assert result.summary.labels_imported == 1
        [item] = list(os_fake.items.values())
        assert item[self._status_field()] == RegionStatus.PENDING_DETECTION.value

    @pytest.mark.usefixtures('reference_region_profile')
    def test_ingest_response_reports_seeded_count(self) -> None:
        from src.routers.curation.ingest import _batch_response
        from src.services.curation.ingest_models import BatchIngestResult, IngestResult

        batch = BatchIngestResult(
            results=[IngestResult(image_path='/a.jpg', n_crops=3, n_region_queued=2)]
        )
        [resp] = _batch_response(batch, []).results
        assert resp.n_plates == 2


# =============================================================================
# N3 — backbone embedding from a dual-head secondary detector
# =============================================================================

_SECONDARY_MODEL = 'secondary_raw'
_FEATURE_DIM = 8
_GRID = 10  # 320 px secondary input / stride 32


class FakeDualTritonPool(FakeTritonPool):
    """Primary end2end detector plus a raw-output secondary detector that
    optionally exposes a backbone feature map (``sppf_feat``).

    Like Triton, asking for an output the model does not have is an error,
    so a caller that requests ``sppf_feat`` from a single-head model fails
    loudly here rather than silently succeeding.
    """

    def __init__(
        self,
        detections: list[tuple[float, float, float, float, float, int]],
        *,
        secondary_raw: np.ndarray,
        feature_map: np.ndarray | None,
        has_probe: bool = True,
        probe_error: bool = False,
    ) -> None:
        super().__init__(detections)
        self.secondary_raw = secondary_raw
        self.feature_map = feature_map
        self.requested: list[list[str]] = []
        self.probe_error = probe_error
        if has_probe:
            self.get_model_output_names = self._output_names

    async def _output_names(self, model_name: str) -> list[str]:
        if self.probe_error:
            raise RuntimeError('metadata unavailable')
        if model_name == _SECONDARY_MODEL:
            return ['output0'] + (['sppf_feat'] if self.feature_map is not None else [])
        return ['num_dets', 'det_boxes', 'det_scores', 'det_classes']

    async def infer(self, model_name: str, inputs: list, outputs: list) -> FakeInferResult:
        if model_name != _SECONDARY_MODEL:
            return await super().infer(model_name, inputs, outputs)
        batch = int(inputs[0].shape()[0])
        names = [o.name() for o in outputs]
        self.calls.append(model_name)
        self.batch_sizes.append(batch)
        self.requested.append(names)
        out = {'output0': np.tile(self.secondary_raw[None], (batch, 1, 1))}
        if 'sppf_feat' in names:
            if self.feature_map is None:
                raise RuntimeError("unexpected inference output 'sppf_feat'")
            out['sppf_feat'] = np.tile(self.feature_map[None], (batch, 1, 1, 1))
        return FakeInferResult(out)


def _secondary_raw() -> np.ndarray:
    """``(N, 5 + nc)`` YOLOv5 rows in 320-px letterbox space. Row 0 matches
    the primary box (letterbox 16..176) and votes class 0; the rest are
    below any confidence floor."""
    rows = np.zeros((4, 7), dtype=np.float32)
    rows[0] = [96.0, 96.0, 160.0, 160.0, 0.95, 1.0, 0.0]
    return rows


def _feature_map(channels: int = _FEATURE_DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((channels, _GRID, _GRID)).astype(np.float32)


def _make_dual_service(
    *,
    feature_map: np.ndarray | None,
    backbone_dim: int = _FEATURE_DIM,
    has_probe: bool = True,
    probe_error: bool = False,
) -> tuple[CurationIngestService, FakeIngestOpenSearch, FakeDualTritonPool]:
    os_fake = FakeIngestOpenSearch()
    triton = FakeDualTritonPool(
        [(0.05, 0.05, 0.55, 0.55, 0.9, 1)],
        secondary_raw=_secondary_raw(),
        feature_map=feature_map,
        has_probe=has_probe,
        probe_error=probe_error,
    )
    svc = CurationIngestService(
        opensearch=os_fake,
        triton_pool=triton,
        registry=_two_class_registry(),
        profile=DetectionProfile(
            name='primary',
            detector_model='primary_end2end',
            assigns_class=True,
            input_size=320,
            batch_limit=8,
        ),
        secondary_profile=DetectionProfile(
            name='secondary',
            detector_model=_SECONDARY_MODEL,
            detector_version='3',
            input_size=320,
            confidence_floor=0.5,
            batch_limit=8,
        ),
        pe_encoder=FakePEEncoder(),
        config=CurationConfig(backbone_embedding_dim=backbone_dim),
    )
    return svc, os_fake, triton


# The primary box (0.05..0.55 of the 320-px input) in the secondary's own
# letterbox space — both detectors share a 320-px input here, so it is the
# same square the primary saw. Edges sit mid-cell (stride 32) so float
# rounding can't move the pooled grid window.
_ITEM_LETTERBOX_BOX = (16.0, 16.0, 176.0, 176.0)


class TestBackboneEmbedding:
    @pytest.mark.asyncio
    async def test_dual_head_secondary_writes_pooled_backbone_embedding(self) -> None:
        from src.config import BACKBONE_EMBEDDING_FIELD
        from src.services.detection.geometry import roi_pool_sppf

        fmap = _feature_map()
        svc, os_fake, triton = _make_dual_service(feature_map=fmap)
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/dual.jpg')
        assert result.status == 'success'

        assert triton.requested == [['output0', 'sppf_feat']]
        [doc] = list(os_fake.items.values())
        expected = roi_pool_sppf(fmap, _ITEM_LETTERBOX_BOX, input_size=320, target_dim=8)
        assert len(doc[BACKBONE_EMBEDDING_FIELD]) == svc.config.backbone_embedding_dim
        assert doc[BACKBONE_EMBEDDING_FIELD] == pytest.approx(expected.tolist(), abs=1e-6)
        # Secondary override also carries its own class provenance (N1).
        assert doc['class_id'] == 0
        assert doc['class_detector'] == _SECONDARY_MODEL
        assert doc['class_detector_version'] == '3'

    @pytest.mark.asyncio
    async def test_single_head_secondary_is_called_exactly_as_before(self) -> None:
        from src.config import BACKBONE_EMBEDDING_FIELD

        svc, os_fake, triton = _make_dual_service(feature_map=None)
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/single.jpg')

        assert result.status == 'success'
        assert triton.requested == [['output0']]
        [doc] = list(os_fake.items.values())
        assert BACKBONE_EMBEDDING_FIELD not in doc
        assert doc['class_id'] == 0  # class override still applied

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('has_probe', 'probe_error'), [(False, False), (True, True)])
    async def test_unknown_outputs_never_request_the_feature_map(
        self, has_probe: bool, probe_error: bool
    ) -> None:
        """No way to learn the model's outputs (pool without a metadata
        probe, or the probe failing) must fall back to ``output0`` only."""
        from src.config import BACKBONE_EMBEDDING_FIELD

        svc, os_fake, triton = _make_dual_service(
            feature_map=_feature_map(), has_probe=has_probe, probe_error=probe_error
        )
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/noprobe.jpg')

        assert result.status == 'success'
        assert triton.requested == [['output0']]
        [doc] = list(os_fake.items.values())
        assert BACKBONE_EMBEDDING_FIELD not in doc

    @pytest.mark.asyncio
    async def test_batched_path_writes_identical_embeddings_in_one_call(self) -> None:
        from src.config import BACKBONE_EMBEDDING_FIELD

        images = [_jpeg_bytes(seed=800 + s) for s in range(3)]
        paths = [f'/tmp/bb{s}.jpg' for s in range(3)]
        fmap = _feature_map()

        svc_batch, os_batch, triton = _make_dual_service(feature_map=fmap)
        await svc_batch.ingest_batch(images, paths)
        assert triton.calls.count(_SECONDARY_MODEL) == 1
        assert triton.requested == [['output0', 'sppf_feat']]
        assert triton.batch_sizes[triton.calls.index(_SECONDARY_MODEL)] == 3

        svc_single, os_single, _ = _make_dual_service(feature_map=fmap)
        for data, path in zip(images, paths, strict=True):
            await svc_single.ingest_one(data, path, source='batch')

        volatile = {'created_at', 'updated_at', 'class_labeled_at'}

        def _comparable(store: dict) -> list[dict]:
            return sorted(
                ({k: v for k, v in d.items() if k not in volatile} for d in store.values()),
                key=lambda d: d['crop_id'],
            )

        assert all(BACKBONE_EMBEDDING_FIELD in d for d in os_batch.items.values())
        assert _comparable(os_batch.items) == _comparable(os_single.items)

    @pytest.mark.asyncio
    async def test_narrower_feature_map_is_zero_padded_to_the_mapping_dim(self) -> None:
        from src.config import BACKBONE_EMBEDDING_FIELD

        svc, os_fake, _ = _make_dual_service(feature_map=_feature_map(), backbone_dim=12)
        await svc.ingest_one(_jpeg_bytes(), '/tmp/pad.jpg')

        [doc] = list(os_fake.items.values())
        vec = doc[BACKBONE_EMBEDDING_FIELD]
        assert len(vec) == 12
        assert vec[_FEATURE_DIM:] == [0.0] * (12 - _FEATURE_DIM)
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_wider_feature_map_than_mapping_dim_is_skipped_not_truncated(self) -> None:
        """Truncating channels would write a vector that silently means
        something else; skip the field (and log) instead. Ingest still
        succeeds."""
        from src.config import BACKBONE_EMBEDDING_FIELD

        svc, os_fake, _ = _make_dual_service(feature_map=_feature_map(), backbone_dim=4)
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/wide.jpg')

        assert result.status == 'success'
        [doc] = list(os_fake.items.values())
        assert BACKBONE_EMBEDDING_FIELD not in doc
        assert doc['class_id'] == 0

    @pytest.mark.asyncio
    async def test_nms_failure_keeps_embedding_and_primary_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The class override and the embedding pooling fail independently."""
        from src.config import BACKBONE_EMBEDDING_FIELD
        from src.services.detection import ensemble_nms

        def _broken_nms(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError('nms fork unavailable')

        monkeypatch.setattr(ensemble_nms, 'apply_ensemble_nms', _broken_nms)
        svc, os_fake, _ = _make_dual_service(feature_map=_feature_map())
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/nonms.jpg')

        assert result.status == 'success'
        [doc] = list(os_fake.items.values())
        assert doc['class_id'] == 1  # primary's class, no override
        assert doc['class_detector'] == 'primary_end2end'
        assert len(doc[BACKBONE_EMBEDDING_FIELD]) == _FEATURE_DIM
