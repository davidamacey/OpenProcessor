"""Shared builders for the reprocess tests: a fake OpenSearch holding the
bound project's items/images indexes, region-shaped item docs, a servable
image root and an ingest service with fake Triton/PE."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from src.config import DetectionProfile, RegionStatus, get_curation_config
from src.config.region_fields import get_region_fields
from src.services.curation.ingest import CurationIngestService


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


F = get_region_fields()

IMPORT = 'import'


def items_index() -> str:
    return get_curation_config().items_index


def images_index() -> str:
    return get_curation_config().images_index


def box(
    box_id: str = 'b1',
    *,
    state: str = 'rejected',
    source: str = 'detector',
    detector: str | None = 'det_a',
    reason: str | None = 'aspect',
    bbox: tuple[float, float, float, float] = (0.2, 0.2, 0.4, 0.3),
    score: float = 0.4,
) -> dict[str, Any]:
    return {
        'box_id': box_id,
        'bbox_norm': list(bbox),
        'state': state,
        'score': score,
        'detector': detector,
        'detector_version': '1' if detector else None,
        'source': source,
        'rejection_reason': reason,
    }


def item(
    crop_id: str,
    status: str | None = None,
    *,
    image_id: str = 'img-1',
    boxes: tuple[dict[str, Any], ...] = (),
    validated: bool | None = None,
    verifier: str | None = None,
    class_source: str = 'unlabeled_proposal',
    class_id: int | None = None,
    class_name: str | None = None,
    class_validated: bool = False,
    test_holdout: bool = False,
    history: list[dict[str, Any]] | None = None,
    image_path: str | None = None,
    bbox_norm: tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.9),
    **extra: Any,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'image_id': image_id,
        'image_path': image_path or f'/nowhere/{crop_id}.jpg',
        'bbox_norm': list(bbox_norm),
        'class_source': class_source,
        'class_validated': class_validated,
        'test_holdout': test_holdout,
        F.boxes: list(boxes),
        F.count: sum(1 for b in boxes if b['state'] == 'accepted'),
        F.rejected_count: sum(1 for b in boxes if b['state'] == 'rejected'),
    }
    if status is not None:
        doc[F.status] = status
    if validated is not None:
        doc[F.validated] = validated
    if verifier is not None:
        doc[F.verifier] = verifier
    if class_id is not None:
        doc['class_id'] = class_id
        doc['class_name'] = class_name
    if history is not None:
        doc['class_id_history'] = history
    doc.update(extra)
    return doc


def make_fake(
    items: list[dict[str, Any]], images: list[dict[str, Any]] | None = None
) -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            items_index(): {d['crop_id']: d for d in items},
            images_index(): {d['image_id']: d for d in images or []},
        }
    )


def docs(fake: QueryFakeOpenSearch) -> dict[str, dict[str, Any]]:
    return fake.docs(items_index())


def servable_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Declare ``tmp_path`` the only configured source root (the guard every
    stored ``image_path`` must pass before a reprocess reads it)."""
    from src.services.curation import image_serving

    root = tmp_path.resolve()
    monkeypatch.setattr(image_serving, '_configured_roots', lambda config=None: (root,))  # noqa: ARG005
    return root


def jpeg_bytes(size: tuple[int, int] = (400, 300), seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode='RGB').save(buf, format='JPEG', quality=95)
    return buf.getvalue()


class FakeInferResult:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self._outputs[name]


class FakeTriton:
    """End2end detections (x1, y1, x2, y2, score, class_id) in network-input
    coordinates; ``detections`` can be swapped between calls."""

    def __init__(self, detections: list[tuple[float, float, float, float, float, int]]) -> None:
        self.detections = detections
        self.calls = 0

    async def infer(self, model_name: str, inputs: list, outputs: list) -> FakeInferResult:  # noqa: ARG002
        self.calls += 1
        batch = int(inputs[0].shape()[0]) if hasattr(inputs[0], 'shape') else 1
        n = len(self.detections)
        boxes = np.array([d[:4] for d in self.detections], dtype=np.float32).reshape(1, n, 4)
        return FakeInferResult(
            {
                'num_dets': np.tile(np.array([[n]], dtype=np.int32), (batch, 1)),
                'det_boxes': np.tile(boxes, (batch, 1, 1)),
                'det_scores': np.tile(
                    np.array([d[4] for d in self.detections], dtype=np.float32).reshape(1, n),
                    (batch, 1),
                ),
                'det_classes': np.tile(
                    np.array([d[5] for d in self.detections], dtype=np.float32).reshape(1, n),
                    (batch, 1),
                ),
            }
        )


class FakePE:
    def __init__(self) -> None:
        self.crop_calls = 0
        self.frame_calls = 0

    async def embed_crops(self, crops: list[Any], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        self.crop_calls += len(crops)
        return np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(crops), 1))

    async def embed_whole_frame(self, path: str) -> np.ndarray | None:  # noqa: ARG002
        self.frame_calls += 1
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


class _Entry:
    def __init__(self, class_id: int, class_name: str) -> None:
        self.class_id = class_id
        self.class_name = class_name
        self.deprecated = False


class _RegFile:
    classes = [_Entry(0, 'gadget'), _Entry(1, 'widget')]


class FakeRegistry:
    def load(self) -> _RegFile:
        return _RegFile()

    def get(self, class_id: int) -> _Entry | None:
        return next((c for c in _RegFile.classes if c.class_id == class_id), None)


def make_service(
    fake: QueryFakeOpenSearch, triton: FakeTriton, pe: FakePE | None = None
) -> CurationIngestService:
    profile = DetectionProfile(
        name='primary',
        detector_model='primary_end2end',
        assigns_class=True,
        detector_version='1',
        input_size=320,
        confidence_floor=0.5,
        batch_limit=8,
    )
    return CurationIngestService(
        opensearch=fake,
        triton_pool=triton,
        registry=FakeRegistry(),
        profile=profile,
        pe_encoder=pe or FakePE(),
        config=get_curation_config(),
    )


__all__ = [
    'IMPORT',
    'F',
    'FakePE',
    'FakeRegistry',
    'FakeTriton',
    'RegionStatus',
    'box',
    'docs',
    'images_index',
    'item',
    'items_index',
    'jpeg_bytes',
    'make_fake',
    'make_service',
    'servable_root',
]
