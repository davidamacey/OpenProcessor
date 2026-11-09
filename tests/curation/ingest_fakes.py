"""Fakes shared by the ingest tests: class registry, Triton pool, PE encoder and the service factory."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from fused_output import GpuDecodeAbsent, fused_tensor
from PIL import Image

from curation.occ_fakes import FakeIngestOpenSearch
from src.config import CurationConfig, DetectionProfile
from src.services.curation.ingest import CurationIngestService


#: The network input size of the primary profile these tests build.
PRIMARY_NET = 320


@dataclass
class _FakeClassEntry:
    class_id: int
    class_name: str
    deprecated: bool = False
    group: str | None = None


class FakeClassRegistry:
    """The registry snapshot the ingest service is handed: ``classes`` and ``get``."""

    def __init__(self, entries: list[_FakeClassEntry] | None = None) -> None:
        self.classes = entries or []

    def get(self, class_id: int) -> _FakeClassEntry | None:
        for c in self.classes:
            if c.class_id == class_id:
                return c
        return None


class FakeInferResult:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self._outputs[name]


class FakeTritonPool(GpuDecodeAbsent):
    """Returns the pre-scripted fused ``output0`` tensor for the primary detector.

    ``detections`` is a list of ``(x1, y1, x2, y2, score, class_id)`` with
    box coordinates as fractions of the network input square; the fake scales
    them to the letterbox pixels (``net`` = the primary profile's ``input_size``)
    the real engine emits.

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

    async def get_model_input_datatype(self, model_name: str) -> str:  # noqa: ARG002
        return 'UINT8'

    async def infer(self, model_name: str, inputs: list, outputs: list) -> FakeInferResult:  # noqa: ARG002
        batch = int(inputs[0].shape()[0]) if hasattr(inputs[0], 'shape') else 1
        if self.fail_on_batch_gt is not None and batch > self.fail_on_batch_gt:
            raise RuntimeError(f'model does not support batch={batch}')
        self.calls.append(model_name)
        self.batch_sizes.append(batch)
        return FakeInferResult(
            {'output0': fused_tensor(self.detections, net=PRIMARY_NET, batch=batch)}
        )


class FakePEEncoder:
    def __init__(self) -> None:
        self.embed_crops_calls: list[int] = []

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        self.embed_crops_calls.append(len(crops))
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(crops), 1))

    async def embed_whole_frame_image(self, img: Any) -> np.ndarray:  # noqa: ARG002
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
        input_size=PRIMARY_NET,
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
