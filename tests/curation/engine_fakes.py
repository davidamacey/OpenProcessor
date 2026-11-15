"""A fake frame engine for the ingest tests: the pool on the engine's address, answering each photo
as the engine would (``src/clients/frame_engine.py`` runs for real in between).

``register`` tells the fake the size of a photo's bytes; ``gates`` holds a photo's answer until the
test releases it (so a test can order the engine's answers against the writes).
"""

from __future__ import annotations

import dataclasses
import json
import struct
from typing import TYPE_CHECKING, Any

import numpy as np

from curation.ingest_fakes import _make_service
from src.domain import frame_engine_wire as wire


if TYPE_CHECKING:
    import asyncio


class _Result:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self.outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray | None:
        return self.outputs.get(name)


def _text(value: str) -> np.ndarray:
    return np.array([value.encode()], dtype=np.object_)


def _one_bytes(inp: Any) -> bytes:
    """The one element of a BYTES input (KServe: a 4-byte length, then the bytes)."""
    raw = inp._raw_content
    (n,) = struct.unpack_from('<I', raw)
    return bytes(raw[4 : 4 + n])


def _axis(dim: int, index: int, value: float) -> np.ndarray:
    out = np.zeros(dim, np.float32)
    out[index] = value
    return out


def engine_result(
    plan: wire.EnginePlan,
    width: int,
    height: int,
    status: str = 'ok',
    jpeg: bytes = b'\xff\xd8crop',
    failed_branch: str | None = None,
    route: str = 'gpu',
    det_class: int = 1,
    embed_dim: int = 3,
) -> _Result:
    if status != 'ok':
        failed = {b: {'status': 'error', 'reason': 'image_failed'} for b in plan.branches}
        return _Result(
            {
                'status': _text(status),
                'decode_route': _text('none'),
                'branch_status': _text(json.dumps(failed)),
            }
        )
    branches: dict[str, dict[str, str]] = {b: {'status': 'ok'} for b in plan.branches}
    if failed_branch is not None:
        branches[failed_branch] = {'status': 'error', 'reason': 'model_error'}
    scale = min(320 / width, 320 / height)
    new_w, new_h = round(width * scale), round(height * scale)
    out = {
        'status': _text('ok'),
        'decode_route': _text(route),
        'branch_status': _text(json.dumps(branches)),
        'size': np.array([width, height], np.int32),
        'letterbox': np.array([scale, (320 - new_w) // 2, (320 - new_h) // 2], np.float64),
        'det_boxes': np.array([[16.0, 16.0, 192.0, 192.0]], np.float32),
        'det_scores': np.array([0.9], np.float32),
        'det_classes': np.array([det_class], np.int32),
        'whole_embedding': _axis(embed_dim, embed_dim - 1, 2.0),
        # frame: n=100, sum=0, squares=500 -> 5.0 ; box: n=10, sum=10, squares=30 -> 2.0
        'blur_sums': np.array([[100, 0, 500], [10, 10, 30]], np.int64),
        'crop_jpegs': np.array([jpeg], dtype=np.object_),
    }
    if plan.runs('embed_crops'):
        # as the engine does: only the picked detections are embedded, their rows come first
        picked = plan.embed_classes is None or det_class in plan.embed_classes
        out['crop_embeddings'] = np.array(
            [_axis(embed_dim, 1, 3.0)] if picked else [], np.float32
        ).reshape(-1, embed_dim)
        out['crop_embedded'] = np.array([picked])
    return _Result(out)


class Engine:
    """The pool on the frame engine's address."""

    def __init__(self, status: str, sizes: dict[bytes, tuple[int, int]]) -> None:
        self.status = status
        self.sizes = sizes
        self.calls: list[dict[str, Any]] = []
        self.ready = True
        self.readiness_probes = 0
        self.failed_branch: str | None = None
        self.raises: Exception | None = None
        self.route = 'gpu'
        self.gates: dict[bytes, asyncio.Event] = {}
        self.answered: list[bytes] = []
        # a test that does not care which photo is which: every bytes is an image of this size
        self.any_size: tuple[int, int] | None = None
        self.embed_dim = 3

    async def is_model_ready(self, model: str) -> bool:
        assert model == wire.MODEL_NAME
        self.readiness_probes += 1
        return self.ready

    async def infer(self, model: str, inputs: list, outputs: list, **_k: Any) -> _Result:
        assert model == wire.MODEL_NAME
        data = _one_bytes(inputs[0])
        plan = wire.parse_plan(_one_bytes(inputs[1]), has_boxes=len(inputs) > 2)
        self.calls.append({'nbytes': len(data), 'plan': plan})
        if self.raises is not None:
            raise self.raises
        gate = self.gates.get(data)
        if gate is not None:
            await gate.wait()
        if data not in self.sizes and self.any_size is None:  # bytes nobody registered are no image
            return engine_result(plan, 0, 0, 'corrupt')
        w, h = self.sizes.get(data) or self.any_size  # type: ignore[misc]
        self.answered.append(data)
        return engine_result(
            plan,
            w,
            h,
            self.status,
            failed_branch=self.failed_branch,
            route=self.route,
            embed_dim=self.embed_dim,
        )


def engine_service(tmp_path: Any, *, status: str = 'ok', **service_kwargs: Any):
    """``(service, opensearch, triton, engine, sizes)``: the ingest service with the engine on."""
    svc, os_fake, triton = _make_service(**service_kwargs)
    svc.config = dataclasses.replace(
        svc.config, ingest_gpu_decode=True, crop_cache_dir=tmp_path / 'crops'
    )
    sizes: dict[bytes, tuple[int, int]] = {}
    engine = Engine(status, sizes)
    svc.frame_engine = engine
    triton.loaded = []

    async def is_model_ready(model: str) -> bool:
        return model in triton.loaded

    async def load_model(model: str) -> None:
        triton.loaded.append(model)

    triton.is_model_ready = is_model_ready
    triton.load_model = load_model
    return svc, os_fake, triton, engine, sizes


def register(sizes: dict[bytes, tuple[int, int]], data: bytes, size: tuple[int, int]) -> bytes:
    sizes[data] = size
    return data
