"""A fake frame engine for the ingest tests: the pool on the engine's address, answering each photo
as the engine would (``src/clients/frame_engine.py`` runs for real in between).

``register`` tells the fake the size of a photo's bytes; ``gates`` holds a photo's answer until the
test releases it (so a test can order the engine's answers against the writes).
"""

from __future__ import annotations

import dataclasses
import io
import struct
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from curation.engine_answer import engine_answer, failed_answer
from curation.ingest_fakes import _make_service
from src.domain import frame_engine_wire as wire


if TYPE_CHECKING:
    import asyncio


class _Result:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self.outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray | None:
        return self.outputs.get(name)


def _one_bytes(inp: Any) -> bytes:
    """The one element of a BYTES input (KServe: a 4-byte length, then the bytes)."""
    raw = inp._raw_content
    (n,) = struct.unpack_from('<I', raw)
    return bytes(raw[4 : 4 + n])


def _decoded_size(data: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.size
    except (OSError, ValueError):
        return None


def _text(value: str) -> np.ndarray:
    return np.array([value.encode()], dtype=np.object_)


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
    """The tensors of one answer as the client reads them (the text ones as one BYTES element)."""
    answer = (
        failed_answer(plan, status)
        if status != 'ok'
        else engine_answer(
            plan,
            width,
            height,
            jpeg=jpeg,
            failed_branch=failed_branch,
            route=route,
            det_class=det_class,
            embed_dim=embed_dim,
        )
    )
    return _Result(
        {
            wire.STATUS: _text(answer.status),
            wire.DECODE_ROUTE: _text(answer.decode_route),
            wire.BRANCH_STATUS: _text(wire.encode_branch_status(answer.branches)),
            **{
                name: np.array(value, dtype=np.object_)
                if name == wire.CROP_JPEGS
                else np.asarray(value)
                for name, value in answer.outputs.items()
            },
        }
    )


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
        # a test that sends real files: the photo's size is read from its bytes (undecodable
        # bytes are no image), so the tests need not register each one
        self.read_sizes = False
        self.embed_dim = 3
        self.det_class = 1

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
        size = self.sizes.get(data) or (_decoded_size(data) if self.read_sizes else None)
        if size is None:  # bytes nobody registered are no image
            return engine_result(plan, 0, 0, 'corrupt')
        w, h = size
        self.answered.append(data)
        return engine_result(
            plan,
            w,
            h,
            self.status,
            failed_branch=self.failed_branch,
            route=self.route,
            embed_dim=self.embed_dim,
            det_class=self.det_class,
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
