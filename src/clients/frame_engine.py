"""The API's client of the frame engine (``frame_engine/``, protocol version 1).

One unary KServe v2 ``ModelInfer`` per image to the model ``frame_engine``
(:mod:`src.domain.frame_engine_wire`), sent through an :class:`AsyncTritonPool` opened on the
engine's address (``OP_FRAME_ENGINE_URL``): the same channels, back-pressure, router and bounded
retry as every Triton call. The engine answers an overload (its device-memory budget or its
shared-memory region stayed full for its bounded wait, or too many requests in flight) with gRPC
``RESOURCE_EXHAUSTED``; the pool retries that status, and an UNAVAILABLE engine, ``RETRIES`` times
with its backoff. What is left after the retries is an :class:`EngineCallError`, whose
``overloaded`` says whether the last answer was the overload status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np

from src.config.env import env_str
from src.domain import frame_engine_wire as wire
from src.utils.retry import TRITON_CALL_ERRORS


if TYPE_CHECKING:
    from src.clients.triton_pool_core import AsyncTritonPool


FRAME_ENGINE_URL_ENV = 'OP_FRAME_ENGINE_URL'
DEFAULT_URL = 'frame-engine:8001'
#: Retries after the first try; with the engine's 4 s budget wait the four tries stay inside the
#: 20 s the Triton prep model waited before it answered gpu_memory.
RETRIES = 3
#: One photo's whole pipeline (decode, detector, encoder, crop JPEGs) plus the engine's waits.
TIMEOUT_SECONDS = 120.0
_OVERLOAD = str(grpc.StatusCode.RESOURCE_EXHAUSTED)
#: The pool also raises RuntimeError: not initialized, or no server for the model.
_CALL_ERRORS: tuple[type[Exception], ...] = (*TRITON_CALL_ERRORS, RuntimeError)


def frame_engine_url() -> str:
    return env_str(FRAME_ENGINE_URL_ENV, DEFAULT_URL)


async def open_pool(*, pool_size: int, max_concurrent: int) -> AsyncTritonPool:
    """An initialized pool on the engine's address (one server: the router's single default)."""
    from src.clients.triton_pool_core import AsyncTritonPool

    pool = AsyncTritonPool(
        url=frame_engine_url(), pool_size=pool_size, max_concurrent=max_concurrent
    )
    await pool.initialize()
    return pool


class EngineCallError(Exception):
    """The engine gave no answer for the image: down, failed, or still overloaded after the
    retries (``overloaded``)."""

    def __init__(self, message: str, *, overloaded: bool) -> None:
        self.overloaded = overloaded
        super().__init__(message)


@dataclass(frozen=True)
class EngineAnswer:
    """One image's answer: ``status`` is ``ok`` or the reason it was not decoded
    (:data:`src.domain.decode_route.PREP_STATUSES`); ``outputs`` holds the branch tensors."""

    status: str
    decode_route: str
    branches: dict[str, tuple[wire.BranchState, str]]
    outputs: dict[str, Any]

    @property
    def ok(self) -> bool:
        """Decoded, and every requested branch ran."""
        return self.status == wire.STATUS_OK and all(
            state == 'ok' for state, _ in self.branches.values()
        )


def _bytes_input(name: str, value: bytes) -> Any:
    from tritonclient.grpc import InferInput

    inp = InferInput(name, [1], 'BYTES')
    inp.set_data_from_numpy(np.array([value], dtype=np.object_))
    return inp


def _boxes_input(boxes: np.ndarray) -> Any:
    from tritonclient.grpc import InferInput

    rows = np.ascontiguousarray(boxes, np.float64).reshape(-1, 4)
    inp = InferInput(wire.BOXES, list(rows.shape), wire.INPUT_DATATYPES[wire.BOXES])
    inp.set_data_from_numpy(rows)
    return inp


def _text(result: Any, name: str) -> str:
    value = result.as_numpy(name)[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def _answer(result: Any, plan: wire.EnginePlan) -> EngineAnswer:
    branches = wire.decode_branch_status(_text(result, wire.BRANCH_STATUS))
    outputs: dict[str, Any] = {}
    status = _text(result, wire.STATUS)
    if status == wire.STATUS_OK:
        for name in (wire.SIZE, *wire.expected_outputs(plan)):
            value = result.as_numpy(name)
            if value is None:
                continue  # the branch that makes it failed; its branch_status says so
            outputs[name] = (
                [bytes(v) for v in value.reshape(-1)]
                if name in (wire.CROP_JPEGS, wire.DERIVATIVE_JPEGS)
                else value
            )
    return EngineAnswer(status, _text(result, wire.DECODE_ROUTE), branches, outputs)


async def process(
    pool: AsyncTritonPool, data: bytes, plan: wire.EnginePlan, boxes: np.ndarray | None = None
) -> EngineAnswer:
    """The engine's answer for the encoded image ``data`` under ``plan``.

    Raises:
        EngineCallError: no answer after the retries (``overloaded`` when the last one was the
            overload status).
    """
    from tritonclient.utils import InferenceServerException

    inputs = [_bytes_input(wire.IMAGE, data), _bytes_input(wire.PLAN, plan.to_json().encode())]
    if boxes is not None:
        inputs.append(_boxes_input(boxes))
    try:
        result = await pool.infer(
            wire.MODEL_NAME, inputs, [], timeout=TIMEOUT_SECONDS, retries=RETRIES
        )
    except _CALL_ERRORS as exc:
        overloaded = isinstance(exc, InferenceServerException) and exc.status() == _OVERLOAD
        raise EngineCallError(str(exc), overloaded=overloaded) from exc
    return _answer(result, plan)


__all__ = [
    'DEFAULT_URL',
    'FRAME_ENGINE_URL_ENV',
    'RETRIES',
    'EngineAnswer',
    'EngineCallError',
    'frame_engine_url',
    'open_pool',
    'process',
]
