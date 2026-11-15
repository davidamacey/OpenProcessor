"""The API's client of the frame engine (``frame_engine/``, protocol version 1).

One unary KServe v2 ``ModelInfer`` per image to the model ``frame_engine``
(:mod:`src.domain.frame_engine_wire`), sent through an :class:`AsyncTritonPool` opened on the
engine's address (``OP_FRAME_ENGINE_URL``): the same channels, back-pressure and router as every
Triton call. The engine answers an overload (its device-memory budget or its shared-memory region
stayed full for its bounded wait, or too many requests in flight) with gRPC
``RESOURCE_EXHAUSTED``. A photo is a unit of work that waits its turn: the overload is retried
with a jittered exponential backoff until the photo's deadline (``DEADLINE_SECONDS``; tenacity
``stop_after_delay`` and ``wait_exponential_jitter``,
https://tenacity.readthedocs.io/en/stable/#stopping), an UNAVAILABLE engine or a call lost in transit ``RETRIES`` times.
The pool's own retry is off (``retries=0``): its 10 ms steps do not outlast a busy card. What is
left is an :class:`EngineCallError`, whose ``overloaded`` says whether the last answer was the
overload status.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_exponential_jitter,
)

from src.config.env import env_str
from src.domain import frame_engine_wire as wire
from src.utils.retry import TRITON_CALL_ERRORS


if TYPE_CHECKING:
    from src.clients.triton_pool_core import AsyncTritonPool


FRAME_ENGINE_URL_ENV = 'OP_FRAME_ENGINE_URL'
DEFAULT_URL = 'frame-engine:8001'
#: Retries of an UNAVAILABLE engine or a call lost in transit after the first try (a restarting
#: engine is the caller's to wait out: the bulk run's 503 backoff); an overload is not counted,
#: the deadline ends it.
RETRIES = 3
#: One photo's whole pipeline (decode, detector, encoder, crop JPEGs) plus the engine's waits: the
#: timeout of one try.
TIMEOUT_SECONDS = 120.0
#: How long a photo waits for the engine across overload retries.
DEADLINE_SECONDS = 120.0
RETRY_SLEEP = asyncio.sleep
MIB = 1024 * 1024
#: The API's channels to the engine, defined here and nowhere else (the engine's own server side is
#: ``frame_engine.front.SERVER_OPTIONS``; ``tests/frame_engine_tests/test_keepalive.py`` pins the
#: two to agree). gRPC keepalive guide and the C-core keepalive document
#: (https://github.com/grpc/grpc/blob/master/doc/keepalive.md): a client that pings while no call
#: is in flight needs ``keepalive_permit_without_calls`` on the server too, and its
#: ``keepalive_time_ms`` may not be below the server's
#: ``http2.min_ping_interval_without_data_ms`` (the document's "Received a GOAWAY with
#: ENHANCE_YOUR_CALM" answers); a client that gets that GOAWAY doubles its keepalive time
#: (gRFC A8). ``http2.max_pings_without_data`` is 0 (no limit): at its default of 2 an idle
#: channel stops pinging after two pings, so a dead engine is not noticed until the next call.
ENGINE_KEEPALIVE_TIME_MS = 30000
ENGINE_KEEPALIVE_TIMEOUT_MS = 10000
ENGINE_CHANNEL_OPTIONS: tuple[tuple[str, int], ...] = (
    ('grpc.keepalive_time_ms', ENGINE_KEEPALIVE_TIME_MS),
    ('grpc.keepalive_timeout_ms', ENGINE_KEEPALIVE_TIMEOUT_MS),
    ('grpc.keepalive_permit_without_calls', 1),
    ('grpc.http2.max_pings_without_data', 0),
    # a photo and its crop JPEGs, far under this
    ('grpc.max_send_message_length', 512 * MIB),
    ('grpc.max_receive_message_length', 512 * MIB),
    # a restarted engine is reached within about the reconnect cap (gRPC connection backoff:
    # the default delay grows towards 120 s) and its recreated address is found again at once
    ('grpc.initial_reconnect_backoff_ms', 250),
    ('grpc.min_reconnect_backoff_ms', 250),
    ('grpc.max_reconnect_backoff_ms', 3000),
    ('grpc.dns_min_time_between_resolutions_ms', 1000),
    ('grpc.http2.initial_stream_window_size', 1 * MIB),
    ('grpc.http2.initial_connection_window_size', 2 * MIB),
    ('grpc.http2.bdp_probe', 0),
)
_BACKOFF = wait_exponential_jitter(multiplier=0.05, max=2.0, jitter=0.2)
_OVERLOAD = str(grpc.StatusCode.RESOURCE_EXHAUSTED)
_UNAVAILABLE = str(grpc.StatusCode.UNAVAILABLE)
_INTERNAL = str(grpc.StatusCode.INTERNAL)
#: The pool also raises RuntimeError: not initialized, or no server for the model.
_CALL_ERRORS: tuple[type[Exception], ...] = (*TRITON_CALL_ERRORS, RuntimeError)


def frame_engine_url() -> str:
    return env_str(FRAME_ENGINE_URL_ENV, DEFAULT_URL)


async def open_pool(*, pool_size: int, max_concurrent: int) -> AsyncTritonPool:
    """An initialized pool on the engine's address (one server: the router's single default)."""
    from src.clients.triton_pool_core import AsyncTritonPool

    pool = AsyncTritonPool(
        url=frame_engine_url(),
        pool_size=pool_size,
        max_concurrent=max_concurrent,
        channel_options=ENGINE_CHANNEL_OPTIONS,
    )
    await pool.initialize()
    return pool


#: The readiness probe's budget. A probe the transport refuses (a GOAWAY on its channel, a reset
#: connection) is not the engine saying "not ready": it is asked again, on the next channel, up to
#: ``PROBE_ATTEMPTS`` times, ``wait_exponential`` apart (0.1 s, 0.2 s; tenacity,
#: https://tenacity.readthedocs.io/en/stable/#waiting-before-retrying). An engine that refuses
#: connections is judged in about 0.3 s; one that accepts and never answers in at most
#: ``PROBE_ATTEMPTS * PROBE_TIMEOUT_S`` plus the waits, 6.3 s.
PROBE_ATTEMPTS = 3
PROBE_TIMEOUT_S = 2.0
_PROBE_WAIT = wait_exponential(multiplier=0.1, max=0.5)
_PROBE_UNDELIVERED: tuple[type[Exception], ...] = TRITON_CALL_ERRORS

PROBE_READY = 'ready'
#: Ready on a later attempt, after the transport refused an earlier one.
PROBE_RECOVERED = 'recovered'
#: The engine answered, and the answer was "not ready".
PROBE_NOT_READY = 'not_ready'
#: No attempt got an answer: the engine could not be asked.
PROBE_UNDELIVERED = 'undelivered'


@dataclass(frozen=True)
class Readiness:
    """What a readiness probe learned: ``outcome`` is one of the ``PROBE_*`` values, ``attempts``
    how many calls it took, ``code`` and ``error`` the last transport failure (empty when every
    attempt was answered)."""

    outcome: str
    attempts: int
    code: str = ''
    error: str = ''

    @property
    def ready(self) -> bool:
        return self.outcome in (PROBE_READY, PROBE_RECOVERED)


async def probe_ready(pool: AsyncTritonPool) -> Readiness:
    """Whether the engine says it is ready, asked until it answers or the budget is spent."""
    attempts = 0
    answered = False
    failure: BaseException | None = None
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_PROBE_UNDELIVERED),
            wait=_PROBE_WAIT,
            stop=stop_after_attempt(PROBE_ATTEMPTS),
            sleep=RETRY_SLEEP,
            reraise=True,
        ):
            with attempt:
                attempts = attempt.retry_state.attempt_number
                answered = await pool.probe_model_ready(
                    wire.MODEL_NAME, channel=attempts - 1, timeout=PROBE_TIMEOUT_S
                )
    except _CALL_ERRORS as exc:  # RuntimeError among them: the pool is not open
        failure = exc
    if failure is not None:
        return Readiness(PROBE_UNDELIVERED, attempts, _status(failure) or '', str(failure))
    if not answered:
        return Readiness(PROBE_NOT_READY, attempts)
    return Readiness(PROBE_READY if attempts == 1 else PROBE_RECOVERED, attempts)


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


def _status(exc: BaseException | None) -> str | None:
    from tritonclient.utils import InferenceServerException

    return exc.status() if isinstance(exc, InferenceServerException) else None


def _lost_in_transit(exc: BaseException | None) -> bool:
    """An ``INTERNAL`` the transport or the gRPC library made (``Stream removed``, ``Received
    RST_STREAM with error code 1``, a flow-control violation): the call or its answer was lost
    with the stream, whatever the engine did with the photo. The engine's own ``INTERNAL``
    (the photo's failure) starts with :data:`wire.ENGINE_ERROR_PREFIX` and is not one."""
    from tritonclient.utils import InferenceServerException

    return (
        isinstance(exc, InferenceServerException)
        and exc.status() == _INTERNAL
        and not (exc.message() or '').startswith(wire.ENGINE_ERROR_PREFIX)
    )


def _retry_again(state: RetryCallState) -> bool:
    """Retry an overload (the deadline ends it), and ``RETRIES`` times an UNAVAILABLE engine or a
    call lost in transit.

    gRPC status-code guide (https://grpc.io/docs/guides/status-codes/): UNAVAILABLE "is most
    likely a transient condition, which can be corrected by retrying with a backoff" and "it is
    not always safe to retry non-idempotent operations"; this call is a pure function of the
    photo and the plan (nothing is stored by the engine), so any repeat is safe. INTERNAL is "for
    serious errors" and the engine's own is deterministic for the photo, so only the transport's
    is repeated (:func:`_lost_in_transit`)."""
    outcome = state.outcome
    exc = outcome.exception() if outcome is not None else None
    status = _status(exc)
    if status == _OVERLOAD:
        return True
    return (status == _UNAVAILABLE or _lost_in_transit(exc)) and state.attempt_number <= RETRIES


async def process(
    pool: AsyncTritonPool, data: bytes, plan: wire.EnginePlan, boxes: np.ndarray | None = None
) -> EngineAnswer:
    """The engine's answer for the encoded image ``data`` under ``plan``.

    Raises:
        EngineCallError: no answer after the retries (``overloaded`` when the last one was the
            overload status).
    """
    inputs = [_bytes_input(wire.IMAGE, data), _bytes_input(wire.PLAN, plan.to_json().encode())]
    if boxes is not None:
        inputs.append(_boxes_input(boxes))
    try:
        async for attempt in AsyncRetrying(
            retry=_retry_again,
            wait=_BACKOFF,
            stop=stop_after_delay(DEADLINE_SECONDS),
            sleep=RETRY_SLEEP,
            reraise=True,
        ):
            with attempt:
                result = await pool.infer(
                    wire.MODEL_NAME, inputs, [], timeout=TIMEOUT_SECONDS, retries=0
                )
                return _answer(result, plan)
    except _CALL_ERRORS as exc:
        raise EngineCallError(str(exc), overloaded=_status(exc) == _OVERLOAD) from exc
    raise AssertionError('unreachable')  # pragma: no cover - AsyncRetrying always yields


__all__ = [
    'DEADLINE_SECONDS',
    'DEFAULT_URL',
    'ENGINE_CHANNEL_OPTIONS',
    'ENGINE_KEEPALIVE_TIME_MS',
    'FRAME_ENGINE_URL_ENV',
    'PROBE_ATTEMPTS',
    'PROBE_NOT_READY',
    'PROBE_READY',
    'PROBE_RECOVERED',
    'PROBE_TIMEOUT_S',
    'PROBE_UNDELIVERED',
    'RETRIES',
    'EngineAnswer',
    'EngineCallError',
    'Readiness',
    'frame_engine_url',
    'open_pool',
    'probe_ready',
    'process',
]
