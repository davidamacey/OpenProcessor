"""The engine's KServe v2 gRPC front (``grpc.aio``), with tritonclient's generated service stubs.

It answers the calls the API's Triton client makes: ``ModelInfer`` (unary, one image), and
``ServerLive``, ``ServerReady``, ``ModelReady`` and ``ModelMetadata`` for the model
``frame_engine``. Every other KServe call answers ``UNIMPLEMENTED`` (the generated base class).

Status codes (gRPC status-code guide: RESOURCE_EXHAUSTED for "some resource has been exhausted",
retryable; UNAVAILABLE for a transient condition; INVALID_ARGUMENT for a request that can never
succeed): an overload (budget or slab wait timed out, or more requests than
``OP_FRAME_ENGINE_MAX_IN_FLIGHT`` admitted) is RESOURCE_EXHAUSTED; a model server that does not
answer, an engine that is not ready or is draining is UNAVAILABLE; a malformed request or plan is
INVALID_ARGUMENT; an unknown model is NOT_FOUND; a request still running at
``OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS`` is DEADLINE_EXCEEDED (logged and counted with the
stage it was in; the photo goes on to release what it holds, every wait in it being bounded);
anything else is INTERNAL. The API's retry helper (``src/utils/retry.py``) retries the first two.

The handler stays on the event loop only to unpack and pack tensors; the photo runs on the
engine's worker threads (``engine.submit``), awaited through ``asyncio.wrap_future``. The in-flight
count is touched only on the loop, so it needs no lock.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

import grpc
import numpy as np
from tritonclient.grpc import service_pb2, service_pb2_grpc

from frame_engine.messages import EngineRequest
from frame_engine.transport import OverloadError, TritonUnavailableError
from src.domain import frame_engine_wire as wire


if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from frame_engine.messages import EngineResult
    from frame_engine.stage_metrics import EngineMetrics


logger = logging.getLogger('frame_engine.front')
MODEL_VERSION = '1'
_NUMPY = {
    'FP64': np.float64,
    'FP32': np.float32,
    'INT32': np.int32,
    'INT64': np.int64,
    'BOOL': np.bool_,
}
_SHAPES: dict[str, list[int]] = {
    wire.IMAGE: [1],
    wire.PLAN: [1],
    wire.BOXES: [-1, 4],
    wire.STATUS: [1],
    wire.DECODE_ROUTE: [1],
    wire.BRANCH_STATUS: [1],
    wire.SIZE: [2],
    wire.DET_BOXES: [-1, 4],
    wire.DET_SCORES: [-1],
    wire.DET_CLASSES: [-1],
    wire.LETTERBOX: [3],
    wire.WHOLE_EMBEDDING: [-1],
    wire.CROP_EMBEDDINGS: [-1, -1],
    wire.CROP_EMBEDDED: [-1],
    wire.BLUR_SUMS: [-1, 3],
    wire.CROP_JPEGS: [-1],
    wire.OCR_QUADS: [-1, 8],
    wire.OCR_TEXTS: [-1],
    wire.OCR_SCORES: [-1],
    wire.OCR_CHAR_MIN: [-1],
    wire.OCR_DET_SCORES: [-1],
    wire.OCR_OVERFLOW: [1],
    wire.DERIVATIVE_JPEGS: [-1],
}


class Engine(Protocol):
    def submit(self, request: EngineRequest) -> Future[EngineResult]: ...


class BadRequestError(ValueError):
    pass


def unpack_bytes(raw: bytes) -> list[bytes]:
    """KServe BYTES tensor contents: each element is a 4-byte little-endian length and the bytes."""
    out: list[bytes] = []
    i, n = 0, len(raw)
    while i < n:
        if i + 4 > n:
            raise BadRequestError('BYTES tensor ends inside a length prefix')
        (size,) = struct.unpack_from('<I', raw, i)
        i += 4
        if i + size > n:
            raise BadRequestError('BYTES tensor element runs past the end')
        out.append(raw[i : i + size])
        i += size
    return out


def pack_bytes(items: list[bytes]) -> bytes:
    return b''.join(struct.pack('<I', len(x)) + x for x in items)


def _one_bytes(tensor: Any, raw: bytes) -> bytes:
    if tensor.datatype != 'BYTES' or list(tensor.shape) != [1]:
        raise BadRequestError(f'{tensor.name} must be BYTES [1]')
    items = unpack_bytes(raw)
    if len(items) != 1:
        raise BadRequestError(f'{tensor.name} must hold one element')
    return items[0]


def parse_request(request: Any) -> tuple[bytes, wire.EnginePlan, np.ndarray | None]:
    """The image, the validated plan and the stored boxes of one ``ModelInferRequest``.

    Raises:
        BadRequestError: a missing, unknown or malformed tensor.
        EnginePlanError: the plan is refused.
    """
    if len(request.raw_input_contents) != len(request.inputs):
        raise BadRequestError('every input must travel in raw_input_contents')
    raw = {t.name: (t, request.raw_input_contents[k]) for k, t in enumerate(request.inputs)}
    unknown = set(raw) - set(wire.INPUT_DATATYPES)
    if unknown:
        raise BadRequestError(f'unknown input {sorted(unknown)[0]!r}')
    for name in (wire.IMAGE, wire.PLAN):
        if name not in raw:
            raise BadRequestError(f'missing input {name!r}')
    image = _one_bytes(*raw[wire.IMAGE])
    plan_text = _one_bytes(*raw[wire.PLAN])
    boxes = None
    if wire.BOXES in raw:
        tensor, data = raw[wire.BOXES]
        shape = list(tensor.shape)
        if tensor.datatype != wire.INPUT_DATATYPES[wire.BOXES] or len(shape) != 2 or shape[1] != 4:
            raise BadRequestError('boxes must be FP64 [n, 4]')
        if len(data) != shape[0] * 4 * 8:
            raise BadRequestError('boxes byte size does not match its shape')
        boxes = np.frombuffer(data, np.float64).reshape(shape)
    plan = wire.parse_plan(plan_text, has_boxes=boxes is not None)
    return image, plan, boxes


def build_response(request: Any, result: EngineResult) -> Any:
    resp = service_pb2.ModelInferResponse(
        model_name=wire.MODEL_NAME, model_version=MODEL_VERSION, id=request.id
    )
    wanted = {o.name for o in request.outputs}

    def add(name: str, shape: list[int], payload: bytes) -> None:
        if wanted and name not in wanted:
            return
        t = resp.outputs.add()
        t.name = name
        t.datatype = wire.OUTPUT_DATATYPES[name]
        t.shape.extend(shape)
        resp.raw_output_contents.append(payload)

    add(wire.STATUS, [1], pack_bytes([result.status.encode()]))
    add(wire.DECODE_ROUTE, [1], pack_bytes([result.decode_route.encode()]))
    add(wire.BRANCH_STATUS, [1], pack_bytes([wire.encode_branch_status(result.branches).encode()]))
    for name, value in result.outputs.items():
        if wire.OUTPUT_DATATYPES[name] == 'BYTES':  # the crop JPEGs, the line texts
            add(name, [len(value)], pack_bytes(list(value)))
            continue
        arr: np.ndarray = np.ascontiguousarray(value, dtype=_NUMPY[wire.OUTPUT_DATATYPES[name]])
        add(name, list(arr.shape), arr.tobytes())
    return resp


def tensor_metadata() -> tuple[list[Any], list[Any]]:
    meta = service_pb2.ModelMetadataResponse.TensorMetadata
    inputs = [meta(name=n, datatype=d, shape=_SHAPES[n]) for n, d in wire.INPUT_DATATYPES.items()]
    outputs = [meta(name=n, datatype=d, shape=_SHAPES[n]) for n, d in wire.OUTPUT_DATATYPES.items()]
    return inputs, outputs


class FrontService(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(
        self,
        engine: Engine,
        ready: Callable[[], bool],
        metrics: EngineMetrics,
        *,
        max_in_flight: int,
        request_deadline: float,
    ) -> None:
        self.engine = engine
        self._ready = ready
        self.metrics = metrics
        self.max_in_flight = max_in_flight
        self.request_deadline = request_deadline
        self.in_flight = 0
        self.draining = False

    def ready(self) -> bool:
        return not self.draining and self._ready()

    async def ServerLive(self, _request: Any, _context: Any) -> Any:
        return service_pb2.ServerLiveResponse(live=True)

    async def ServerReady(self, _request: Any, _context: Any) -> Any:
        return service_pb2.ServerReadyResponse(ready=self.ready())

    async def ModelReady(self, request: Any, _context: Any) -> Any:
        return service_pb2.ModelReadyResponse(
            ready=request.name == wire.MODEL_NAME
            and request.version in ('', MODEL_VERSION)
            and self.ready()
        )

    async def ModelMetadata(self, request: Any, context: Any) -> Any:
        if request.name != wire.MODEL_NAME:
            await context.abort(grpc.StatusCode.NOT_FOUND, f'unknown model {request.name!r}')
        inputs, outputs = tensor_metadata()
        return service_pb2.ModelMetadataResponse(
            name=wire.MODEL_NAME,
            versions=[MODEL_VERSION],
            platform='frame_engine',
            inputs=inputs,
            outputs=outputs,
        )

    async def ModelInfer(self, request: Any, context: Any) -> Any:
        if request.model_name != wire.MODEL_NAME:
            await context.abort(grpc.StatusCode.NOT_FOUND, f'unknown model {request.model_name!r}')
        try:
            image, plan, boxes = parse_request(request)
        except (BadRequestError, wire.EnginePlanError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if not self.ready():
            await context.abort(grpc.StatusCode.UNAVAILABLE, 'frame engine not ready')
        if self.in_flight >= self.max_in_flight:
            self.metrics.overloads.labels('in_flight').inc()
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f'overload: {self.in_flight} requests in flight (cap {self.max_in_flight})',
            )
        self.in_flight += 1
        self.metrics.in_flight.set(self.in_flight)
        job = EngineRequest(image, plan, boxes)
        job.deadline = job.arrived + self.request_deadline
        result: EngineResult | None = None
        failure: tuple[grpc.StatusCode, str] | None = None
        try:
            waiter = asyncio.wrap_future(self.engine.submit(job))
            done, _ = await asyncio.wait({waiter}, timeout=self.request_deadline)
            if done:
                result = waiter.result()
            else:
                waiter.cancel()  # a photo still queued never starts; a running one ends bounded
        except OverloadError as exc:
            self.metrics.overloads.labels(exc.cause).inc()
            failure = grpc.StatusCode.RESOURCE_EXHAUSTED, f'overload: {exc}'
        except TritonUnavailableError as exc:
            failure = grpc.StatusCode.UNAVAILABLE, f'model server unavailable: {exc}'
        except Exception as exc:
            logger.exception('frame engine request failed')
            failure = grpc.StatusCode.INTERNAL, f'frame engine: {exc}'
        finally:
            self.in_flight -= 1
            self.metrics.in_flight.set(self.in_flight)
        if failure is not None:
            # abort sends the status before it raises: the request must already be uncounted
            await context.abort(*failure)
        if result is None:
            await self._past_deadline(context, job)
        return build_response(request, result)

    async def _past_deadline(self, context: Any, job: EngineRequest) -> NoReturn:
        self.metrics.deadlines.labels(job.stage).inc()
        logger.error(
            'request answered DEADLINE_EXCEEDED after %.1f s in stage %s',
            self.request_deadline, job.stage,
        )  # fmt: skip
        await context.abort(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            f'frame engine: no answer within {self.request_deadline:g} s (stage {job.stage})',
        )
        raise RuntimeError('context.abort returned')  # grpc.aio documents that it raises


def make_server(
    service: FrontService, *, address: str, max_message_bytes: int
) -> tuple[grpc.aio.Server, int]:
    """The unstarted server and the port it bound (``address`` may ask for port 0)."""
    server = grpc.aio.server(
        options=[
            ('grpc.max_send_message_length', max_message_bytes),
            ('grpc.max_receive_message_length', max_message_bytes),
        ]
    )
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(service, server)
    return server, server.add_insecure_port(address)
