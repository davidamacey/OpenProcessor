"""The two engines the contract test drives with one set of cases, and what the stub answers.

* :class:`RustEngine` starts the release binary ``op-frame-engine --pipeline stub`` on free ports.
* :class:`PythonEngine` serves the Python engine's real front (``frame_engine/front.py``) in this
  process, over a stand-in engine that answers as the stub does (:class:`StubEngine`: the
  Python engine's own tests use the same kind of stand-in). The front is the Python engine's;
  what stands in is the photo work, which needs a GPU.

Both read the same script out of the image: a photo whose bytes start with ``stub:`` is a list of
``key=value`` words (``fail=``, ``hold_ms=``, ``stage=``, ``status=``; ``native/frame_engine/src/stub.rs``),
so one case makes either engine fail, hold a request or refuse a photo.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import httpx
import numpy as np
from prometheus_client import CollectorRegistry
from tritonclient.grpc import InferInput, service_pb2, service_pb2_grpc

from frame_engine.cuda_fault import ContextLostError
from frame_engine.front import FrontService, make_server, pack_bytes
from frame_engine.messages import EngineRequest, EngineResult
from frame_engine.stage_metrics import EngineMetrics
from frame_engine.transport import OverloadError, TritonUnavailableError
from src.domain import frame_engine_wire as wire
from src.domain.image_geometry import letterbox_geometry, undo_letterbox


ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'native' / 'target'))
BINARY = TARGET / 'release' / 'op-frame-engine'
MIB = 1024 * 1024
START_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 15.0
DETECTIONS = 2
STAGES = tuple(
    json.loads((ROOT / 'contracts' / 'json' / 'frame_engine_v1.json').read_text(encoding='utf-8'))[
        'stages'
    ]
)


def build_binary() -> Path:
    """The release binary, built when ``cargo`` is on the path (an up-to-date build is a no-op);
    CI builds it in the step before. A missing binary fails the tests, it never skips them."""
    cargo = shutil.which('cargo') or str(Path.home() / '.cargo' / 'bin' / 'cargo')
    if Path(cargo).exists():
        done = subprocess.run(  # fixed argument list, no shell
            [cargo, 'build', '--release', '--locked', '--manifest-path',
             str(ROOT / 'native' / 'Cargo.toml'), '-p', 'frame_engine'],
            capture_output=True, text=True, timeout=1800, check=False,
        )  # fmt: skip
        if done.returncode != 0:
            raise RuntimeError(f'cargo build failed:\n{done.stdout}\n{done.stderr}')
    if not BINARY.is_file():
        raise RuntimeError(f'{BINARY} is missing: build the release profile first')
    return BINARY


def script(**words: Any) -> bytes:
    """An image that steers the stub: ``script(fail='overload', hold_ms=300)``."""
    return b'stub:' + ','.join(f'{k}={v}' for k, v in words.items()).encode()


def free_ports(n: int) -> list[int]:
    sockets = [socket.socket() for _ in range(n)]
    try:
        for s in sockets:
            s.bind(('127.0.0.1', 0))
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


# ----------------------------------------------------------------------------- what the stub answers


STUB_SIZE = (96, 64)
ENCODER_DIM = 1024
FACE_DIM = 512
_FRAME_BOXES = [(0.1, 0.1, 0.5, 0.6), (0.5, 0.3, 0.9, 0.9)]
_FACE_BOXES = [(8.0, 8.0, 40.0, 40.0), (48.0, 16.0, 88.0, 56.0)]


def _vectors(rows: int, dim: int) -> np.ndarray:
    """The stub's vectors: row ``r``, column ``c`` is +1 when ``c + r`` is even, else -1."""
    return np.where(
        (np.arange(dim)[None, :] + np.arange(rows)[:, None]) % 2 == 0, 1.0, -1.0
    ).astype(np.float32)


def _landmarks(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Five landmarks around the centre of a face box, as ``face_tensor`` in ``stub.rs`` places them."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [cx - 4, cy - 4, cx + 4, cy - 4, cx, cy, cx - 3, cy + 6, cx + 3, cy + 6]


def _sums_row(pixels: int) -> list[int]:
    return [pixels, 8 * pixels, 100 * pixels]


def _detect_tensors(input_size: int) -> dict[str, Any]:
    """``detections`` of the stub: the boxes in letterboxed canvas pixels, the letterbox, and the
    blur rows of the frame and of each box's rectangle, as ``geometry.rs`` and ``detect.rs`` do."""
    w, h = STUB_SIZE
    lb = letterbox_geometry(w, h, input_size)
    scale, pad_w, pad_h = lb.scale, lb.pad_w, lb.pad_h
    boxes = np.array(
        [
            [
                x1 * w * scale + pad_w,
                y1 * h * scale + pad_h,
                x2 * w * scale + pad_w,
                y2 * h * scale + pad_h,
            ]
            for x1, y1, x2, y2 in _FRAME_BOXES
        ],
        np.float32,
    )
    rows = [_sums_row(w * h)]
    for box in boxes:
        bx1, by1, bx2, by2 = (float(v) for v in box)
        x1, y1, x2, y2 = undo_letterbox((bx1, by1, bx2, by2), scale, (pad_w, pad_h))
        rx1, rx2 = (min(max(round(v), 0), w) for v in (x1, x2))
        ry1, ry2 = (min(max(round(v), 0), h) for v in (y1, y2))
        rows.append(_sums_row((rx2 - rx1) * (ry2 - ry1)) if rx2 > rx1 and ry2 > ry1 else [0, 0, 0])
    return {
        wire.DET_BOXES: boxes,
        wire.LETTERBOX: np.array([scale, pad_w, pad_h], np.float64),
        wire.BLUR_SUMS: np.array(rows, np.int64),
    }


def stub_outputs(plan: wire.EnginePlan, rows: int) -> dict[str, Any]:
    """The tensors the stub gives for ``plan`` (``native/frame_engine/src/stub.rs``): ``rows`` is the
    number of stored boxes, or the two detections; arrays, and lists of bytes for BYTES."""
    n, d = rows, DETECTIONS
    detect = _detect_tensors(plan.detector.input_size) if plan.detector is not None else {}
    every: dict[str, Any] = {
        wire.DET_SCORES: np.full(d, 0.9, np.float32),
        wire.DET_CLASSES: np.array([1, 2], np.int32),
        wire.WHOLE_EMBEDDING: _vectors(1, ENCODER_DIM)[0],
        wire.CROP_EMBEDDINGS: _vectors(n, ENCODER_DIM),
        wire.CROP_EMBEDDED: np.ones(n, np.bool_),
        wire.CROP_JPEGS: [b'\xff\xd8' + str(r).encode() for r in range(n)],
        wire.FACE_BOXES: np.array(_FACE_BOXES, np.float32),
        wire.FACE_SCORES: np.full(d, 0.9, np.float32),
        wire.FACE_LANDMARKS: np.array([_landmarks(*b) for b in _FACE_BOXES], np.float32),
        wire.FACE_BLUR_SUMS: np.array(
            [_sums_row(int((x2 - x1) * (y2 - y1))) for x1, y1, x2, y2 in _FACE_BOXES], np.int64
        ),
        wire.FACE_OVERFLOW: np.zeros(1, np.int32),
        wire.FACE_EMBEDDINGS: _vectors(d, FACE_DIM),
        wire.FACE_EMBEDDED: np.ones(d, np.bool_),
        wire.OCR_QUADS: np.zeros((d, 8), np.float32),
        wire.OCR_TEXTS: [b'STOP', 'caf\u00e9'.encode()],
        wire.OCR_SCORES: np.full(d, 0.9, np.float32),
        wire.OCR_CHAR_MIN: np.full(d, 0.8, np.float32),
        wire.OCR_DET_SCORES: np.full(d, 0.7, np.float32),
        wire.OCR_OVERFLOW: np.zeros(1, np.int32),
        wire.DERIVATIVE_JPEGS: [b'\xff\xd8' + r.name.encode() for r in plan.derivatives],
    }
    every.update(detect)
    out: dict[str, Any] = {wire.SIZE: np.array(STUB_SIZE, np.int32)}
    out.update({name: every[name] for name in wire.expected_outputs(plan)})
    return out


def raw_of(name: str, value: Any) -> bytes:
    """The ``raw_output_contents`` element of a tensor."""
    if wire.OUTPUT_DATATYPES[name] == 'BYTES':
        return pack_bytes(list(value))
    return np.ascontiguousarray(value, dtype=_NUMPY[wire.OUTPUT_DATATYPES[name]]).tobytes()


def shape_of(name: str, value: Any) -> list[int]:
    if wire.OUTPUT_DATATYPES[name] == 'BYTES':
        return [len(value)]
    return list(np.asarray(value).shape)


_NUMPY = {
    'FP64': np.float64,
    'FP32': np.float32,
    'INT32': np.int32,
    'INT64': np.int64,
    'BOOL': np.bool_,
}


def _words(image: bytes | memoryview) -> dict[str, str]:
    data = bytes(image)
    if not data.startswith(b'stub:'):
        return {}
    out: dict[str, str] = {}
    for word in data[5:].decode().split(','):
        if not word:
            continue
        key, sep, value = word.partition('=')
        if not sep or key not in ('fail', 'hold_ms', 'stage', 'status'):
            raise RuntimeError(f'stub photo: unknown word {word!r}')
        if key == 'stage' and value not in STAGES:
            raise RuntimeError(f'stub photo: stage={value!r}')
        if key == 'hold_ms' and not value.isdigit():
            raise RuntimeError(f'stub photo: hold_ms={value!r}')
        out[key] = value
    return out


class StubEngine:
    """The Python front's engine stand-in: answers as ``native/frame_engine/src/stub.rs`` does."""

    def __init__(self) -> None:
        self.stopped = threading.Event()

    def submit(self, request: EngineRequest) -> Future[EngineResult]:
        fut: Future[EngineResult] = Future()
        threading.Thread(target=self._work, args=(request, fut), daemon=True).start()
        return fut

    def _work(self, request: EngineRequest, fut: Future[EngineResult]) -> None:
        if not fut.set_running_or_notify_cancel():
            return
        try:
            fut.set_result(self._answer(request))
        except Exception as exc:  # the front maps each class
            fut.set_exception(exc)

    def _answer(self, request: EngineRequest) -> EngineResult:
        words = _words(request.data)
        if 'hold_ms' in words:
            request.enter(words.get('stage', 'detect'))
            self.stopped.wait(int(words['hold_ms']) / 1000)
        match words.get('fail'):
            case None:
                pass
            case 'overload':
                raise OverloadError('budget', 'stub')
            case 'unavailable':
                raise TritonUnavailableError('stub: down')
            case 'context_lost':
                raise ContextLostError('stub: lost')
            case 'internal':
                raise RuntimeError('stub failure')
            case other:
                raise RuntimeError(f'stub photo: fail={other!r}')
        plan = request.plan
        if 'status' in words:
            return EngineResult(
                words['status'], 'none',
                dict.fromkeys(plan.reported(), ('error', wire.IMAGE_FAILED)), {},
            )  # fmt: skip
        rows = len(request.boxes) if request.boxes is not None else DETECTIONS
        return EngineResult(
            'ok', 'gpu', dict.fromkeys(plan.reported(), ('ok', '')), stub_outputs(plan, rows)
        )


# ----------------------------------------------------------------------------- the two engines


@dataclass
class Settings:
    max_in_flight: int = 8
    request_deadline: float = 1.0
    max_message_mib: int = 64
    drain_seconds: float = 5.0


class Engine:
    kind: str
    target: str

    def __init__(self) -> None:
        self.channel: grpc.aio.Channel | None = None

    def stub(self) -> service_pb2_grpc.GRPCInferenceServiceStub:
        """A raw KServe stub on its own channel, unlimited message sizes."""
        if self.channel is None:
            self.channel = grpc.aio.insecure_channel(
                self.target,
                options=[
                    ('grpc.max_send_message_length', -1),
                    ('grpc.max_receive_message_length', -1),
                ],
            )
        return service_pb2_grpc.GRPCInferenceServiceStub(self.channel)

    async def close_channel(self) -> None:
        if self.channel is not None:
            await self.channel.close()
            self.channel = None

    async def set_ready(self, ready: bool) -> None:
        raise NotImplementedError

    async def begin_drain(self) -> None:
        raise NotImplementedError

    async def deadlines(self) -> dict[str, float]:
        raise NotImplementedError

    async def in_flight(self) -> int:
        raise NotImplementedError

    async def overloads(self) -> dict[str, float]:
        raise NotImplementedError

    async def stop(self) -> None:
        await self.close_channel()

    async def wait_model_ready(self, want: bool, *, timeout: float = 10.0) -> None:
        """Until ``ModelReady`` answers ``want`` (the health tick has probed), bounded."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            reply = await self.stub().ModelReady(
                service_pb2.ModelReadyRequest(name=wire.MODEL_NAME)
            )
            if reply.ready == want:
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f'ModelReady did not turn {want} within {timeout} s')


def _sample_map(text: str, family: str, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith(f'{family}{{{label}="'):
            head, value = line.rsplit(' ', 1)
            out[head.split('"')[1]] = float(value)
    return out


class RustEngine(Engine):
    kind = 'rust'

    def __init__(
        self, binary: Path, tmp: Path, settings: Settings, extra_env: dict[str, str] | None = None
    ) -> None:
        super().__init__()
        port, metrics = free_ports(2)
        self.target = f'127.0.0.1:{port}'
        self.metrics_url = f'http://127.0.0.1:{metrics}/metrics'
        self.not_ready_file = tmp / f'not-ready-{port}'
        self.log = tmp / f'engine-{port}.log'
        env = {
            'PATH': os.environ.get('PATH', ''),
            'OP_FRAME_ENGINE_PORT': str(port),
            'OP_FRAME_ENGINE_METRICS_PORT': str(metrics),
            'OP_FRAME_ENGINE_MAX_IN_FLIGHT': str(settings.max_in_flight),
            'OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS': str(settings.request_deadline),
            'OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS': str(min(15.0, settings.request_deadline)),
            'OP_FRAME_ENGINE_MAX_MESSAGE_MIB': str(settings.max_message_mib),
            'OP_FRAME_ENGINE_DRAIN_SECONDS': str(settings.drain_seconds),
            'OP_FRAME_ENGINE_HEALTH_SECONDS': '0.1',
            **(extra_env or {}),
        }
        self._log = self.log.open('wb')
        self.proc = subprocess.Popen(  # the release binary of this checkout, a fixed argument list
            [str(binary), '--pipeline', 'stub', '--stub-not-ready-file', str(self.not_ready_file)],
            env=env, stdout=self._log, stderr=subprocess.STDOUT,
        )  # fmt: skip

    def output(self) -> str:
        return self.log.read_text(errors='replace')

    async def wait_started(self) -> None:
        end = time.monotonic() + START_TIMEOUT_S
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f'the engine exited with {self.proc.returncode}:\n{self.output()}'
                )
            try:
                reply = await self.stub().ServerLive(service_pb2.ServerLiveRequest(), timeout=1.0)
            except grpc.aio.AioRpcError:
                await asyncio.sleep(0.05)
                continue
            if reply.live:
                return
        raise AssertionError(
            f'the engine did not start within {START_TIMEOUT_S} s:\n{self.output()}'
        )

    async def set_ready(self, ready: bool) -> None:
        if ready:
            self.not_ready_file.unlink(missing_ok=True)
        else:
            self.not_ready_file.write_bytes(b'')
        await self.wait_model_ready(ready)

    async def begin_drain(self) -> None:
        self.proc.send_signal(signal.SIGTERM)

    async def _scrape(self) -> str:
        async with httpx.AsyncClient(timeout=5.0) as http:
            return (await http.get(self.metrics_url)).text

    async def deadlines(self) -> dict[str, float]:
        return _sample_map(await self._scrape(), 'op_frame_engine_request_deadlines_total', 'stage')

    async def overloads(self) -> dict[str, float]:
        return _sample_map(await self._scrape(), 'op_frame_engine_overloads_total', 'cause')

    async def in_flight(self) -> int:
        for line in (await self._scrape()).splitlines():
            if line.startswith('op_frame_engine_in_flight '):
                return int(float(line.split()[1]))
        raise AssertionError('the metrics hold no op_frame_engine_in_flight')

    async def stop(self) -> None:
        await super().stop()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                await asyncio.to_thread(self.proc.wait, STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.proc.kill()  # this Popen's own pid
                await asyncio.to_thread(self.proc.wait, STOP_TIMEOUT_S)
        self._log.close()


class PythonEngine(Engine):
    kind = 'python'

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.ready = True
        self.engine = StubEngine()
        self.service = FrontService(
            self.engine, lambda: self.ready, EngineMetrics(CollectorRegistry()),
            max_in_flight=settings.max_in_flight, request_deadline=settings.request_deadline,
        )  # fmt: skip
        self.server: Any = None

    async def start(self) -> None:
        self.server, port = make_server(
            self.service,
            address='127.0.0.1:0',
            max_message_bytes=self.settings.max_message_mib * MIB,
        )
        self.target = f'127.0.0.1:{port}'
        await self.server.start()

    async def set_ready(self, ready: bool) -> None:
        self.ready = ready

    async def begin_drain(self) -> None:
        self.service.draining = True

    def _counts(self, counter: Any, label: str) -> dict[str, float]:
        return {
            s.labels[label]: s.value
            for m in counter.collect() for s in m.samples if s.name.endswith('_total')
        }  # fmt: skip

    async def deadlines(self) -> dict[str, float]:
        return self._counts(self.service.metrics.deadlines, 'stage')

    async def overloads(self) -> dict[str, float]:
        return self._counts(self.service.metrics.overloads, 'cause')

    async def in_flight(self) -> int:
        return int(self.service.in_flight)

    async def stop(self) -> None:
        await super().stop()
        self.engine.stopped.set()
        if self.server is not None:
            await self.server.stop(None)


# ----------------------------------------------------------------------------- requests

DETECTOR = {'model': 'det', 'input_size': 640, 'letterbox_fill': 114}
EMBED = {'model': 'pe', 'size': 336}
CROP_EMBED = {'model': 'pe_crop', 'size': 224, 'batch': 32}
FACES = {'detector': 'scrfd', 'input_size': 640, 'recogniser': 'arcface', 'batch': 32,
         'score_min': 0.5, 'nms': 0.4, 'max_faces': 128}  # fmt: skip
OCR = {
    'detector': 'ocr_det',
    'recogniser': 'ocr_rec',
    'batch': 16,
    'max_lines': 128,
    'det_side': 736,
}
PHOTO = b'\xff\xd8photo'


def plan_text(branches: list[str], **over: Any) -> str:
    body: dict[str, Any] = {'version': 1, 'branches': branches, 'crops': 'none', **over}
    for branch, key, spec in (
        ('detect', 'detector', DETECTOR), ('embed_frame', 'embed', EMBED),
        ('embed_crops', 'crop_embed', CROP_EMBED), ('faces', 'faces', FACES), ('ocr', 'ocr', OCR),
    ):  # fmt: skip
        if branch in branches:
            body.setdefault(key, spec)
    return json.dumps(body)


INGEST = plan_text(['detect', 'embed_frame', 'embed_crops'], crops='inline')
FRAME = plan_text(['detect', 'embed_frame'])


def inputs(image: bytes, plan: str, boxes: np.ndarray | None = None) -> list[InferInput]:
    made = []
    for name, value in ((wire.IMAGE, image), (wire.PLAN, plan.encode())):
        tensor = InferInput(name, [1], 'BYTES')
        tensor.set_data_from_numpy(np.array([value], dtype=np.object_))
        made.append(tensor)
    if boxes is not None:
        tensor = InferInput(wire.BOXES, list(boxes.shape), 'FP64')
        tensor.set_data_from_numpy(boxes)
        made.append(tensor)
    return made
