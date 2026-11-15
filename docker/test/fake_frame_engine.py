"""A frame engine stand-in for the live harness: the real gRPC front, a CPU engine behind it.

The API sends every photo of an ingest to the frame engine (``src/clients/frame_engine.py``,
protocol version 1: one KServe v2 ``ModelInfer`` to the model ``frame_engine``). This process
answers it with the engine's own front (``frame_engine/front.py``: request parsing, plan
validation, the status codes, the response packing), so the wire and the 503 mapping run for real.
Behind the front a CPU engine stands in for the GPU one:

* the photo is decoded with Pillow (``src/utils/image_decode.py``), so an undecodable file answers
  ``corrupt`` as the engine does;
* the detection, the frame vector and the crop vectors are the ones ``fake_triton.py`` gives the
  host path (one box over the middle of the letterbox canvas; a vector from the picture's coarse
  colour layout), through the API's own preprocessing (``src/utils/pe_preprocess.py``), so a
  picture is its own nearest neighbour and two pictures differ, as they do on the host path;
* the answer's structure (branch states, letterbox, blur rows, the rows the plan embeds) is built
  by ``tests/curation/engine_answer.py``, the builder the ingest unit tests' fake uses too.

It decodes on the CPU and says so: the ``decode_route`` is ``cpu_small``, the route a small photo
takes in the real engine. It proves nothing about nvJPEG, the CUDA kernels, the model server's
real vectors or the engine's memory budget; those belong to the GPU tier.

Refusals, for the tests: the file ``FAKE_ENGINE_MODE_FILE`` (default
``/verify-data/fake_engine_mode``) holds the mode, read at every request and probe; no file or
``ok`` is normal operation.

* ``not_ready``: ``ServerReady`` and ``ModelReady`` answer false and a request is UNAVAILABLE (an
  engine that is down or restarting);
* ``unavailable``: the probes answer ready but every request is UNAVAILABLE (an engine that died
  after the API's readiness check);
* ``overloaded``: every request is RESOURCE_EXHAUSTED (the client retries it until the photo's
  deadline).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import fake_triton
import numpy as np
from engine_answer import engine_answer, failed_answer, letterbox_of

from frame_engine.front import FrontService, make_server
from frame_engine.stage_metrics import EngineMetrics
from frame_engine.transport import OverloadError, TritonUnavailableError
from src.domain.image_geometry import crop_rect, undo_letterbox
from src.utils.image_decode import ImageDecodeError, decode_image
from src.utils.pe_preprocess import letterbox_crop_rgb, to_chw, whole_frame_rgb_from_image


if TYPE_CHECKING:
    from PIL import Image

    from frame_engine.messages import EngineRequest, EngineResult

MODE_FILE = Path(os.environ.get('FAKE_ENGINE_MODE_FILE', '/verify-data/fake_engine_mode'))
PORT = 8001
ROUTE = 'cpu_small'
JPEG_QUALITY = 90
logger = logging.getLogger('fake_frame_engine')


def mode() -> str:
    try:
        return MODE_FILE.read_text().strip() or 'ok'
    except FileNotFoundError:
        return 'ok'


def _vectors(rgb_chw: np.ndarray) -> np.ndarray:
    vectors: np.ndarray = fake_triton._image_embeddings(rgb_chw, fake_triton.EMBED_DIM)
    return vectors


def _crop_jpeg(image: Image.Image, rect: tuple[int, int, int, int]) -> bytes:
    out = io.BytesIO()
    image.crop(rect).save(out, format='JPEG', quality=JPEG_QUALITY)
    return out.getvalue()


def process(request: EngineRequest) -> EngineResult:
    """One photo, the way the engine answers it: decode, detect (or take the stored boxes), embed."""
    plan = request.plan
    try:
        image = decode_image(bytes(request.data)).pil.convert('RGB')
    except ImageDecodeError:
        refused: EngineResult = failed_answer(plan, 'corrupt')
        return refused
    width, height = image.size
    canvas = plan.detector.input_size if plan.detector is not None else 640
    size = plan.embed.size if plan.embed is not None else 336
    crop_size = plan.crop_embed.size if plan.crop_embed is not None else size
    detections = None
    if request.boxes is not None:
        rects = [
            crop_rect((x1 * width, y1 * height, x2 * width, y2 * height), width, height)
            for x1, y1, x2, y2 in request.boxes
        ]
    else:
        # the box fake_triton's detector gives the host path, in canvas pixels
        detections = fake_triton._detect(np.empty((1, 3, canvas, canvas), np.uint8))[0, :1]
        scale, pad_x, pad_y = letterbox_of(width, height, canvas)
        rects = [
            crop_rect(undo_letterbox((x1, y1, x2, y2), scale, (pad_x, pad_y)), width, height)
            for x1, y1, x2, y2 in detections[:, :4].tolist()
        ]
    whole = _vectors(to_chw(whole_frame_rgb_from_image(image, size))[None])[0]
    crops = [np.asarray(image.crop(rect)) for rect in rects]
    crop_vectors = (
        _vectors(np.stack([to_chw(letterbox_crop_rgb(c, crop_size)) for c in crops]))
        if crops
        else np.zeros((0, whole.shape[0]), np.float32)
    )
    answer: EngineResult = engine_answer(
        plan,
        width,
        height,
        route=ROUTE,
        canvas=canvas,
        detections=detections,
        whole=whole,
        crop_vectors=crop_vectors,
        crop_jpegs=[_crop_jpeg(image, rect) for rect in rects],
        box_count=None if request.boxes is None else len(rects),
    )
    return answer


class FakeEngine:
    """The ``Engine`` the front submits to: a thread pool, and the refusals of the mode file."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(4, thread_name_prefix='fake-engine')

    def submit(self, request: EngineRequest) -> Future[EngineResult]:
        refused: Future[EngineResult] = Future()
        match mode():
            case 'overloaded':
                refused.set_exception(OverloadError('admission', 'fake engine: overloaded'))
            case 'unavailable' | 'not_ready':
                refused.set_exception(TritonUnavailableError('fake engine: unavailable'))
            case _:
                return self._pool.submit(process, request)
        return refused


async def serve() -> None:
    metrics = EngineMetrics()
    service = FrontService(
        FakeEngine(),
        lambda: mode() != 'not_ready',
        metrics,
        max_in_flight=64,
        request_deadline=60.0,
    )
    server, _port = make_server(service, address=f'0.0.0.0:{PORT}', max_message_bytes=2**28)
    await server.start()
    logger.info('fake frame engine serving on %d, mode file %s', PORT, MODE_FILE)
    await server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(serve())
