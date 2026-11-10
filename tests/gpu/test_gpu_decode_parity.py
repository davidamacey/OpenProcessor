"""GPU decode and prep parity: ``gpu_image_prep`` on a running Triton against the CPU ingest path.

This is the guard the nvImageCodec + CV-CUDA rewrite (#40) must keep green. Every test drives the
real model over gRPC from the host venv (``tritonclient``) and compares what it answers with what
the CPU path computes on the same bytes: the oriented size, the detector letterbox, the frame's
Laplacian variance (a decode check), the whole-frame PE embedding, and the detections.

Run (GPU tier; needs a Triton with ``gpu_image_prep``, ``pe_image_encoder`` and the detector,
see docker/test/README.md "GPU tier"):

    OP_TEST_TRITON_GRPC=127.0.0.1:4601 .venv/bin/python -m pytest tests/gpu -q --no-cov -m gpu

An unreachable Triton FAILS these tests; they are never skipped. The tolerances below are the
contract; tighten them as the first GPU runs show the real spread.
"""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from src.domain.image_geometry import letterbox_geometry
from src.services.detection.crop_quality import variance_from_sums
from src.services.detection.geometry import letterbox_to_square
from src.utils.pe_preprocess import whole_frame_rgb_from_image


pytestmark = pytest.mark.gpu

NET_SIZE = 640
DETECTOR = 'yolov11_small_trt_end2end'
PE_MODEL = 'pe_image_encoder'
SAMPLE = Path(__file__).resolve().parents[1] / 'fixtures' / 'sample_image.jpg'

# Explicit tolerances against the CPU path (nvJPEG vs libjpeg decode, CUDA vs Pillow resample).
LETTERBOX_ATOL = 1e-9  # pure arithmetic, must be exact
LAP_VAR_REL = 0.05  # frame Laplacian variance, relative
PE_COSINE_MIN = 0.99  # whole-frame embedding vs the CPU-preprocessed frame through the same PE
DET_IOU_MIN = 0.90  # a CPU detection above DET_SCORE_MIN has a GPU one this close
DET_SCORE_ATOL = 0.05
DET_SCORE_MIN = 0.5


def _with_orientation(data: bytes, orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert('RGB').save(buf, 'JPEG', quality=92, exif=exif.tobytes())
    return buf.getvalue()


@pytest.fixture(scope='module')
def url() -> str:
    return os.environ.get('OP_TEST_TRITON_GRPC', '127.0.0.1:4601')


@pytest.fixture(scope='module')
def client(url: str) -> Any:
    from tritonclient.grpc import InferenceServerClient

    c = InferenceServerClient(url=url)
    try:
        ready = c.is_server_ready()
    except Exception as exc:
        pytest.fail(f'GPU tier needs a Triton gRPC endpoint at {url} (OP_TEST_TRITON_GRPC): {exc}')
    assert ready, f'Triton at {url} is not ready'
    for model in ('gpu_image_prep', PE_MODEL, DETECTOR):
        if not c.is_model_ready(model):
            c.load_model(model)  # explicit model control: gpu_image_prep loads on first use
        assert c.is_model_ready(model), f'{model} is not ready on {url}'
    return c


@pytest.fixture(scope='module', params=['upright', 'rot6', 'rot3'])
def photo(request: pytest.FixtureRequest) -> bytes:
    data = SAMPLE.read_bytes()
    return data if request.param == 'upright' else _with_orientation(data, int(request.param[3:]))


def _gpu(url: str, data: bytes) -> dict[str, np.ndarray]:
    from tritonclient.grpc import InferInput
    from tritonclient.grpc.aio import InferenceServerClient

    async def run() -> dict[str, np.ndarray]:
        inp = InferInput('jpeg', [len(data)], 'UINT8')
        inp.set_data_from_numpy(np.frombuffer(data, np.uint8))

        async def requests() -> Any:
            yield {'model_name': 'gpu_image_prep', 'inputs': [inp]}

        aio = InferenceServerClient(url=url)
        try:
            async for result, error in aio.stream_infer(requests(), stream_timeout=120.0):
                if error is not None:
                    raise error
                names = [
                    'status',
                    'size',
                    'letterbox',
                    'det_boxes',
                    'det_scores',
                    'det_classes',
                    'whole_embedding',
                    'blur_sums',
                ]
                return {n: result.as_numpy(n) for n in names}
        finally:
            await aio.close()
        raise RuntimeError('stream closed without a response')

    return asyncio.run(run())


def _infer(client: Any, model: str, chw: np.ndarray, output: str) -> np.ndarray:
    from tritonclient.grpc import InferInput, InferRequestedOutput

    inp = InferInput('images', list(chw.shape), 'UINT8')
    inp.set_data_from_numpy(chw)
    return client.infer(model, [inp], outputs=[InferRequestedOutput(output)]).as_numpy(output)


@pytest.fixture
def pair(client: Any, url: str, photo: bytes) -> tuple[dict, Image.Image]:
    gpu = _gpu(url, photo)
    cpu = ImageOps.exif_transpose(Image.open(io.BytesIO(photo))).convert('RGB')
    return gpu, cpu


def test_the_gpu_path_answers_ok_with_the_oriented_size(pair: tuple[dict, Image.Image]) -> None:
    gpu, cpu = pair
    assert gpu['status'][0].decode() == 'ok'
    assert tuple(int(v) for v in gpu['size']) == cpu.size


def test_the_letterbox_is_the_cpu_letterbox(pair: tuple[dict, Image.Image]) -> None:
    gpu, cpu = pair
    _chw, scale, (pad_w, pad_h) = letterbox_to_square(cpu, NET_SIZE)
    lb = letterbox_geometry(*cpu.size, NET_SIZE)
    np.testing.assert_allclose(gpu['letterbox'], [scale, pad_w, pad_h], atol=LETTERBOX_ATOL)
    np.testing.assert_allclose(
        gpu['letterbox'], [lb.scale, lb.pad_w, lb.pad_h], atol=LETTERBOX_ATOL
    )


def test_the_decoded_frame_has_the_cpu_laplacian_variance(pair: tuple[dict, Image.Image]) -> None:
    """The decode check: nvJPEG and libjpeg frames must score the same sharpness."""
    gpu, cpu = pair
    pixels, lap_sum, lap_sq = (int(v) for v in gpu['blur_sums'][0])
    w, h = cpu.size
    assert pixels == w * h
    gray = cv2.cvtColor(np.asarray(cpu), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_16S).astype(np.int64)
    want = variance_from_sums(lap.size, int(lap.sum()), int((lap * lap).sum()))
    got = variance_from_sums(pixels, lap_sum, lap_sq)
    assert want is not None
    assert got == pytest.approx(want, rel=LAP_VAR_REL)


def test_the_whole_frame_embedding_matches_the_cpu_preprocess(
    client: Any, pair: tuple[dict, Image.Image]
) -> None:
    gpu, cpu = pair
    chw = np.ascontiguousarray(whole_frame_rgb_from_image(cpu).transpose(2, 0, 1))[None]
    want = _infer(client, PE_MODEL, chw, 'image_embeddings')[0].astype(np.float64)
    got = gpu['whole_embedding'].astype(np.float64)
    cosine = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want)))
    assert cosine >= PE_COSINE_MIN


def test_the_detections_match_the_cpu_letterboxed_detector(
    client: Any, pair: tuple[dict, Image.Image]
) -> None:
    gpu, cpu = pair
    chw, _scale, _pad = letterbox_to_square(cpu, NET_SIZE)
    rows = _infer(client, DETECTOR, chw, 'output0')[0]
    rows = rows[rows[:, 4] >= DET_SCORE_MIN]
    boxes, scores, classes = gpu['det_boxes'], gpu['det_scores'], gpu['det_classes']

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return float(inter / union) if union > 0 else 0.0

    for row in rows:
        same = [i for i in range(len(boxes)) if int(classes[i]) == int(row[5])]
        best = max(same, key=lambda i: iou(boxes[i], row[:4]), default=None)
        assert best is not None, f'no GPU detection of class {int(row[5])} for {row}'
        assert iou(boxes[best], row[:4]) >= DET_IOU_MIN
        assert float(scores[best]) == pytest.approx(float(row[4]), abs=DET_SCORE_ATOL)
