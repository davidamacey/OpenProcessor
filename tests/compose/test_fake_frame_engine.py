"""The harness frame engine speaks the engine's wire protocol to the API's own client.

The fake (``docker/test/fake_frame_engine.py``) is started in-process on the engine's real gRPC front
(``frame_engine/front.py``) and called through ``src/clients/frame_engine.py``, the client the API
uses, so a pass proves the tensor names, datatypes, status codes and the client's retries against
the code, not against a copy. What it proves: the protocol, the refusals the 503 mapping starts
from, the answer's shape. What it cannot: nvJPEG, kernels, real vectors (the GPU tier).

Protocol: https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from PIL import Image, ImageDraw

from src.clients import frame_engine
from src.clients.triton_pool_core import AsyncTritonPool
from src.domain import frame_engine_wire as wire


ROOT = Path(__file__).resolve().parents[2]
PLAN = wire.EnginePlan(
    branches=('detect', 'embed_frame', 'embed_crops'),
    detector=wire.DetectorSpec('yolov11_small_trt_end2end', 640, 114),
    embed=wire.EmbedSpec('pe_image_encoder', 336, 32),
    crops='inline',
)


def _picture(colour: tuple[int, int, int], size: tuple[int, int] = (160, 120)) -> bytes:
    img = Image.new('RGB', size, (128, 128, 128))
    w, h = size
    ImageDraw.Draw(img).rectangle((w // 5, h // 5, 4 * w // 5, 4 * h // 5), fill=colour)
    out = io.BytesIO()
    img.save(out, format='JPEG', quality=90)
    return out.getvalue()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.syspath_prepend(str(ROOT / 'docker' / 'test'))
    monkeypatch.syspath_prepend(str(ROOT / 'tests' / 'curation'))
    spec = importlib.util.spec_from_file_location(
        'fake_frame_engine_harness', ROOT / 'docker' / 'test' / 'fake_frame_engine.py'
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'MODE_FILE', tmp_path / 'mode')
    return module


async def _serve(fake: Any) -> tuple[Any, AsyncTritonPool]:
    from frame_engine.front import FrontService, make_server
    from frame_engine.stage_metrics import EngineMetrics

    service = FrontService(
        fake.FakeEngine(),
        lambda: fake.mode() != 'not_ready',
        EngineMetrics(),
        max_in_flight=8,
        request_deadline=30.0,
    )
    server, port = make_server(service, address='127.0.0.1:0', max_message_bytes=2**26)
    await server.start()
    pool = AsyncTritonPool(url=f'127.0.0.1:{port}', pool_size=1, max_concurrent=4)
    await pool.initialize()
    return server, pool


@pytest_asyncio.fixture
async def engine(fake: Any, monkeypatch: pytest.MonkeyPatch):
    server, pool = await _serve(fake)
    # the refusals are the thing under test: do not wait out the real backoff and deadline
    monkeypatch.setattr(frame_engine, 'DEADLINE_SECONDS', 0.3)

    async def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(frame_engine, 'RETRY_SLEEP', no_sleep)
    yield pool, fake
    await pool.close()
    await server.stop(0)


@pytest.mark.asyncio
async def test_a_picture_is_detected_embedded_and_cropped_over_the_wire(engine: Any) -> None:
    pool, _ = engine

    answer = await frame_engine.process(pool, _picture((200, 30, 30)), PLAN)

    assert answer.ok
    assert answer.decode_route == 'cpu_small'
    assert [int(v) for v in answer.outputs[wire.SIZE]] == [160, 120]
    assert answer.outputs[wire.DET_BOXES].shape == (1, 4)
    assert answer.outputs[wire.WHOLE_EMBEDDING].shape == (1024,)
    assert answer.outputs[wire.CROP_EMBEDDINGS].shape == (1, 1024)
    assert answer.outputs[wire.CROP_EMBEDDED].tolist() == [True]
    (crop,) = answer.outputs[wire.CROP_JPEGS]
    assert Image.open(io.BytesIO(crop)).format == 'JPEG'


@pytest.mark.asyncio
async def test_a_picture_is_its_own_nearest_neighbour_and_the_answer_is_deterministic(
    engine: Any,
) -> None:
    pool, _ = engine
    red, blue = _picture((200, 30, 30)), _picture((30, 30, 200))

    a1 = await frame_engine.process(pool, red, PLAN)
    a2 = await frame_engine.process(pool, red, PLAN)
    b = await frame_engine.process(pool, blue, PLAN)

    v1, v2, vb = (x.outputs[wire.WHOLE_EMBEDDING] for x in (a1, a2, b))
    assert np.array_equal(v1, v2)
    assert float(v1 @ v2) > float(v1 @ vb)
    assert not np.array_equal(v1, vb)


@pytest.mark.asyncio
async def test_bytes_that_are_no_image_answer_corrupt_and_run_no_branch(engine: Any) -> None:
    pool, _ = engine

    answer = await frame_engine.process(pool, b'not an image', PLAN)

    assert answer.status == 'corrupt'
    assert not answer.ok
    assert set(answer.branches.values()) == {('error', wire.IMAGE_FAILED)}


@pytest.mark.asyncio
async def test_stored_boxes_are_embedded_without_the_detector(engine: Any) -> None:
    pool, _ = engine
    plan = wire.EnginePlan(branches=('embed_frame', 'embed_crops'), embed=PLAN.embed)
    boxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.4, 0.3, 0.9, 0.8]], np.float64)

    answer = await frame_engine.process(pool, _picture((10, 200, 10)), plan, boxes)

    assert answer.ok
    assert answer.outputs[wire.CROP_EMBEDDINGS].shape == (2, 1024)
    assert wire.DET_BOXES not in answer.outputs


@pytest.mark.asyncio
async def test_not_ready_refuses_the_probe_and_the_call_and_recovers(engine: Any) -> None:
    pool, fake = engine
    assert await pool.is_model_ready(wire.MODEL_NAME)
    fake.MODE_FILE.write_text('not_ready')

    assert not await pool.is_model_ready(wire.MODEL_NAME)
    with pytest.raises(frame_engine.EngineCallError) as refused:
        await frame_engine.process(pool, _picture((1, 2, 3)), PLAN)
    assert not refused.value.overloaded

    fake.MODE_FILE.unlink()
    assert await pool.is_model_ready(wire.MODEL_NAME)
    assert (await frame_engine.process(pool, _picture((1, 2, 3)), PLAN)).ok


@pytest.mark.asyncio
async def test_unavailable_answers_the_probe_ready_and_refuses_the_call(engine: Any) -> None:
    pool, fake = engine
    fake.MODE_FILE.write_text('unavailable')

    assert await pool.is_model_ready(wire.MODEL_NAME)
    with pytest.raises(frame_engine.EngineCallError) as refused:
        await frame_engine.process(pool, _picture((1, 2, 3)), PLAN)
    assert not refused.value.overloaded


@pytest.mark.asyncio
async def test_overloaded_is_the_status_the_client_retries(engine: Any) -> None:
    pool, fake = engine
    fake.MODE_FILE.write_text('overloaded')

    with pytest.raises(frame_engine.EngineCallError) as refused:
        await frame_engine.process(pool, _picture((1, 2, 3)), PLAN)

    assert refused.value.overloaded


def test_every_process_of_the_api_image_reaches_the_fake_engine_and_the_api_waits_for_it() -> None:
    from compose._helpers import render

    services = render({}, 'docker/test/compose.yml')
    host, _, port = services['api']['environment']['OP_FRAME_ENGINE_URL'].partition(':')
    assert host in services
    assert port == '8001'
    for name in ('api', 'jobs-worker', 'interactive-worker', 'detection-worker'):
        assert services[name]['environment']['OP_FRAME_ENGINE_URL'] == f'{host}:{port}', name
    assert services['api']['depends_on'][host]['condition'] == 'service_healthy'
