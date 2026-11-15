"""Batch ingest on the frame engine (#217): every photo of a curation ingest goes to the engine
(one request per photo, protocol version 1), whatever its size or format; the API never decodes
it and has no fallback for it. No GPU here: a fake pool on the engine's address answers, and the
real client (``src/clients/frame_engine.py``) runs in between."""

from __future__ import annotations

import dataclasses
from typing import Any

import grpc
import pytest
from tritonclient.utils import InferenceServerException

from curation.engine_fakes import engine_service, register
from curation.ingest_fakes import _jpeg_bytes
from src.clients import frame_engine
from src.core.exceptions import EngineUnavailableError
from src.domain import frame_engine_wire as wire
from src.services.curation import ingest_gpu_decode
from src.services.curation.ingest_policy import DetectFilter, EmbeddingPolicy, IngestPolicy


@pytest.fixture(autouse=True)
def _fresh_ready_cache() -> Any:
    ingest_gpu_decode._READY.clear()
    yield
    ingest_gpu_decode._READY.clear()


def _route_sample(route: str) -> float:
    from prometheus_client import REGISTRY

    return (
        REGISTRY.get_sample_value('op_decode_route_total', {'route': route, 'path': 'ingest'})
        or 0.0
    )


def _routes(result: Any, os_fake: Any) -> dict[str, str | None]:
    """Route by image path from the results, checked against the stored images docs."""
    stored = {d['image_path']: d.get('decode_route') for d in os_fake.images.values()}
    by_path = {r.image_path: r.decode_route for r in result.results}
    assert {r.image_path: r.decode_route for r in result.results if r.status == 'success'} == stored
    return by_path


@pytest.mark.asyncio
async def test_every_photo_goes_to_the_engine_whatever_its_size(tmp_path) -> None:
    svc, os_fake, triton, engine, sizes = engine_service(tmp_path)
    big = register(sizes, _jpeg_bytes((640, 480), seed=1), (640, 480))
    small = register(sizes, _jpeg_bytes((32, 24), seed=2), (32, 24))
    result = await svc.ingest_batch([big, small], ['/x/big.jpg', '/x/small.jpg'])
    assert result.summary.successful == 2
    assert len(engine.calls) == 2  # no size gate in the API
    docs = {d['image_path']: d for d in os_fake.images.values()}
    assert docs['/x/big.jpg']['pe_embedding'] == pytest.approx([0.0, 0.0, 1.0])  # normalized
    assert docs['/x/small.jpg']['pe_embedding'] == pytest.approx([0.0, 0.0, 1.0])
    [(crop_id, item)] = [
        (k, d) for k, d in os_fake.items.items() if d['image_path'] == '/x/big.jpg'
    ]
    assert item['pe_embedding'] == pytest.approx([0.0, 1.0, 0.0])
    assert item['blur_full_var'] == 5.0
    assert item['blur_lap_var'] == 2.0
    assert item['class_name'] == 'widget'
    assert (tmp_path / 'crops' / 'default' / f'{crop_id}.jpg').read_bytes() == b'\xff\xd8crop'
    assert triton.calls == []  # the API called no detector: the engine ran it
    assert triton.loaded == ['primary_end2end', 'pe_image_encoder']  # loaded on first use, once


@pytest.mark.asyncio
async def test_the_plan_names_the_profile_detector_the_pe_encoder_and_inline_crops(
    tmp_path,
) -> None:
    svc, _os, _triton, engine, sizes = engine_service(tmp_path)
    big = register(sizes, _jpeg_bytes((640, 480), seed=21), (640, 480))
    await svc.ingest_batch([big], ['/x/p.jpg'])
    [call] = engine.calls
    plan = call['plan']
    assert plan.branches == ('detect', 'embed_frame', 'embed_crops')
    assert plan.detector == wire.DetectorSpec('primary_end2end', 320, 114)
    assert plan.embed == wire.EmbedSpec('pe_image_encoder', 336, 32)
    assert plan.crops == 'inline'
    assert plan.embed_classes is None  # no class list restricts the crops: every detection


@pytest.fixture
def _labels(tmp_path, monkeypatch) -> None:
    (tmp_path / 'labels.txt').write_text('gadget\nwidget\ngizmo\n')
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))


@pytest.mark.parametrize(
    ('policy', 'expected'),
    [
        (IngestPolicy(), None),
        (IngestPolicy(embedding=EmbeddingPolicy(mode='selected', classes=['Widget'])), {1}),
        (IngestPolicy(detect=DetectFilter(classes=['gadget', 'widget'])), {0, 1}),
        (
            IngestPolicy(
                detect=DetectFilter(classes=['gadget', 'widget']),
                embedding=EmbeddingPolicy(mode='selected', classes=['widget', 'gizmo']),
            ),
            {1},
        ),
        (IngestPolicy(detect=DetectFilter(exclude_classes=['gadget'])), {1, 2}),
        (IngestPolicy(embedding=EmbeddingPolicy(mode='all', classes=['widget'])), None),
    ],
)
@pytest.mark.usefixtures('_labels')
def test_the_plan_carries_the_class_ids_whose_names_pass_the_class_lists(
    tmp_path, policy: IngestPolicy, expected: set[int] | None
) -> None:
    svc, *_ = engine_service(tmp_path)
    svc.profile = dataclasses.replace(svc.profile, labels_path=str(tmp_path / 'labels.txt'))
    svc.policy = policy
    plan = ingest_gpu_decode.ingest_engine_plan(svc)
    assert plan.embed_classes == (None if expected is None else frozenset(expected))


@pytest.mark.asyncio
async def test_lazy_embedding_asks_for_no_crop_vectors(tmp_path) -> None:
    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path)
    svc.policy = IngestPolicy(embedding=EmbeddingPolicy(mode='lazy'))
    big = register(sizes, _jpeg_bytes((640, 480), seed=5), (640, 480))
    await svc.ingest_batch([big], ['/x/l.jpg'])
    [call] = engine.calls
    assert call['plan'].branches == ('detect', 'embed_frame')
    assert call['plan'].embed_classes is None
    [item] = list(os_fake.items.values())
    assert 'pe_embedding' not in item


@pytest.mark.parametrize(
    ('status', 'kind'),
    [
        ('corrupt', 'decode_failed'),
        ('decoder', 'decode_failed'),
        ('huge', 'decode_failed'),
        ('over_budget', 'decode_failed'),
        ('gpu_memory', 'engine_unavailable'),
    ],
)
@pytest.mark.asyncio
async def test_a_photo_the_engine_does_not_answer_fails_by_what_a_retry_can_change(
    tmp_path, status: str, kind: str
) -> None:
    svc, os_fake, triton, engine, sizes = engine_service(tmp_path, status=status)
    good = register(sizes, _jpeg_bytes((640, 480), seed=3), (640, 480))
    other = register(sizes, _jpeg_bytes((640, 480), seed=4), (640, 480))
    original = engine.infer

    async def infer(model: str, inputs: list, outputs: list, **k: Any) -> Any:
        engine.status = status if len(engine.calls) == 0 else 'ok'
        return await original(model, inputs, outputs, **k)

    engine.infer = infer
    result = await svc.ingest_batch([good, other], ['/x/a.jpg', '/x/b.jpg'])
    failed = [r for r in result.results if r.status == 'failed']
    assert [r.error_kind for r in failed] == [kind]
    assert result.summary.successful == 1  # the failure of one photo cancelled no other
    assert triton.calls == []  # no API-side detector run as a fallback
    assert len(os_fake.images) == 1


@pytest.mark.asyncio
async def test_an_engine_still_overloaded_past_the_deadline_is_engine_unavailable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(frame_engine, 'DEADLINE_SECONDS', 0.05)
    svc, os_fake, triton, engine, sizes = engine_service(tmp_path)
    engine.raises = InferenceServerException(
        msg='overload: budget', status=str(grpc.StatusCode.RESOURCE_EXHAUSTED)
    )
    big = register(sizes, _jpeg_bytes((640, 480), seed=22), (640, 480))
    with pytest.raises(EngineUnavailableError):  # one photo, ended engine_unavailable: 503
        await svc.ingest_batch([big], ['/x/o.jpg'])
    assert triton.calls == []
    assert not os_fake.images


@pytest.mark.parametrize(
    'error',
    [
        InferenceServerException(msg='boom', status=str(grpc.StatusCode.INTERNAL)),
        OSError('connection refused'),
        RuntimeError('stream closed without a response'),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_engine_call_is_engine_unavailable_and_no_fallback_runs(
    tmp_path, error: Exception
) -> None:
    svc, os_fake, triton, engine, sizes = engine_service(tmp_path)
    engine.raises = error
    big = register(sizes, _jpeg_bytes((640, 480), seed=16), (640, 480))
    with pytest.raises(EngineUnavailableError):
        await svc.ingest_batch([big], ['/x/f.jpg'])
    assert triton.calls == []
    assert not os_fake.images


@pytest.mark.parametrize('branch', ['detect', 'embed_frame', 'embed_crops'])
@pytest.mark.asyncio
async def test_a_branch_that_errored_fails_the_photo_as_detector_infer(
    tmp_path, branch: str
) -> None:
    svc, os_fake, triton, engine, sizes = engine_service(tmp_path)
    engine.failed_branch = branch
    big = register(sizes, _jpeg_bytes((640, 480), seed=23), (640, 480))
    result = await svc.ingest_batch([big], ['/x/b.jpg'])
    assert [(r.status, r.error_kind) for r in result.results] == [('failed', 'detector_infer')]
    assert triton.calls == []
    assert not os_fake.images


@pytest.mark.asyncio
async def test_a_letterbox_fill_of_three_values_runs_the_host_stage(tmp_path) -> None:
    """The engine fills its canvas with one grey; another fill would change the detector input."""
    svc, os_fake, triton, engine, sizes = engine_service(tmp_path)
    svc.profile = dataclasses.replace(svc.profile, letterbox_fill=(114, 114, 0))
    big = register(sizes, _jpeg_bytes((640, 480), seed=4), (640, 480))
    result = await svc.ingest_batch([big], ['/x/q.jpg'])
    assert result.summary.successful == 1
    assert engine.calls == []
    assert _routes(result, os_fake) == {'/x/q.jpg': 'cpu_host'}
    assert triton.calls == ['primary_end2end']


@pytest.mark.asyncio
async def test_gpu_decode_off_runs_the_host_stage_and_never_asks_the_engine(tmp_path) -> None:
    svc, os_fake, triton, engine, sizes = engine_service(tmp_path)
    svc.config = dataclasses.replace(svc.config, ingest_gpu_decode=False)
    big = register(sizes, _jpeg_bytes((640, 480), seed=6), (640, 480))
    small = _jpeg_bytes((320, 240), seed=15)
    before = _route_sample('cpu_host')
    result = await svc.ingest_batch([big, small], ['/x/a.jpg', '/x/b.jpg'])
    assert engine.calls == []
    assert engine.readiness_probes == 0
    assert triton.loaded == []
    assert _routes(result, os_fake) == {'/x/a.jpg': 'cpu_host', '/x/b.jpg': 'cpu_host'}
    assert result.summary.decode_routes == {'cpu_host': 2}
    assert _route_sample('cpu_host') == before + 2


@pytest.mark.asyncio
async def test_a_service_without_an_engine_client_runs_the_host_stage(tmp_path) -> None:
    svc, os_fake, _triton, _engine, sizes = engine_service(tmp_path)
    svc.frame_engine = None
    big = register(sizes, _jpeg_bytes((640, 480), seed=24), (640, 480))
    result = await svc.ingest_batch([big], ['/x/n.jpg'])
    assert _routes(result, os_fake) == {'/x/n.jpg': 'cpu_host'}


@pytest.mark.asyncio
async def test_an_engine_that_is_not_ready_answers_503_before_any_work(tmp_path) -> None:
    svc, os_fake, triton, engine, sizes = engine_service(tmp_path)
    engine.ready = False
    big = register(sizes, _jpeg_bytes((640, 480), seed=25), (640, 480))
    with pytest.raises(EngineUnavailableError):
        await svc.ingest_batch([big], ['/x/r.jpg'])
    assert engine.calls == []
    assert triton.calls == []
    assert not os_fake.images
    engine.ready = True  # a later request asks again
    assert (await svc.ingest_batch([big], ['/x/r.jpg'])).summary.successful == 1


@pytest.mark.asyncio
async def test_an_all_duplicate_batch_does_not_ask_the_engine(tmp_path) -> None:
    svc, _os, _triton, engine, sizes = engine_service(tmp_path)
    big = register(sizes, _jpeg_bytes((640, 480), seed=26), (640, 480))
    await svc.ingest_batch([big], ['/x/d.jpg'])
    ingest_gpu_decode._READY.clear()
    engine.ready = False
    probes = engine.readiness_probes
    again = await svc.ingest_batch([big], ['/x/d.jpg'])
    assert again.results[0].status == 'duplicate'
    assert engine.readiness_probes == probes


@pytest.mark.asyncio
async def test_one_photo_during_an_engine_restart_raises_for_the_503(tmp_path) -> None:
    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path)
    engine.ready = False
    big = register(sizes, _jpeg_bytes((640, 480), seed=27), (640, 480))
    with pytest.raises(EngineUnavailableError):
        await svc.ingest_one(big, '/x/one.jpg')
    assert not os_fake.images


@pytest.mark.asyncio
async def test_a_batch_with_some_photos_unanswered_is_partial_not_a_503(tmp_path) -> None:
    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path)
    ok = register(sizes, _jpeg_bytes((640, 480), seed=28), (640, 480))
    down = register(sizes, _jpeg_bytes((640, 480), seed=29), (640, 480))
    original = engine.infer

    async def infer(model: str, inputs: list, outputs: list, **k: Any) -> Any:
        if len(engine.calls) == 1:
            engine.raises = OSError('engine died')
        else:
            engine.raises = None
        return await original(model, inputs, outputs, **k)

    engine.infer = infer
    result = await svc.ingest_batch([ok, down], ['/x/ok.jpg', '/x/down.jpg'])
    assert result.status == 'partial'
    assert sorted(r.status for r in result.results) == ['failed', 'success']
    assert [r.error_kind for r in result.results if r.status == 'failed'] == ['engine_unavailable']
    assert len(os_fake.images) == 1


def test_the_gpu_decode_is_on_by_default() -> None:
    from src.config.curation import CurationConfig

    assert CurationConfig().ingest_gpu_decode is True
    assert not hasattr(CurationConfig(), 'ingest_gpu_min_megapixels')


@pytest.mark.asyncio
async def test_a_model_that_fails_to_load_is_not_ready_and_is_asked_again(tmp_path) -> None:
    """Triton out of memory for the detector or the encoder the engine calls: the load raises,
    the request answers 503 and nothing is processed. A failure is a transient, not a verdict, so
    the next request asks again."""
    svc, os_fake, triton, engine, sizes = engine_service(tmp_path)
    attempts: list[str] = []
    real_load = triton.load_model

    async def load_model(model: str) -> None:
        attempts.append(model)
        if len(attempts) <= 2:
            raise InferenceServerException('failed to load: CUDA out of memory')
        await real_load(model)

    triton.load_model = load_model
    big = register(sizes, _jpeg_bytes((640, 480), seed=7), (640, 480))
    for _ in range(2):
        with pytest.raises(EngineUnavailableError):
            await svc.ingest_batch([big], ['/x/a.jpg'])
    assert engine.calls == []
    assert (await svc.ingest_batch([big], ['/x/a.jpg'])).summary.successful == 1
    assert attempts[:2] == ['primary_end2end'] * 2
    assert len(os_fake.images) == 1


@pytest.mark.asyncio
async def test_a_verdict_is_kept_for_five_seconds_then_asked_again(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc, _os, _triton, engine, _sizes = engine_service(tmp_path)
    now = [1000.0]
    monkeypatch.setattr(ingest_gpu_decode.time, 'monotonic', lambda: now[0])
    assert await ingest_gpu_decode.engine_ready(svc) is True
    now[0] += ingest_gpu_decode.READY_TTL_S - 0.1
    assert await ingest_gpu_decode.engine_ready(svc) is True
    assert engine.readiness_probes == 1
    now[0] += 0.2
    engine.ready = False
    assert await ingest_gpu_decode.engine_ready(svc) is False
    assert engine.readiness_probes == 2


@pytest.mark.asyncio
async def test_a_failed_engine_call_drops_the_verdict(tmp_path) -> None:
    svc, _os, _triton, engine, sizes = engine_service(tmp_path)
    assert await ingest_gpu_decode.engine_ready(svc) is True
    assert ingest_gpu_decode._READY
    engine.raises = OSError('engine died')
    big = register(sizes, _jpeg_bytes((640, 480), seed=30), (640, 480))
    with pytest.raises(EngineUnavailableError):
        await svc.ingest_batch([big], ['/x/z.jpg'])
    assert not ingest_gpu_decode._READY  # the next request probes again


@pytest.mark.asyncio
async def test_a_transient_probe_failure_is_not_cached(tmp_path) -> None:
    svc, _os, triton, _engine, _sizes = engine_service(tmp_path)
    real_load = triton.load_model
    failures = [True]

    async def flaky_load(model: str) -> None:
        if failures:
            failures.pop()
            raise InferenceServerException('triton restarting')
        await real_load(model)

    triton.load_model = flaky_load
    assert await ingest_gpu_decode.engine_ready(svc) is False
    assert await ingest_gpu_decode.engine_ready(svc) is True


@pytest.mark.asyncio
async def test_every_route_decision_is_on_the_result_and_the_image_doc(tmp_path) -> None:
    svc, os_fake, _triton, _engine, sizes = engine_service(tmp_path)
    big = register(sizes, _jpeg_bytes((640, 480), seed=11), (640, 480))
    not_image = b'not an image at all'
    result = await svc.ingest_batch([big, not_image], ['/x/big.jpg', '/x/bad.bin'])
    by_path = _routes(result, os_fake)
    assert by_path == {'/x/big.jpg': 'gpu', '/x/bad.bin': None}
    assert result.summary.decode_routes == {'gpu': 1}
    bad = next(r for r in result.results if r.image_path == '/x/bad.bin')
    assert (bad.status, bad.error_kind) == ('failed', 'decode_failed')  # the engine said corrupt


@pytest.mark.parametrize('route', ['cpu_small', 'cpu_not_jpeg', 'cpu_progressive', 'cpu_truncated'])
@pytest.mark.asyncio
async def test_a_photo_the_engine_decodes_on_the_host_carries_its_reason(
    tmp_path, route: str
) -> None:
    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path)
    engine.route = route
    big = register(sizes, _jpeg_bytes((640, 480), seed=13), (640, 480))
    before = _route_sample(route)
    result = await svc.ingest_batch([big], ['/x/d.jpg'])
    assert _routes(result, os_fake) == {'/x/d.jpg': route}
    assert result.summary.decode_routes == {route: 1}
    assert _route_sample(route) == before + 1


@pytest.mark.asyncio
async def test_the_single_image_path_goes_to_the_engine_too(tmp_path) -> None:
    svc, os_fake, triton, engine, sizes = engine_service(tmp_path)
    big = register(sizes, _jpeg_bytes((640, 480), seed=17), (640, 480))
    result = await svc.ingest_one(big, '/x/one.jpg')
    assert len(engine.calls) == 1
    assert triton.calls == []
    assert result.decode_route == 'gpu'
    [doc] = list(os_fake.images.values())
    assert doc['decode_route'] == 'gpu'


@pytest.mark.asyncio
async def test_a_duplicate_has_no_route_and_the_doc_keeps_its_first_route(tmp_path) -> None:
    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path)
    big = register(sizes, _jpeg_bytes((640, 480), seed=18), (640, 480))
    first = await svc.ingest_batch([big], ['/x/1.jpg'])
    assert first.results[0].decode_route == 'gpu'
    engine.route = 'cpu_small'
    again = await svc.ingest_batch([big], ['/x/1.jpg'])
    assert again.results[0].status == 'duplicate'
    assert again.results[0].decode_route is None
    assert again.summary.decode_routes == {}
    [doc] = list(os_fake.images.values())
    assert doc['decode_route'] == 'gpu'


def _probe_count(outcome: str) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value('op_engine_probe_total', {'outcome': outcome}) or 0.0


@pytest.fixture
def _no_probe_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_wait(_s: float) -> None:
        return None

    monkeypatch.setattr(frame_engine, 'RETRY_SLEEP', no_wait)


@pytest.mark.asyncio
@pytest.mark.usefixtures('_no_probe_wait')
async def test_a_probe_the_transport_refuses_is_asked_again_before_the_503(tmp_path) -> None:
    """A GOAWAY or a reset on the probe's channel is not the engine saying "not ready"."""
    svc, _os, _triton, engine, _sizes = engine_service(tmp_path)
    engine.probe_errors = [
        InferenceServerException('Too many pings', status=str(grpc.StatusCode.UNAVAILABLE))
    ]
    before = _probe_count('recovered')
    assert await ingest_gpu_decode.engine_ready(svc) is True
    assert engine.readiness_probes == 2
    assert _probe_count('recovered') == before + 1


@pytest.mark.asyncio
@pytest.mark.usefixtures('_no_probe_wait')
async def test_a_probe_that_is_never_delivered_is_counted_and_logged_apart(tmp_path) -> None:
    from structlog.testing import capture_logs

    svc, _os, _triton, engine, _sizes = engine_service(tmp_path)
    refusal = InferenceServerException(
        'connection refused', status=str(grpc.StatusCode.UNAVAILABLE)
    )
    engine.probe_errors = [refusal] * frame_engine.PROBE_ATTEMPTS
    before = _probe_count('undelivered')
    with capture_logs() as logs:
        assert await ingest_gpu_decode.engine_ready(svc) is False
    assert engine.readiness_probes == frame_engine.PROBE_ATTEMPTS
    assert _probe_count('undelivered') == before + 1
    [line] = [x for x in logs if x['event'] == 'ingest_engine_probe_undelivered']
    assert line['attempts'] == frame_engine.PROBE_ATTEMPTS
    assert line['code'] == 'StatusCode.UNAVAILABLE'
    assert not [x for x in logs if x['event'] == 'ingest_engine_not_ready']


@pytest.mark.asyncio
@pytest.mark.usefixtures('_no_probe_wait')
async def test_an_engine_that_answers_not_ready_is_believed_at_once(tmp_path) -> None:
    from structlog.testing import capture_logs

    svc, _os, _triton, engine, _sizes = engine_service(tmp_path)
    engine.ready = False
    before = _probe_count('not_ready')
    with capture_logs() as logs:
        assert await ingest_gpu_decode.engine_ready(svc) is False
    assert engine.readiness_probes == 1  # an answer is not retried
    assert _probe_count('not_ready') == before + 1
    assert [x for x in logs if x['event'] == 'ingest_engine_not_ready']
    assert not [x for x in logs if x['event'] == 'ingest_engine_probe_undelivered']
