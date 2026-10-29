"""The embed pass on the GPU decode (#259): a photo at or above the ingest size gate is sent to
``gpu_image_prep`` with its stored boxes and never decoded in the API; everything else, and every
photo the model declines, takes the CPU path as before. No GPU here: a fake pool answers."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from prometheus_client import REGISTRY

from curation.reprocess_fixtures import (
    F,
    FakePE,
    FakeTriton,
    box,
    docs,
    images_index,
    item,
    jpeg_bytes,
    make_fake,
    make_service,
    servable_root,
)
from src.config.curation import CurationConfig
from src.services.curation import ingest_gpu_decode, reprocess_embed
from src.services.curation.reprocess_embed import EmbedTarget, reembed_items
from src.services.curation.reprocess_embed_spec import EmbedSpec
from src.services.curation.reprocess_gpu_embed import BOXES_INPUT, gpu_embed_route
from src.services.curation.reprocess_images import process_images


if TYPE_CHECKING:
    from pathlib import Path

CROP_BOX = (0.1, 0.1, 0.6, 0.6)
REGION_BOX = (0.2, 0.2, 0.4, 0.3)
PARAMS = {'DETECTOR_MODEL': 'primary_end2end', 'NET_SIZE': '320', 'LETTERBOX_FILL': '114'}


class _Result:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self.outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self.outputs[name]


class GpuTriton(FakeTriton):
    """``gpu_image_prep`` in stored-box mode: one row per box sent, row ``k`` is ``[1, k, 0]``."""

    def __init__(
        self, *, status: str = 'ok', inputs: tuple[str, ...] = ('jpeg', 'embed_crops', 'boxes')
    ) -> None:
        super().__init__([])
        self.status = status
        self.model_inputs = list(inputs)
        self.streams: list[dict[str, Any]] = []
        self.fail: Exception | None = None
        self.rows_short = False

    async def is_model_ready(self, _model: str) -> bool:
        return True

    async def get_model_parameters(self, model: str) -> dict[str, str]:
        assert model == ingest_gpu_decode.GPU_IMAGE_PREP_MODEL
        return PARAMS

    async def get_model_input_names(self, model: str) -> list[str]:
        assert model == ingest_gpu_decode.GPU_IMAGE_PREP_MODEL
        return self.model_inputs

    async def stream_one(self, model: str, inputs: list) -> _Result:
        assert model == ingest_gpu_decode.GPU_IMAGE_PREP_MODEL
        by_name = {i.name(): i for i in inputs}
        boxes = np.frombuffer(by_name[BOXES_INPUT]._raw_content, np.float64).reshape(-1, 4)
        self.streams.append(
            {
                'jpeg': bytes(by_name['jpeg']._raw_content),
                'boxes': boxes,
                'dtype': by_name[BOXES_INPUT].datatype(),
                'names': sorted(by_name),
            }
        )
        if self.fail is not None:
            raise self.fail
        if self.status != 'ok':
            return _Result({'status': np.array([self.status.encode()], np.object_)})
        n = boxes.shape[0] - (1 if self.rows_short else 0)
        rows = np.array([[1.0, float(k), 0.0] for k in range(n)], np.float32).reshape(-1, 3)
        return _Result(
            {
                'status': np.array([b'ok'], np.object_),
                'size': np.array([400, 300], np.int32),
                'whole_embedding': np.array([0.0, 0.0, 2.0], np.float32),
                'crop_embeddings': rows,
            }
        )


def _unit(v: list[float]) -> list[float]:
    a = np.asarray(v, np.float64)
    return list(a / np.linalg.norm(a))


@pytest.fixture(autouse=True)
def _fresh_ready_cache() -> Any:
    ingest_gpu_decode._READY.clear()
    yield
    ingest_gpu_decode._READY.clear()


def _sample(route: str) -> float:
    return REGISTRY.get_sample_value('op_reprocess_decode_total', {'route': route}) or 0.0


def _world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    triton: GpuTriton,
    *,
    gate: float = 0.1,
    gpu_decode: bool = True,
) -> tuple[Any, FakePE, Any, bytes, EmbedTarget]:
    root = servable_root(tmp_path, monkeypatch)
    data = jpeg_bytes((400, 300), seed=11)  # 0.12 MP
    path = root / 'a.jpg'
    path.write_bytes(data)
    doc = item(
        'c0',
        image_path=str(path),
        bbox_norm=CROP_BOX,
        boxes=(
            box('b1', state='accepted', bbox=REGION_BOX, reason=None),
            box('b2', state='accepted', bbox=CROP_BOX, reason=None),  # same box as the crop
            box('b3', state='rejected', bbox=(0.5, 0.5, 0.7, 0.7)),  # never embedded
        ),
    )
    fake = make_fake([doc], [{'image_id': 'img-1', 'image_path': str(path)}])
    pe = FakePE()
    service = make_service(fake, triton, pe)
    service.config = dataclasses.replace(
        CurationConfig(), ingest_gpu_decode=gpu_decode, ingest_gpu_min_megapixels=gate
    )
    target = EmbedTarget(image_id='img-1', image_path=str(path), items=[('c0', doc)])
    return fake, pe, service, data, target


@pytest.mark.asyncio
async def test_a_photo_above_the_gate_is_embedded_from_the_gpu_decode_with_its_stored_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triton = GpuTriton()
    fake, pe, service, data, target = _world(tmp_path, monkeypatch, triton)

    def no_cpu_decode(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError('the photo was decoded in the API')

    monkeypatch.setattr(reprocess_embed, '_decode_image', no_cpu_decode)
    before = _sample('gpu')
    route = await gpu_embed_route(service)
    assert route is not None
    counts = await reembed_items(fake, pe, [target], gpu=route)

    [call] = triton.streams
    assert call['jpeg'] == data  # the stored file as it is, decoded once on the GPU
    assert call['dtype'] == 'FP64'  # the stored floats exactly, so the rectangle is the CPU's
    assert call['names'] == ['boxes', 'jpeg']
    # The crop box, then the embeddable region boxes; a region box equal to the crop box is
    # sent once and the rejected box not at all.
    np.testing.assert_array_equal(call['boxes'], np.array([CROP_BOX, REGION_BOX], np.float64))
    assert (pe.crop_calls, pe.frame_calls) == (0, 0)

    stored = docs(fake)['c0']
    assert stored['pe_embedding'] == pytest.approx(_unit([1.0, 0.0, 0.0]))
    assert stored['embedding_state'] == 'embedded'
    vectors = {e['box_id']: e['embedding'] for e in stored[F.box_embeddings]}
    assert vectors['b1'] == pytest.approx(_unit([1.0, 1.0, 0.0]))
    assert vectors['b2'] == pytest.approx(_unit([1.0, 0.0, 0.0]))
    assert set(vectors) == {'b1', 'b2'}
    assert fake.docs(images_index())['img-1']['pe_embedding'] == pytest.approx([0.0, 0.0, 1.0])
    assert counts['crop_written'] == 1
    assert counts['frame_written'] == 1
    assert counts['region_written'] == 2
    assert _sample('gpu') == before + 1


@pytest.mark.asyncio
async def test_a_photo_below_the_gate_keeps_the_cpu_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triton = GpuTriton()
    fake, pe, service, _data, target = _world(tmp_path, monkeypatch, triton, gate=3.0)
    before = _sample('cpu_small')
    route = await gpu_embed_route(service)
    await reembed_items(fake, pe, [target], gpu=route)
    assert triton.streams == []
    assert pe.crop_calls == 3  # the crop, then the two region boxes (decoded from JPEG)
    assert pe.frame_calls == 1
    assert docs(fake)['c0']['pe_embedding'] == pytest.approx([0.0, 0.0, 1.0])  # the CPU fake's
    assert _sample('cpu_small') == before + 1


@pytest.mark.parametrize(
    ('status', 'fail', 'short', 'route'),
    [
        ('progressive', None, False, 'cpu_progressive'),
        ('ok', RuntimeError('stream closed'), False, 'cpu_call_failed'),
        ('ok', None, True, 'cpu_box_count'),
    ],
)
@pytest.mark.asyncio
async def test_a_photo_the_gpu_decode_declines_falls_back_to_the_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    fail: Exception | None,
    short: bool,
    route: str,
) -> None:
    triton = GpuTriton(status=status)
    triton.fail = fail
    triton.rows_short = short
    fake, pe, service, _data, target = _world(tmp_path, monkeypatch, triton)
    before = _sample(route)
    counts = await reembed_items(fake, pe, [target], gpu=await gpu_embed_route(service))
    assert len(triton.streams) == 1
    assert (pe.crop_calls, pe.frame_calls) == (3, 1)
    assert docs(fake)['c0']['pe_embedding'] == pytest.approx([0.0, 0.0, 1.0])
    assert counts['crop_written'] == 1
    assert counts['region_written'] == 2
    assert _sample(route) == before + 1


@pytest.mark.asyncio
async def test_a_model_without_the_boxes_input_is_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model files older than the API (not re-copied): fail closed to the CPU path once, not one
    refused call per photo."""
    triton = GpuTriton(inputs=('jpeg', 'embed_crops'))
    _fake, _pe, service, _data, _target = _world(tmp_path, monkeypatch, triton)
    assert await gpu_embed_route(service) is None


@pytest.mark.asyncio
async def test_gpu_decode_off_gives_no_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triton = GpuTriton()
    _fake, _pe, service, _data, _target = _world(tmp_path, monkeypatch, triton, gpu_decode=False)
    assert await gpu_embed_route(service) is None


@pytest.mark.asyncio
async def test_the_reprocess_embed_scope_takes_the_gpu_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triton = GpuTriton()
    fake, pe, service, _data, _target = _world(tmp_path, monkeypatch, triton)
    [res], cancelled = await process_images(
        fake,
        service,
        scopes=['embed'],
        image_ids=['img-1'],
        embed=EmbedSpec(parts=['crop', 'frame', 'region']),
    )
    assert not cancelled
    assert (res.queued, res.failed) == (1, 0)
    assert len(triton.streams) == 1
    assert (pe.crop_calls, pe.frame_calls) == (0, 0)
    assert docs(fake)['c0']['pe_embedding'] == pytest.approx(_unit([1.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_only_the_parts_asked_for_are_sent_and_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triton = GpuTriton()
    fake, pe, service, _data, target = _world(tmp_path, monkeypatch, triton)
    counts = await reembed_items(
        fake, pe, [target], parts=frozenset({'region'}), gpu=await gpu_embed_route(service)
    )
    [call] = triton.streams
    np.testing.assert_array_equal(call['boxes'], np.array([REGION_BOX, CROP_BOX], np.float64))
    assert (counts['crop_written'], counts['frame_written'], counts['region_written']) == (0, 0, 2)
    assert 'pe_embedding' not in fake.docs(images_index())['img-1']
