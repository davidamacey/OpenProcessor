"""#220: the OCR pipeline takes uint8 pixels (a quarter of the bytes) from both callers.

``ocr_images`` and ``original_image`` are sent as UINT8 CHW BGR; the ``ocr_pipeline`` BLS
normalizes the detector input itself. A pipeline model still declaring FP32 is refused with the
repair command (no FP32 fallback). The arithmetic is checked bit-for-bit against the old
client-side float path, so the change cannot move OCR output.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest
from PIL import Image
from test_ocr_pipeline_rec_width import _FakeTensor, model_module  # noqa: F401 -- fixture + stub

from src.clients import triton_client as tc, triton_ocr
from src.clients.triton_pool_core import AsyncTritonPool
from src.services.detection.cascade_detect.ocr_recognizer import PaddleOcrTextRecognizer


if TYPE_CHECKING:
    from tritonclient.grpc import InferInput

RNG = np.random.default_rng(7)
BGR = RNG.integers(0, 256, (70, 90, 3), dtype=np.uint8)


class _StopError(Exception):
    """Raised by the stubs to end a call once the inputs were captured."""


def _png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.png', arr)
    assert ok
    return buf.tobytes()


def _values(inp: InferInput) -> np.ndarray:
    dtypes = {'UINT8': np.uint8, 'FP32': np.float32, 'INT32': np.int32}
    return np.frombuffer(inp._get_content(), dtype=dtypes[inp.datatype()]).reshape(inp.shape())


class _Response:
    def as_numpy(self, name: str) -> np.ndarray:
        return {
            'num_texts': np.array([0], dtype=np.int32),
            'text_boxes': np.zeros((128, 8), np.float32),
            'text_boxes_normalized': np.zeros((128, 4), np.float32),
            'texts': np.array([b''] * 128, dtype=object),
            'text_scores': np.zeros(128, np.float32),
            'rec_scores': np.zeros(128, np.float32),
        }[name]


def _run_client(datatype: str, image: bytes) -> list[Any]:
    sent: list[Any] = []

    def infer(_model: str, inputs: list, _outputs: list) -> _Response:
        sent.extend(inputs)
        return _Response()

    triton_ocr.infer_ocr(infer, image, lambda _name: datatype)
    return sent


class TestSyncClient:
    def test_sends_uint8_chw_pixels(self) -> None:
        by_name = {i.name(): i for i in _run_client('UINT8', _png(BGR))}
        assert by_name['ocr_images'].datatype() == 'UINT8'
        assert by_name['original_image'].datatype() == 'UINT8'
        assert by_name['original_image'].shape() == [3, 70, 90]
        # Exactly H*W*3 bytes: a quarter of the old float tensor.
        assert len(by_name['original_image']._get_content()) == 70 * 90 * 3
        assert np.array_equal(_values(by_name['original_image']), BGR.transpose(2, 0, 1))

    @pytest.mark.parametrize('datatype', ['FP32', ''])
    def test_non_uint8_model_is_refused_before_any_request(self, datatype: str) -> None:
        called: list[Any] = []

        def infer(*a: Any) -> None:
            called.append(a)

        with pytest.raises(ValueError, match=r"'ocr_images'.*models repair"):
            triton_ocr.infer_ocr(infer, _png(BGR), lambda _n: datatype)
        assert called == []

    def test_second_input_is_checked_too(self) -> None:
        with pytest.raises(ValueError, match="'original_image'"):
            triton_ocr.infer_ocr(
                lambda *_a: None,
                _png(BGR),
                lambda n: 'UINT8' if n == 'ocr_images' else 'FP32',
            )

    def test_sixteen_bit_png_is_scaled_to_uint8(self) -> None:
        wide = np.full((40, 40, 3), 65535, dtype=np.uint16)
        sent = _run_client('UINT8', _png(wide))
        orig = next(i for i in sent if i.name() == 'original_image')
        assert set(np.unique(_values(orig))) == {255}

    def test_grey_input_becomes_three_channels(self) -> None:
        grey = RNG.integers(0, 256, (40, 50), dtype=np.uint8)
        sent = _run_client('UINT8', _png(grey))
        orig = next(i for i in sent if i.name() == 'original_image')
        assert orig.shape() == [3, 40, 50]
        assert np.array_equal(_values(orig)[0], grey)

    def test_detector_input_normalizes_like_the_old_client_side_float_tensor(self) -> None:
        sent = _run_client('UINT8', _png(BGR))
        det = _values(next(i for i in sent if i.name() == 'ocr_images'))
        old = det.transpose(1, 2, 0).astype(np.float32) / 127.5 - 1.0
        new = det.astype(np.float32) / 127.5 - 1.0
        assert np.array_equal(old.transpose(2, 0, 1), new)


class _MetaClient:
    def __init__(self) -> None:
        self.calls = 0

    async def get_model_metadata(self, _name: str, **_kw: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            'inputs': [
                {'name': 'ocr_images', 'datatype': 'UINT8'},
                {'name': 'original_image', 'datatype': 'FP32'},
            ]
        }


def _pool(monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncTritonPool, _MetaClient]:
    async def _loaded(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr('src.clients.triton_pool_core.ensure_loaded', _loaded)
    pool = AsyncTritonPool(pool_size=1)
    meta = _MetaClient()
    pool._clients = [meta]  # type: ignore[list-item]
    pool._lock = asyncio.Lock()
    pool._initialized = True
    return pool, meta


class TestPoolDatatypeByInput:
    def test_reads_the_named_input_and_caches_per_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pool, meta = _pool(monkeypatch)

        async def run() -> list[str]:
            return [
                await pool.get_model_input_datatype('m', input_name='ocr_images'),
                await pool.get_model_input_datatype('m', input_name='original_image'),
                await pool.get_model_input_datatype('m', input_name='original_image'),
            ]

        assert asyncio.run(run()) == ['UINT8', 'FP32', 'FP32']
        assert meta.calls == 2

    def test_missing_input_names_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pool, _ = _pool(monkeypatch)
        with pytest.raises(ValueError, match='nope'):
            asyncio.run(pool.get_model_input_datatype('m', input_name='nope'))


class TestSyncDatatypeLookup:
    def test_reads_and_caches_per_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        metadata = SimpleNamespace(
            inputs=[
                SimpleNamespace(name='ocr_images', datatype='UINT8'),
                SimpleNamespace(name='original_image', datatype='UINT8'),
            ]
        )
        monkeypatch.setattr(tc, 'ensure_loaded_sync', lambda *_a: None)
        client = tc.TritonClient.__new__(tc.TritonClient)
        client._input_datatypes = {}
        client.client = SimpleNamespace(get_model_metadata=lambda n: calls.append(n) or metadata)
        assert client._input_datatype('ocr_pipeline', 'ocr_images') == 'UINT8'
        assert client._input_datatype('ocr_pipeline', 'ocr_images') == 'UINT8'
        assert len(calls) == 1
        with pytest.raises(ValueError, match='nope'):
            client._input_datatype('ocr_pipeline', 'nope')


class _PoolStub:
    def __init__(self, datatype: str) -> None:
        self.datatype = datatype
        self.sent: list[Any] = []

    async def get_model_input_datatype(self, _m: str, *, input_name: str) -> str:  # noqa: ARG002 -- mirrors the pool signature
        return self.datatype

    async def infer(self, _m: str, inputs: list, outputs: list) -> Any:  # noqa: ARG002 -- mirrors the pool signature
        self.sent = inputs
        raise _StopError


def _recognizer(datatype: str) -> tuple[PaddleOcrTextRecognizer, _PoolStub]:
    pool = _PoolStub(datatype)
    rec = PaddleOcrTextRecognizer.__new__(PaddleOcrTextRecognizer)
    rec.triton_pool = pool  # type: ignore[assignment]
    rec.model_name = 'ocr_pipeline'
    return rec, pool


class TestWorkerClient:
    IMG = Image.fromarray(BGR[:, :, ::-1])

    def test_preprocess_is_uint8_bgr_chw(self) -> None:
        rec, _ = _recognizer('UINT8')
        det, orig, shape = rec._preprocess(self.IMG)
        assert det.dtype == np.uint8
        assert orig.dtype == np.uint8
        assert np.array_equal(orig, BGR.transpose(2, 0, 1))
        assert shape.tolist() == [70, 90]

    def test_sends_uint8(self) -> None:
        rec, pool = _recognizer('UINT8')
        with pytest.raises(_StopError):
            asyncio.run(rec._infer_lines(self.IMG))
        assert [i.datatype() for i in pool.sent[:2]] == ['UINT8', 'UINT8']

    def test_fp32_pipeline_is_refused(self) -> None:
        rec, pool = _recognizer('FP32')
        with pytest.raises(ValueError, match='models repair'):
            asyncio.run(rec._infer_lines(self.IMG))
        assert pool.sent == []


class _Request:
    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        self.tensors = tensors


def _request(module: Any, ocr: np.ndarray, orig: np.ndarray) -> _Request:
    tensors = {
        'ocr_images': ocr,
        'original_image': orig,
        'orig_shape': np.array([4, 4], dtype=np.int32),
    }
    module.pb_utils.get_input_tensor_by_name = lambda r, n: _FakeTensor(n, r.tensors[n])
    return _Request(tensors)


def _bls(module: Any) -> Any:
    obj: Any = module.TritonPythonModel.__new__(module.TritonPythonModel)
    obj.seen = {}

    def detection(ocr_image: np.ndarray) -> np.ndarray:
        obj.seen['det_in'] = ocr_image
        raise _StopError

    obj._call_detection = detection
    return obj


class TestPipelineBLS:
    def test_uint8_is_normalized_for_the_detector(self, model_module: Any) -> None:  # noqa: F811
        model = _bls(model_module)
        u8 = RNG.integers(0, 256, (3, 32, 32), dtype=np.uint8)
        with pytest.raises(_StopError):
            model._process_request(_request(model_module, u8, np.zeros((3, 4, 4), np.uint8)))
        assert model.seen['det_in'].dtype == np.float32
        assert np.array_equal(model.seen['det_in'], u8.astype(np.float32) / 127.5 - 1.0)

    def test_crop_source_is_the_uint8_hwc_array(self, model_module: Any) -> None:  # noqa: F811
        model = _bls(model_module)
        model._call_detection = lambda _i: np.zeros((1, 1, 8, 8), np.float32)
        model.db_postprocess = lambda _o, _s: (np.zeros((1, 4, 2)), np.ones(1))
        model._sorted_boxes = lambda b: b
        model.max_texts = 8
        got: dict[str, Any] = {}

        def crops(img: np.ndarray, _boxes: Any) -> list:
            got['img'] = img
            return []

        model._get_text_crops = crops
        model._create_empty_response = lambda: 'empty'
        orig = RNG.integers(0, 256, (3, 5, 6), dtype=np.uint8)
        model._process_request(_request(model_module, np.zeros((3, 32, 32), np.uint8), orig))
        assert got['img'].dtype == np.uint8
        assert np.array_equal(got['img'], orig.transpose(1, 2, 0))

    def test_float_input_is_refused(self, model_module: Any) -> None:  # noqa: F811
        model = _bls(model_module)
        with pytest.raises(ValueError, match=r"'ocr_images' must be UINT8"):
            model._process_request(
                _request(
                    model_module, np.zeros((3, 32, 32), np.float32), np.zeros((3, 4, 4), np.uint8)
                )
            )
