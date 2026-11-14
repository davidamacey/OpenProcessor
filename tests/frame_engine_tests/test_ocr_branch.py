"""The ``ocr`` branch (``frame_engine/branches/ocr.py``) inside the engine, against stand-ins for
CuPy, the decoder, the CUDA kernels and Triton: the control flow, the batches, the failure mapping
and the outputs are the real code. The detector answers a map with known bright rectangles; the
recogniser answers per-step indices that spell the batch position of each line."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from frame_engine.transport import ModelCallError
from src.domain import frame_engine_wire as wire

from . import test_engine as te
from .test_engine import _Kernels, _plan, _run, _Transport


eng = te.eng  # the engine fixture of test_engine.py, collected here by name


OCR = {
    'detector': 'ocr_det',
    'recogniser': 'ocr_rec',
    'batch': 2,
    'max_lines': 128,
    'det_side': 960,
}
CHARS = ['blank', 'a', 'b', 'c', 'd', 'e', ' ']


class _OcrTransport(_Transport):
    """Detector map with ``RECTS`` (frame is 96 x 64, map the same size); recogniser: step 0 of
    line ``k`` of a call is class ``1 + k``, so the text is 'a', 'b', ... per batch position."""

    RECTS: list[tuple[int, int, int, int]] = []

    def collect(self, p: Any) -> dict[str, Any]:
        if p.model == 'ocr_det':
            self.events.append(('collect', p.model))
            h, w = p.shape[2], p.shape[3]
            pred = np.zeros((1, 1, h, w), np.float32)
            for x1, y1, x2, y2 in self.RECTS:
                pred[0, 0, y1:y2, x1:x2] = 0.9
            return {p.outputs[0]: pred}
        if p.model == 'ocr_rec':
            self.events.append(('collect', p.model))
            if p.model in self.fail:
                raise self.fail[p.model]
            n, _, _, width = p.shape
            idx = np.zeros((n, width // 4), np.int32)
            idx[np.arange(n), 0] = 1 + np.arange(n)
            return {'indices': idx, 'scores': np.full(idx.shape, 0.75, np.float32)}
        return super().collect(p)


class _Warp:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def warp_into(self, frame: Any, maps: Any, widths: Any, taps: Any, dst: Any) -> None:
        self.calls.append((maps.copy(), widths.copy(), taps.copy(), dst.shape))
        dst[...] = 7


@pytest.fixture
def ocr_engine(eng: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    d = tmp_path / 'dict.txt'
    d.write_text('\n'.join(CHARS[1:-1]) + '\n', encoding='utf-8')
    warp = _Warp()
    monkeypatch.setattr(eng.module, 'LineWarp', lambda: warp)

    def build(rects: list[tuple[int, int, int, int]], **over: Any) -> Any:
        _OcrTransport.RECTS = rects
        engine = eng(_OcrTransport(), ocr_dictionary=str(d), **over)
        engine.warp = warp
        return engine

    return build


def _ocr_plan(branches: list[str] | None = None, **over: Any) -> Any:
    body: dict[str, Any] = {'embed': None, 'crops': 'none', 'ocr': {**OCR, **over}}
    return _plan(branches or ['ocr'], detector=None, **body)


RECTS = [(10, 8, 50, 20), (10, 30, 30, 40), (50, 30, 90, 44)]


def test_lines_are_batched_decoded_and_reported_in_reading_order(ocr_engine: Any) -> None:
    engine = ocr_engine(RECTS)
    r = _run(engine, _ocr_plan())
    assert r.status == 'ok'
    assert r.branches == {'ocr': ('ok', '')}
    assert set(r.outputs) == {'size', *wire.expected_outputs(_ocr_plan())}
    models = engine.transport.models()
    assert models[0] == 'ocr_det'
    assert models.count('ocr_rec') == 2
    quads = r.outputs['ocr_quads']
    assert quads.shape == (3, 8)
    assert quads.dtype == np.float32
    assert np.all(np.diff(quads[:, 1]) >= -10)  # top to bottom
    assert quads[0, 0] < 20
    assert quads[0, 1] < 20
    assert len(r.outputs['ocr_texts']) == 3
    assert all(isinstance(t, bytes) for t in r.outputs['ocr_texts'])
    assert np.allclose(r.outputs['ocr_scores'], 0.75)
    assert np.allclose(r.outputs['ocr_char_min'], 0.75)
    assert r.outputs['ocr_det_scores'].shape == (3,)
    assert np.all(r.outputs['ocr_det_scores'] > 0.45)
    assert r.outputs['ocr_overflow'].tolist() == [0]
    assert engine.transport.staged == 0


def test_the_batches_follow_the_recipe_widest_last_and_width_at_least_320(ocr_engine: Any) -> None:
    engine = ocr_engine(RECTS)
    _run(engine, _ocr_plan())
    shapes = [s for m, s in engine.transport.calls if m == 'ocr_rec']
    assert [s[0] for s in shapes] == [2, 1]
    assert all(s[1:3] == [3, 48] and s[3] >= 320 for s in shapes)
    assert shapes[0][3] <= shapes[1][3]  # narrowest crops first
    maps, widths, taps, _ = engine.warp.calls[0]
    assert maps.shape == (2, 9)
    assert widths.shape == (2,)
    assert taps.shape == (2, 2)


def test_a_photo_without_text_costs_the_detector_call_and_nothing_else(ocr_engine: Any) -> None:
    engine = ocr_engine([])
    r = _run(engine, _ocr_plan())
    assert engine.transport.models() == ['ocr_det']
    assert engine.warp.calls == []
    assert r.branches == {'ocr': ('ok', '')}
    assert r.outputs['ocr_quads'].shape == (0, 8)
    assert r.outputs['ocr_texts'] == []
    assert r.outputs['ocr_overflow'].tolist() == [0]


def test_lines_past_the_cap_are_dropped_and_counted_not_silent(ocr_engine: Any) -> None:
    engine = ocr_engine(RECTS)
    r = _run(engine, _ocr_plan(max_lines=2))
    assert r.outputs['ocr_quads'].shape == (2, 8)
    assert r.outputs['ocr_overflow'].tolist() == [1]
    assert engine.transport.models().count('ocr_rec') == 1


def test_the_detector_input_is_written_as_bgr(ocr_engine: Any) -> None:
    seen: list[Any] = []

    class Kernels(_Kernels):
        def resample_into(
            self, frame: Any, roi: Any, size: Any, dst: Any, *a: Any, **k: Any
        ) -> None:
            seen.append((dst.strides[0], dst.shape, k))
            super().resample_into(frame, roi, size, dst, *a, **k)

    engine = ocr_engine(RECTS)
    engine.kernels = Kernels()
    _run(engine, _ocr_plan())
    stride, shape, kwargs = seen[0]
    assert stride < 0, 'the detector input must be the frame with its planes reversed (RGB to BGR)'
    assert shape[0] == 3
    assert kwargs == {'antialias': False}


def test_a_recogniser_failure_is_the_ocr_status_alone(ocr_engine: Any) -> None:
    engine = ocr_engine(RECTS)
    engine.transport.fail['ocr_rec'] = ModelCallError('ocr_rec', 'INTERNAL', 'boom')
    plan = _plan(['detect', 'ocr'], embed=None, crops='none', ocr={**OCR})
    r = _run(engine, plan)
    assert r.status == 'ok'
    assert r.branches['ocr'] == ('error', 'model_error')
    assert r.branches['detect'] == ('ok', '')
    assert not {'ocr_texts', 'ocr_quads'} & set(r.outputs)
    assert engine.transport.staged == 0


def test_a_missing_dictionary_fails_the_photo_not_the_branch(eng: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(eng.module, 'LineWarp', lambda: _Warp())
    _OcrTransport.RECTS = RECTS
    engine = eng(_OcrTransport(), ocr_dictionary='/nonexistent/dict.txt')
    with pytest.raises(FileNotFoundError):
        _run(engine, _ocr_plan())
