"""The frame engine's ``ocr`` branch on a real GPU: rendered text photos through the engine's gRPC front.

Run (GPU tier; needs a frame engine image with the OCR dictionary baked in, and its Triton
with ``paddleocr_det_trt_uint8`` and ``paddleocr_rec_trt_uint8`` loaded, see docker/test/README.md
"GPU tier"):

    OP_TEST_FRAME_ENGINE_GRPC=<engine host:port> \\
        .venv/bin/python -m pytest tests/gpu/test_gpu_ocr.py -q --no-cov -m gpu

An unreachable engine FAILS these tests; they are never skipped. The photos are rendered here (large
black words on white, Pillow's bundled font), so nothing depends on a data set.
"""

from __future__ import annotations

import io
import json
import os
import re
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.domain import frame_engine_wire as wire


pytestmark = pytest.mark.gpu

DETECTOR = 'paddleocr_det_trt_uint8'
RECOGNISER = 'paddleocr_rec_trt_uint8'


def _spec(**over: Any) -> wire.OcrSpec:
    base: dict[str, Any] = {
        'detector': DETECTOR, 'recogniser': RECOGNISER, 'batch': 16, 'max_lines': 128,
        'det_side': 736, 'det_limit': 'min', 'det_max_side': 1920, 'det_antialias': False,
        'thresh': 0.2, 'box_thresh': 0.45, 'unclip_ratio': 1.4, 'taps': 1,
    }  # fmt: skip
    base.update(over)
    return wire.OcrSpec(**base)


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=95)
    return buf.getvalue()


def _words(words: list[str], *, size: int = 54, cols: int = 1, width: int = 520) -> Image.Image:
    font = ImageFont.load_default(size=size)
    rows = (len(words) + cols - 1) // cols
    img = Image.new('RGB', (cols * width, rows * (size + 40) + 40), 'white')
    draw = ImageDraw.Draw(img)
    for k, word in enumerate(words):
        draw.text(
            (30 + (k % cols) * width, 30 + (k // cols) * (size + 40)), word, fill='black', font=font
        )
    return img


@pytest.fixture(scope='module')
def engine() -> Any:
    from tritonclient.grpc import InferenceServerClient

    url = os.environ.get('OP_TEST_FRAME_ENGINE_GRPC')
    if not url:
        pytest.fail('GPU tier needs the frame engine gRPC address in OP_TEST_FRAME_ENGINE_GRPC')
    e = InferenceServerClient(url=url)
    try:
        ready = e.is_model_ready(wire.MODEL_NAME)
    except Exception as exc:
        pytest.fail(f'GPU tier needs a frame engine at {url}: {exc}')
    assert ready, f'the frame engine at {url} is not ready'
    return e


def _ocr(engine: Any, data: bytes, **over: Any) -> dict[str, Any]:
    from tritonclient.grpc import InferInput

    plan = wire.EnginePlan(branches=('ocr',), ocr=_spec(**over))
    inputs = []
    for name, value in ((wire.IMAGE, data), (wire.PLAN, plan.to_json().encode())):
        inp = InferInput(name, [1], 'BYTES')
        inp.set_data_from_numpy(np.array([value], dtype=np.object_))
        inputs.append(inp)
    result = engine.infer(wire.MODEL_NAME, inputs, client_timeout=120)
    out: dict[str, Any] = {n: result.as_numpy(n) for n in wire.BRANCH_OUTPUTS['ocr']}
    out['branch_status'] = json.loads(result.as_numpy('branch_status')[0])
    out['size'] = result.as_numpy('size')
    out['texts'] = [t.decode() if isinstance(t, bytes) else str(t) for t in out['ocr_texts']]
    return out


def _norm(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', s.lower())


def test_rendered_words_are_found_and_read(engine: Any) -> None:
    out = _ocr(engine, _jpeg(_words(['OPEN DAILY', 'Welcome', 'EXIT 24'])))
    assert out['branch_status'] == {'ocr': {'status': 'ok'}}
    got = {_norm(t) for t in out['texts']}
    assert {'opendaily', 'welcome', 'exit24'} <= got, out['texts']
    quads = out['ocr_quads']
    assert quads.shape == (len(out['texts']), 8)
    assert quads.dtype == np.float32
    w, h = out['size']
    assert quads[:, 0::2].max() <= w
    assert quads[:, 1::2].max() <= h
    assert np.all(out['ocr_scores'] > 0.5)
    assert np.all(out['ocr_char_min'] <= out['ocr_scores'] + 1e-6)
    assert out['ocr_overflow'].tolist() == [0]


def test_lines_come_back_top_to_bottom(engine: Any) -> None:
    out = _ocr(engine, _jpeg(_words(['FIRST', 'SECOND', 'THIRD'])))
    order = [_norm(t) for t in out['texts']]
    assert order == ['first', 'second', 'third'], order


def test_a_turned_page_is_read_through_the_quarter_turn_path(engine: Any) -> None:
    # clockwise-turned text (reads top to bottom), what PaddleOCR's quarter turn of a tall crop expects
    page = _words(['SIDEWAYS TEXT']).rotate(-90, expand=True)
    out = _ocr(engine, _jpeg(page))
    assert 'sidewaystext' in {_norm(t) for t in out['texts']}, out['texts']


def test_a_photo_with_no_text_answers_empty_and_ok(engine: Any) -> None:
    out = _ocr(engine, _jpeg(Image.new('RGB', (640, 480), 'white')))
    assert out['branch_status'] == {'ocr': {'status': 'ok'}}
    assert out['ocr_quads'].shape == (0, 8)
    assert out['texts'] == []
    assert out['ocr_overflow'].tolist() == [0]


def test_many_lines_are_capped_and_the_overflow_is_reported(engine: Any) -> None:
    words = [f'WORD{k:02d}' for k in range(40)]
    out = _ocr(engine, _jpeg(_words(words, size=40, cols=3)), max_lines=10)
    assert len(out['texts']) == 10
    assert out['ocr_overflow'][0] >= 10  # at least the rest of the 40 that were found


def test_taps_do_not_change_what_is_read_on_large_text(engine: Any) -> None:
    img = _jpeg(_words(['OPEN DAILY', 'Welcome'], size=120, width=1100))
    one = {_norm(t) for t in _ocr(engine, img, taps=1)['texts']}
    four = {_norm(t) for t in _ocr(engine, img, taps=4)['texts']}
    assert {'opendaily', 'welcome'} <= one
    assert {'opendaily', 'welcome'} <= four


def test_five_hundred_lines_stay_bounded_and_the_overflow_is_reported(engine: Any) -> None:
    words = [f'WORD{k:03d}' for k in range(500)]
    out = _ocr(engine, _jpeg(_words(words, size=36, cols=10, width=400)), max_lines=128)
    assert out['branch_status'] == {'ocr': {'status': 'ok'}}
    assert len(out['texts']) == 128
    assert out['ocr_quads'].shape == (128, 8)
    assert out['ocr_overflow'][0] > 0  # the rest of what the detector found is counted, not silent
    assert np.all(np.diff(out['ocr_quads'][:, 1]) > -40)  # still in reading order
