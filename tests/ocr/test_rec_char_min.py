"""``rec_char_min``: the lowest per-character probability of each recognised line.

Stand-in ``pb_utils`` (tests/_ocr_model_module.py), no Triton, no GPU. The new output sits
beside ``rec_scores`` (the mean); ``rec_scores`` and ``texts`` must not move.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from _ocr_model_module import FakeInferenceRequest, FakeTensor


CHARS = ['a', 'b', 'c']  # classes: 0 blank, 1..3 = a b c, 4 = space


def _dict_file(tmp_path):
    path = tmp_path / 'dict.txt'
    path.write_text('\n'.join(CHARS) + '\n', encoding='utf-8')
    return path


def _onehot(rows: list[tuple[int, float]]) -> np.ndarray:
    """[1, T, 5] probabilities: each row puts ``p`` on class ``idx``, the rest spread evenly."""
    out = np.zeros((1, len(rows), 5), dtype=np.float32)
    for t, (idx, p) in enumerate(rows):
        out[0, t, :] = (1.0 - p) / 4
        out[0, t, idx] = p
    return out


def _reference_decode(character: list[str], preds: np.ndarray) -> list[tuple[str, float]]:
    """The pre-char_min decoder, kept here as the byte-parity control."""
    results = []
    for idx, prob in zip(preds.argmax(axis=2), preds.max(axis=2), strict=True):
        text, conf_list, prev = '', [], 0
        for ci, cp in zip(idx, prob, strict=True):
            if ci not in (0, prev) and ci < len(character):
                text += character[ci]
                conf_list.append(cp)
            prev = ci
        results.append((text, float(np.mean(conf_list)) if conf_list else 0.0))
    return results


def test_char_min_is_the_minimum_over_decoded_characters(model_module, tmp_path):
    from ocr_postprocess import CTCLabelDecode

    dec = CTCLabelDecode(_dict_file(tmp_path))
    # a(0.9) a(0.5, repeat of the same char: not decoded) blank(0.99) b(0.7) c(0.8)
    preds = _onehot([(1, 0.9), (1, 0.5), (0, 0.99), (2, 0.7), (3, 0.8)])
    [(text, conf, char_min)] = dec(preds)
    assert text == 'abc'
    assert char_min == pytest.approx(0.7)
    assert conf == pytest.approx((0.9 + 0.7 + 0.8) / 3)


def test_char_min_is_zero_for_a_line_with_no_decoded_characters(model_module, tmp_path):
    from ocr_postprocess import CTCLabelDecode

    dec = CTCLabelDecode(_dict_file(tmp_path))
    [(text, conf, char_min)] = dec(_onehot([(0, 0.9), (0, 0.8)]))
    assert (text, conf, char_min) == ('', 0.0, 0.0)


def test_texts_and_rec_scores_are_byte_identical_to_the_old_decoder(model_module, tmp_path):
    from ocr_postprocess import CTCLabelDecode

    dec = CTCLabelDecode(_dict_file(tmp_path))
    rng = np.random.default_rng(7)
    preds = rng.random((6, 12, 5)).astype(np.float32)
    preds /= preds.sum(axis=2, keepdims=True)
    new = [(t, c) for t, c, _ in dec(preds)]
    assert new == _reference_decode(dec.character, preds)
    assert [np.float32(c).tobytes() for _, c in new] == [
        np.float32(c).tobytes() for _, c in _reference_decode(dec.character, preds)
    ]


def _recognizer(model_module, decoded):
    obj: Any = model_module.TritonPythonModel.__new__(model_module.TritonPythonModel)
    obj.rec_height = model_module.REC_HEIGHT
    obj.max_texts = model_module.MAX_TEXTS
    obj.ctc_decode = lambda _o: decoded
    obj._triton_to_numpy = lambda tensor: tensor.array
    return obj


def test_call_recognition_returns_char_min_per_crop(model_module):
    def ok(_shape):
        r = MagicMock()
        r.has_error.return_value = False
        r.array = np.zeros((1, 1, 4), dtype=np.float32)
        return r

    FakeInferenceRequest.response_factory = ok
    rec = _recognizer(model_module, [('abc', 0.8, 0.7)])
    crops = [np.zeros((48, 100, 3), dtype=np.uint8)] * 2
    texts, scores, char_mins = rec._call_recognition(crops)
    assert (texts, scores, char_mins) == (['abc', 'abc'], [0.8, 0.8], [0.7, 0.7])


def test_failed_recognition_uses_the_minus_one_sentinel_for_char_min(model_module):
    def boom(_shape):
        r = MagicMock()
        r.has_error.return_value = True
        r.error.return_value.message.return_value = 'down'
        return r

    FakeInferenceRequest.response_factory = boom
    rec = _recognizer(model_module, [])
    texts, scores, char_mins = rec._call_recognition([np.zeros((48, 100, 3), dtype=np.uint8)])
    assert (texts, scores, char_mins) == ([''], [-1.0], [-1.0])


def test_response_carries_rec_char_min_padded_with_zero(model_module, monkeypatch):
    pb = model_module.pb_utils
    monkeypatch.setattr(
        pb,
        'InferenceResponse',
        lambda output_tensors: {t.name: t for t in output_tensors},
        raising=False,
    )
    rec = _recognizer(model_module, [])
    boxes = np.array([[[0, 0], [10, 0], [10, 5], [0, 5]]], dtype=np.float32)
    out = rec._create_response(boxes, np.array([0.9]), ['abc'], [0.8], [0.7], 100, 100)
    tensor: FakeTensor = out['rec_char_min']
    assert tensor.array.dtype == np.float32
    assert tensor.array.shape == (128,)
    assert tensor.array[0] == pytest.approx(0.7)
    assert not tensor.array[1:].any()
    assert out['rec_scores'].array[0] == pytest.approx(0.8)
    assert out['texts'].array[0] == b'abc'


class _Result:
    def __init__(self, table: dict[str, np.ndarray]) -> None:
        self.table = table

    def as_numpy(self, name: str):
        return self.table.get(name)


def _bls_table(with_char_min: bool) -> dict[str, np.ndarray]:
    table = {
        'num_texts': np.array([2], dtype=np.int32),
        'text_boxes_normalized': np.array([[0.1, 0.1, 0.5, 0.3], [0.1, 0.4, 0.5, 0.6]], np.float32),
        'texts': np.array([b'ab12', b'cd34'], dtype=object),
        'text_scores': np.array([0.9, 0.9], dtype=np.float32),
        'rec_scores': np.array([0.8, 0.9], dtype=np.float32),
    }
    if with_char_min:
        table['rec_char_min'] = np.array([0.35, 0.6], dtype=np.float32)
    return table


def test_cascade_reader_sets_ocr_line_char_min() -> None:
    from src.services.detection.cascade_detect.ocr_recognizer import _parse_ocr_pipeline_result

    lines = _parse_ocr_pipeline_result(_Result(_bls_table(True)))
    assert [ln.char_min for ln in lines] == [pytest.approx(0.35), pytest.approx(0.6)]
    assert [ln.score for ln in lines] == [pytest.approx(0.8), pytest.approx(0.9)]


def test_cascade_reader_leaves_char_min_unset_when_the_pipeline_omits_it() -> None:
    from src.services.detection.cascade_detect.ocr_recognizer import _parse_ocr_pipeline_result

    lines = _parse_ocr_pipeline_result(_Result(_bls_table(False)))
    assert [ln.char_min for ln in lines] == [None, None]


def test_unframed_line_keeps_char_min() -> None:
    from src.services.detection.cascade_detect.ocr_recognizer import _unframe_line
    from src.services.detection.region_text import OcrLine

    line = OcrLine(text='ab', box=(0.2, 0.2, 0.8, 0.8), score=0.9, char_min=0.4)
    out = _unframe_line(line, (100, 100), (0, 0, 100, 100))
    assert out is not None
    assert out.char_min == pytest.approx(0.4)
