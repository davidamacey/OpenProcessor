"""``frame_engine/ocr/ctc.py``: the collapse of the recogniser's per-step indices and scores.

The oracle is the host decode the Triton ``ocr_pipeline`` ran on the full probability tensor
(``models/ocr_pipeline/1/ocr_postprocess.py`` ``CTCLabelDecode``): same text, same mean and
lowest character probability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from frame_engine.ocr.ctc import CtcDecoder


ROOT = Path(__file__).resolve().parents[2]
DICT = ROOT / 'models' / 'paddleocr_rec_trt' / 'ppocrv6_dict.txt'
sys.path.insert(0, str(ROOT / 'models' / 'ocr_pipeline' / '1'))
from ocr_postprocess import CTCLabelDecode


def _probs(rng: np.random.Generator, b: int, t: int, c: int) -> np.ndarray:
    logits = rng.normal(size=(b, t, c)) * 3
    logits[:, :, 0] += 2.5  # the blank wins often, as it does on real lines
    e = np.exp(logits - logits.max(axis=2, keepdims=True))
    return (e / e.sum(axis=2, keepdims=True)).astype(np.float32)


def test_decode_equals_the_host_decode_of_the_probabilities() -> None:
    old = CTCLabelDecode(DICT)
    new = CtcDecoder.from_file(DICT)
    probs = _probs(np.random.default_rng(1), 6, 40, len(new.characters))
    expect = old(probs)
    idx, score = probs.argmax(axis=2).astype(np.int32), probs.max(axis=2)
    for k in range(6):
        text, mean, low = new.decode(idx[k], score[k])
        assert text == expect[k][0]
        assert mean == pytest.approx(expect[k][1], abs=1e-6)
        assert low == pytest.approx(expect[k][2], abs=1e-6)


def test_blank_and_repeats_collapse_and_a_gap_allows_a_repeat() -> None:
    dec = CtcDecoder(['blank', 'a', 'b', ' '])
    text, mean, low = dec.decode(
        np.array([0, 1, 1, 0, 1, 2, 2, 0], np.int32),
        np.array([0.9, 0.8, 0.7, 0.9, 0.6, 0.5, 0.4, 0.9], np.float32),
    )
    assert text == 'aab'
    assert mean == pytest.approx((0.8 + 0.6 + 0.5) / 3)
    assert low == pytest.approx(0.5)


def test_a_line_of_blanks_is_empty_with_zero_scores() -> None:
    dec = CtcDecoder(['blank', 'a'])
    assert dec.decode(np.zeros(5, np.int32), np.ones(5, np.float32)) == ('', 0.0, 0.0)


def test_an_index_past_the_dictionary_adds_no_character() -> None:
    dec = CtcDecoder(['blank', 'a'])
    text, _, _ = dec.decode(np.array([1, 7, 1], np.int32), np.ones(3, np.float32))
    assert (
        text == 'aa'
    )  # the skipped step still counts as the previous index, as in the host decode
