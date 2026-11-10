"""PT-F1: a reading writes the match keys, the frame's own reading and the reader fields."""

from __future__ import annotations

import dataclasses

from src.domain.region_boxes import RegionBox
from src.services.detection.region_text import (
    DominantTextConfig,
    OcrLine,
    Reader,
    read_dominant_text,
    resolve_region_text,
)


CFG = DominantTextConfig()
READER = Reader('paddleocr_rec_trt', '1')
CONFUSABLES = {'O': '0', 'I': '1'}


def _ocr(text: str, char_min: float | None):
    return read_dominant_text([OcrLine(text, (0.1, 0.3, 0.9, 0.7), 0.91, 1.0, char_min)], CFG)


def _resolve(ocr):
    return resolve_region_text(
        'ocr',
        vlm_text=None,
        vlm_confidence=None,
        vlm_engine='',
        ocr=ocr,
        ocr_reader=READER,
        normalizer=CFG.normalizer,
        confusables=CONFUSABLES,
    )


def test_a_reading_writes_text_norm_and_text_fold_from_text() -> None:
    out = _resolve(_ocr('abo-1i2-xyz-987', 0.8))

    assert out['text'] == 'abo-1i2-xyz-987'
    assert out['text_frame'] == 'abo-1i2-xyz-987'
    assert out['text_norm'] == 'ABO1I2XYZ987'
    assert out['text_fold'] == 'AB0112XYZ987'
    assert (out['text_reader'], out['text_reader_version']) == ('paddleocr_rec_trt', '1')
    assert out['text_conf_char_min'] == 0.8


def test_a_missing_char_min_is_not_written() -> None:
    out = _resolve(_ocr('abo-1i2-xyz-987', None))

    assert 'text_conf_char_min' not in out


def test_text_engine_version_is_gone() -> None:
    names = {f.name for f in dataclasses.fields(RegionBox)}

    assert 'text_engine_version' not in names
    assert {'text_frame', 'text_norm', 'text_fold', 'text_reader', 'text_reader_version'} <= names
    assert 'text_engine_version' not in _resolve(_ocr('abo-1i2-xyz-987', 0.8))
