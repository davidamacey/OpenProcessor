"""PP-OCRv6 swap (D-82): the dictionary, class count, model files and BLS constants agree."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DICT = ROOT / 'models/paddleocr_rec_trt/ppocrv6_dict.txt'
# CTCLabelDecode adds a blank at the front and a space at the end of the dictionary.
OCRV6_CLASSES = 18708 + 2
CLASS_COUNT_FILES = [
    'models/paddleocr_rec_trt/config.pbtxt',
    'scripts/lib/config_models.sh',
    'scripts/lib/paddleocr_config.sh',
]


def test_dictionary_matches_model_class_count() -> None:
    assert len(DICT.read_text(encoding='utf-8').split('\n')) - 1 + 2 == OCRV6_CLASSES


def test_old_dictionary_is_gone() -> None:
    assert not (ROOT / 'models/paddleocr_rec_trt/ppocrv5_dict.txt').exists()


@pytest.mark.parametrize('rel', CLASS_COUNT_FILES)
def test_config_declares_ocrv6_class_count(rel: str) -> None:
    text = (ROOT / rel).read_text(encoding='utf-8')
    assert f'dims: [ -1, {OCRV6_CLASSES} ]' in text
    assert '18385' not in text


def test_export_script_class_count() -> None:
    text = (ROOT / 'export/export_paddleocr_rec.py').read_text(encoding='utf-8')
    assert re.search(rf'^NUM_CHARS = {OCRV6_CLASSES}\b', text, re.M)


def test_bls_loads_ocrv6_dictionary_and_det_normaliser() -> None:
    text = (ROOT / 'models/ocr_pipeline/1/model.py').read_text(encoding='utf-8')
    assert 'ppocrv6_dict.txt' in text
    assert 'normalize_det_image(ocr_image)' in text


def test_model_version_bumped() -> None:
    assert (ROOT / 'models/ocr_pipeline/MODEL_VERSION').read_text().strip() == '2'


@pytest.mark.parametrize(
    'rel',
    ['scripts/lib/model_groups.sh', 'tests/installer/test_engine_stamps.py'],
)
def test_onnx_names_are_ocrv6(rel: str) -> None:
    text = (ROOT / rel).read_text(encoding='utf-8')
    assert 'ppocrv6_rec_small.onnx' in text
    assert 'ppocrv6_det_small.onnx' in text
    assert '_v5_mobile' not in text


def test_det_normalisation_values() -> None:
    import sys

    import numpy as np

    sys.path.insert(0, str(ROOT / 'models/ocr_pipeline/1'))
    from ocr_postprocess import normalize_det_image

    img = np.zeros((3, 2, 2), dtype=np.uint8)
    img[0] = 255  # channel 0 is B in the BGR array
    out = normalize_det_image(img)
    assert out.dtype == np.float32
    assert out[0, 0, 0] == pytest.approx((1 - 0.485) / 0.229)
    assert out[1, 0, 0] == pytest.approx((0 - 0.456) / 0.224)
    assert out[2, 0, 0] == pytest.approx((0 - 0.406) / 0.225)
