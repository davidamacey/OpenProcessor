"""DB post-processing unclips with pyclipper, and a missing pyclipper fails loudly (GPU-2).

PaddleOCR's ``DBPostProcess.unclip`` offsets the polygon with pyclipper by
``area * unclip_ratio / perimeter``. A silent centre-scale fallback gave different boxes
(300x30 instead of 227x47 for a 200x20 line at ratio 1.5), so the dependency is mandatory:
declared for the Triton image that runs the python backend, and imported at module top level.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BLS_DIR = ROOT / 'models' / 'ocr_pipeline' / '1'


@pytest.fixture
def postprocess(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(BLS_DIR))
    monkeypatch.delitem(sys.modules, 'ocr_postprocess', raising=False)
    yield importlib.import_module('ocr_postprocess')
    sys.modules.pop('ocr_postprocess', None)


def _rect(w: float, h: float) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def _size(box: np.ndarray) -> tuple[float, float]:
    (_, _), (a, b), _ = cv2.minAreaRect(box.astype(np.float32))
    return max(a, b), min(a, b)


class TestUnclip:
    def test_line_grows_by_the_db_offset(self, postprocess) -> None:
        out = postprocess.DBPostProcess(unclip_ratio=1.5)._unclip(_rect(200, 20))
        w, h = _size(out)
        # d = area * ratio / perimeter = 4000 * 1.5 / 440 = 13.6 on every side.
        assert w == pytest.approx(227, abs=1.5)
        assert h == pytest.approx(47, abs=1.5)

    def test_square_grows_evenly(self, postprocess) -> None:
        # Control: for a square the DB offset and a centre scale agree closely.
        out = postprocess.DBPostProcess(unclip_ratio=1.5)._unclip(_rect(100, 100))
        w, h = _size(out)
        assert w == pytest.approx(h, abs=1.5)
        assert w == pytest.approx(100 + 2 * (100 * 100 * 1.5 / 400), abs=1.5)


class TestMissingDependency:
    def test_import_fails_loudly_without_pyclipper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.syspath_prepend(str(BLS_DIR))
        monkeypatch.delitem(sys.modules, 'ocr_postprocess', raising=False)
        monkeypatch.setitem(sys.modules, 'pyclipper', None)  # makes `import pyclipper` fail
        with pytest.raises(ImportError):
            importlib.import_module('ocr_postprocess')
        sys.modules.pop('ocr_postprocess', None)


class TestDeclaredRequirement:
    def test_triton_image_installs_pyclipper_pinned(self) -> None:
        # The BLS runs in the Triton python backend, i.e. in Dockerfile.triton's image.
        text = (ROOT / 'Dockerfile.triton').read_text()
        assert re.search(r'pip install[^\n]*\bpyclipper==\d+(\.\d+)+', text.replace('\\\n', ' '))

    def test_offline_tests_install_pyclipper(self) -> None:
        text = (ROOT / 'requirements-test.txt').read_text()
        assert re.search(r'^pyclipper\b', text, re.M)
