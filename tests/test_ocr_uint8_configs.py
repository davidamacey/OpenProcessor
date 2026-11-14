"""The shipped ``config.pbtxt`` of the frame engine's two OCR builds are what the export scripts write
(``--uint8``), and name the tensors the ``ocr`` branch reads (``frame_engine/branches/ocr.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import export_paddleocr_det as det
import export_paddleocr_rec as rec

from frame_engine.branches import ocr as branch


def _shipped(model: str) -> str:
    return (ROOT / 'models' / model / 'config.pbtxt').read_text()


def test_the_detector_config_is_what_the_exporter_writes(tmp_path: Path) -> None:
    det.create_triton_config(tmp_path, 4, uint8=True)
    assert _shipped(det.UINT8_MODEL_NAME) == (tmp_path / 'config.pbtxt').read_text()


def test_the_recogniser_config_is_what_the_exporter_writes(tmp_path: Path) -> None:
    plan = tmp_path / '1' / 'model.plan'
    plan.parent.mkdir()
    rec.create_triton_config(plan, uint8=True)
    assert _shipped(rec.UINT8_MODEL_NAME) == (tmp_path / 'config.pbtxt').read_text()


@pytest.mark.parametrize(
    ('model', 'tensors'),
    [
        (det.UINT8_MODEL_NAME, [branch.DET_INPUT, branch.DET_OUTPUT]),
        (rec.UINT8_MODEL_NAME, [branch.REC_INPUT, branch.REC_INDICES, branch.REC_SCORES]),
    ],
)
def test_the_configs_name_the_tensors_the_branch_reads(model: str, tensors: list[str]) -> None:
    text = _shipped(model)
    assert f'name: "{model}"' in text
    assert all(f'name: "{t}"' in text for t in tensors)
    assert 'data_type: TYPE_UINT8' in text  # the engine sends uint8 pixels


def test_the_recogniser_profile_reaches_the_branch_width_cap() -> None:
    assert rec.MAX_WIDTH == branch.REC_MAX_WIDTH


def test_the_detector_profile_reaches_the_plan_default_cap() -> None:
    from src.domain import frame_engine_wire as wire

    spec = wire.parse_plan(
        '{"version": 1, "branches": ["ocr"], "ocr": {"detector": "d", "recogniser": "r",'
        ' "batch": 1, "max_lines": 1, "det_side": 32}}',
        has_boxes=False,
    ).ocr
    assert spec is not None
    assert spec.det_max_side <= det.UINT8_MAX_RESOLUTION
