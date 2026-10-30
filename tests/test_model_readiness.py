"""Which Triton models /health requires, and how it judges them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services import model_readiness as mr


if TYPE_CHECKING:
    import pytest


def _entry(name: str, state: str, reason: str = '', version: str = '1') -> dict[str, str]:
    return {'name': name, 'version': version, 'state': state, 'reason': reason}


def test_default_required_models_cover_core_and_pe_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_REQUIRED_MODELS', raising=False)
    required = mr.required_model_names()
    assert 'yolov11_small_trt_end2end' in required
    assert 'scrfd_10g_bnkps' in required
    assert 'arcface_w600k_r50' in required
    assert 'mobileclip2_s2_image_encoder' in required
    assert 'pe_image_encoder' in required
    # OCR is skipped by the minimal profile, so it is not required by default.
    assert 'paddleocr_det_trt' not in required


def test_env_overrides_the_required_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_REQUIRED_MODELS', ' a_model , b_model ,')
    assert mr.required_model_names() == ['a_model', 'b_model']


def test_blank_env_means_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_REQUIRED_MODELS', '')
    assert 'pe_image_encoder' in mr.required_model_names()


def test_all_ready_reports_no_failures() -> None:
    index = [_entry('a', 'READY'), _entry('b', 'READY')]
    assert mr.models_not_ready(index, ['a', 'b']) == []


def test_unavailable_model_is_named_with_state_and_reason() -> None:
    index = [_entry('a', 'READY'), _entry('b', 'UNAVAILABLE', 'unable to load plan file')]
    bad = mr.models_not_ready(index, ['a', 'b'])
    assert bad == [{'name': 'b', 'state': 'UNAVAILABLE', 'reason': 'unable to load plan file'}]


def test_model_missing_from_index_is_not_ready() -> None:
    bad = mr.models_not_ready([_entry('a', 'READY')], ['a', 'ghost'])
    assert [b['name'] for b in bad] == ['ghost']
    assert bad[0]['state'] == 'MISSING'


def test_one_ready_version_is_enough() -> None:
    index = [_entry('a', 'UNAVAILABLE', version='1'), _entry('a', 'READY', version='2')]
    assert mr.models_not_ready(index, ['a']) == []


def test_models_outside_the_required_list_are_ignored() -> None:
    index = [_entry('a', 'READY'), _entry('ocr', 'UNAVAILABLE')]
    assert mr.models_not_ready(index, ['a']) == []
