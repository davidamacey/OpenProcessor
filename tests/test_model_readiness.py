"""Which Triton models /health requires, and how it judges them."""

from __future__ import annotations

import pytest

from src.services.models import model_readiness as mr


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
    assert 'pe_crop_encoder' in required
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


FACE_MODELS = {
    'scrfd_10g_bnkps',
    'arcface_w600k_r50',
    'scrfd_10g_bnkps_uint8',
    'arcface_w600k_r50_uint8',
}


def test_faces_off_drops_the_face_models_from_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_REQUIRED_MODELS', raising=False)
    monkeypatch.setenv('OP_FACES_ENABLED', 'false')
    required = mr.required_model_names()
    assert not FACE_MODELS & set(required)
    assert 'yolov11_small_trt_end2end' in required


def test_faces_off_drops_the_face_models_from_an_explicit_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_FACES_ENABLED', 'false')
    monkeypatch.setenv(
        'OP_REQUIRED_MODELS',
        'yolov11_small_trt_end2end,scrfd_10g_bnkps,arcface_w600k_r50,'
        'scrfd_10g_bnkps_uint8,arcface_w600k_r50_uint8',
    )
    assert mr.required_model_names() == ['yolov11_small_trt_end2end']


# ---- the crop encoder's built size is checked against the configured one ----


def _config(dims: list[object], name: str = 'images') -> dict[str, object]:
    """``GET /v2/models/<m>/config``: protobuf JSON, so int64 dims arrive as strings."""
    return {
        'name': 'pe_crop_encoder',
        'max_batch_size': 32,
        'input': [{'name': name, 'dims': dims}],
    }


def test_a_crop_engine_built_at_the_configured_size_passes() -> None:
    assert mr.crop_input_problem(_config(['3', '224', '224']), 'pe_crop_encoder', 224) is None
    assert mr.crop_input_problem(_config([3, 224, 224]), 'pe_crop_encoder', 224) is None


def test_a_crop_engine_built_at_another_size_is_named_with_both_sizes() -> None:
    why = mr.crop_input_problem(_config(['3', '336', '336']), 'pe_crop_encoder', 224)
    assert why is not None
    for part in ('pe_crop_encoder', '336', '224', 'PE_CROP_SIZE'):
        assert part in why


def test_the_rebuild_hint_is_a_command_that_exists() -> None:
    why = mr.crop_input_problem(_config(['3', '336', '336']), 'pe_crop_encoder', 224)
    assert why is not None
    assert './openprocessor models install --only pe' in why
    assert 'make pe-build-crop-trt' not in why


@pytest.mark.parametrize(
    'config',
    [None, {}, {'input': []}, _config(['3', '224', '224'], name='pixels'), _config(['3', '224'])],
)
def test_a_config_that_cannot_be_read_is_a_problem_not_a_pass(config: object) -> None:
    assert mr.crop_input_problem(config, 'pe_crop_encoder', 224) is not None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_readiness_refuses_a_crop_engine_of_the_wrong_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers import health
    from src.services.models import triton_control

    monkeypatch.setenv('OP_REQUIRED_MODELS', 'pe_crop_encoder')

    async def index(_self: object) -> list[dict[str, str]]:
        return [_entry('pe_crop_encoder', 'READY')]

    async def config(_self: object, _name: str) -> dict[str, object]:
        return _config(['3', '336', '336'])

    monkeypatch.setattr(triton_control.TritonControlService, 'get_repository_index', index)
    monkeypatch.setattr(triton_control.TritonControlService, 'get_model_config', config)
    ok, detail, bad = await health._probe_required_models()
    assert ok is False
    assert 'PE_CROP_SIZE' in detail
    assert bad[0]['name'] == 'pe_crop_encoder'


@pytest.mark.asyncio
async def test_readiness_passes_a_crop_engine_of_the_configured_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers import health
    from src.services.models import triton_control

    monkeypatch.setenv('OP_REQUIRED_MODELS', 'pe_crop_encoder')

    async def index(_self: object) -> list[dict[str, str]]:
        return [_entry('pe_crop_encoder', 'READY')]

    async def config(_self: object, _name: str) -> dict[str, object]:
        return _config(['3', '224', '224'])

    monkeypatch.setattr(triton_control.TritonControlService, 'get_repository_index', index)
    monkeypatch.setattr(triton_control.TritonControlService, 'get_model_config', config)
    assert (await health._probe_required_models())[0] is True
