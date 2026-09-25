"""Tests for `DELETE /curation/models/{model_name}`.

Covers the router-level guardrails: region-detector pipeline models
(driven by the active ``DetectionProfile``) can never be unloaded (even
with `force=true`), core pipeline models (CLIP image encoder, face
detect/recognition) require `force=true`, and a plain throwaway model
unloads with no flag needed.

We mount just the shared curation router (models.py registers its
routes onto `src.routers.curation._common.router`) and monkeypatch
`unload_triton_model` at the point models.py imported it, so no real
Triton HTTP call or filesystem mutation happens here — that's covered
by `test_triton_promote.py`'s disposable-tmp_path unit tests instead.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.training.triton_promote import ModelNotPromotedError, UnloadResult


@pytest.fixture
def app_client():
    """Mount models.py's routes."""
    import src.routers.curation.models as models_mod

    app = FastAPI()
    app.include_router(models_mod.router)
    with TestClient(app) as client:
        yield client


def _mock_unload(monkeypatch: pytest.MonkeyPatch, **overrides) -> AsyncMock:
    result = UnloadResult(
        triton_name=overrides.get('triton_name', 'op_smoke_v1'),
        triton_unloaded=True,
        directory_removed=True,
    )
    mock = AsyncMock(return_value=result)
    monkeypatch.setattr('src.routers.curation.models.unload_triton_model', mock)
    return mock


# =============================================================================
# Region-detector guard — hard block, no force override
# =============================================================================


@pytest.mark.parametrize('model_name', ['paddleocr_det_trt', 'paddleocr_rec_trt'])
def test_unload_refuses_region_protected_model(app_client, monkeypatch, model_name):
    mock = _mock_unload(monkeypatch)
    resp = app_client.delete(f'/curation/models/{model_name}')
    assert resp.status_code == 403
    mock.assert_not_awaited()


@pytest.mark.usefixtures('reference_region_profile')
def test_unload_refuses_active_profiles_detector_model(app_client, monkeypatch):
    """S8: the guard has no hardcoded 'lpr_' prefix. A model is protected
    because it IS the active profile's configured detector_model, not
    because of its name's shape."""
    mock = _mock_unload(monkeypatch)
    resp = app_client.delete('/curation/models/license_plate_detector')
    assert resp.status_code == 403
    mock.assert_not_awaited()


def test_unload_allows_an_unconfigured_lpr_shaped_name(app_client, monkeypatch):
    """No hardcoded 'lpr_' prefix guard: an lpr-shaped name that isn't the
    active profile's detector_model (here: no profile configured at all)
    unloads like any other throwaway model."""
    mock = _mock_unload(monkeypatch)
    resp = app_client.delete('/curation/models/lpr_some_future_model')
    assert resp.status_code == 200, resp.text
    mock.assert_awaited_once()


@pytest.mark.parametrize('model_name', ['paddleocr_det_trt'])
def test_unload_refuses_region_protected_model_even_with_force(app_client, monkeypatch, model_name):
    """force=true must NOT bypass the region-detector guard — this is the
    one guard in the whole endpoint with no override, per the "never
    touch the configured detection pipeline's models" constraint."""
    mock = _mock_unload(monkeypatch)
    resp = app_client.delete(f'/curation/models/{model_name}', params={'force': 'true'})
    assert resp.status_code == 403
    mock.assert_not_awaited()


@pytest.mark.usefixtures('reference_region_profile')
def test_unload_refuses_active_profiles_detector_model_even_with_force(app_client, monkeypatch):
    mock = _mock_unload(monkeypatch)
    resp = app_client.delete('/curation/models/license_plate_detector', params={'force': 'true'})
    assert resp.status_code == 403
    mock.assert_not_awaited()


# =============================================================================
# Core pipeline model guard — requires force=true
# =============================================================================


@pytest.mark.parametrize('model_name', ['mobileclip2_s2_image_encoder', 'scrfd_10g_bnkps'])
def test_unload_refuses_core_pipeline_model_without_force(app_client, monkeypatch, model_name):
    mock = _mock_unload(monkeypatch)
    resp = app_client.delete(f'/curation/models/{model_name}')
    assert resp.status_code == 409
    mock.assert_not_awaited()


def test_unload_allows_core_pipeline_model_with_force(app_client, monkeypatch):
    mock = _mock_unload(monkeypatch, triton_name='mobileclip2_s2_image_encoder')
    resp = app_client.delete(
        '/curation/models/mobileclip2_s2_image_encoder', params={'force': 'true'}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body['forced'] is True
    assert body['warning'] is not None
    mock.assert_awaited_once_with('mobileclip2_s2_image_encoder')


# =============================================================================
# Happy path — a throwaway promoted model, no guard, no force needed
# =============================================================================


def test_unload_happy_path_no_force_needed_for_throwaway_model(app_client, monkeypatch):
    mock = _mock_unload(monkeypatch, triton_name='op_vehicle_smoke_v1')
    resp = app_client.delete('/curation/models/op_vehicle_smoke_v1')
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        'triton_name': 'op_vehicle_smoke_v1',
        'triton_unloaded': True,
        'directory_removed': True,
        'forced': False,
        'warning': None,
    }
    mock.assert_awaited_once_with('op_vehicle_smoke_v1')


def test_unload_propagates_not_promoted_as_404(app_client, monkeypatch):
    monkeypatch.setattr(
        'src.routers.curation.models.unload_triton_model',
        AsyncMock(side_effect=ModelNotPromotedError('never_promoted')),
    )
    resp = app_client.delete('/curation/models/never_promoted')
    assert resp.status_code == 404


# =============================================================================
# _discover_promoted_models — /models/status surfacing
#
# Before this existed, /models/status only ever returned the fixed core
# models — a throwaway promote would be invisible to the UI entirely,
# requiring a shell into the host to clean up.
# =============================================================================


def test_discover_promoted_models_finds_a_promoted_extra(tmp_path):
    import src.routers.curation.models as models_mod

    model_dir = tmp_path / 'op_vehicle_smoke_v1'
    model_dir.mkdir()
    (model_dir / 'promote.json').write_text(
        '{"job_id": "job-abc", "version": "1", "promoted_at": "2026-09-11T00:00:00Z"}'
    )

    found = models_mod._discover_promoted_models(models_dir=tmp_path)
    assert found == [
        {
            'name': 'op_vehicle_smoke_v1',
            'job_id': 'job-abc',
            'version': '1',
            'promoted_at': '2026-09-11T00:00:00Z',
        }
    ]


def test_discover_promoted_models_skips_fixed_pipeline_models(tmp_path):
    import src.routers.curation.models as models_mod

    # A fixed pipeline model dir that happens to also carry a promote.json
    # (plausible if it was itself promoted through this pipeline once) must
    # not be double-counted — /models already lists it via _core_models().
    model_dir = tmp_path / 'mobileclip2_s2_image_encoder'
    model_dir.mkdir()
    (model_dir / 'promote.json').write_text('{"job_id": "job-xyz"}')

    found = models_mod._discover_promoted_models(models_dir=tmp_path)
    assert found == []


def test_discover_promoted_models_ignores_dirs_without_promote_json(tmp_path):
    import src.routers.curation.models as models_mod

    (tmp_path / 'yolov11_small_trt_end2end').mkdir()  # a real dir, no promote.json

    found = models_mod._discover_promoted_models(models_dir=tmp_path)
    assert found == []


def test_discover_promoted_models_missing_dir_returns_empty(tmp_path):
    import src.routers.curation.models as models_mod

    found = models_mod._discover_promoted_models(models_dir=tmp_path / 'does_not_exist')
    assert found == []


def test_discover_promoted_models_corrupt_promote_json_is_skipped_not_fatal(tmp_path):
    import src.routers.curation.models as models_mod

    model_dir = tmp_path / 'op_bad_promote_json'
    model_dir.mkdir()
    (model_dir / 'promote.json').write_text('{not valid json')

    found = models_mod._discover_promoted_models(models_dir=tmp_path)
    # Still discovered (the dir + promote.json presence is the trigger),
    # just with no metadata since the file couldn't be parsed.
    assert found == [
        {'name': 'op_bad_promote_json', 'job_id': None, 'version': None, 'promoted_at': None}
    ]


def test_is_region_protected_model_matches_names_and_prefixes():
    import src.routers.curation.models as models_mod

    # No hardcoded 'lpr_' prefix (S8): an lpr-shaped name is not protected
    # merely by its name, only via the active profile's detector_model or
    # the fixed paddleocr_ OCR prefix.
    assert models_mod._is_region_protected_model('lpr_nanov11_640') is False
    assert models_mod._is_region_protected_model('paddleocr_det_trt') is True
    assert models_mod._is_region_protected_model('op_vehicle_smoke_v1') is False


@pytest.mark.usefixtures('reference_region_profile')
def test_is_region_protected_model_matches_the_active_profiles_detector_model():
    import src.routers.curation.models as models_mod

    assert models_mod._is_region_protected_model('license_plate_detector') is True
    assert models_mod._is_region_protected_model('lpr_some_future_model') is False


def test_core_pipeline_models_includes_clip_and_face_models():
    import src.routers.curation.models as models_mod

    core = models_mod._core_pipeline_models()
    assert 'mobileclip2_s2_image_encoder' in core
    assert 'scrfd_10g_bnkps' in core
    assert 'lpr_nanov11_640' not in core
