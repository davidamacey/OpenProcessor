"""``text_reader=vlm`` validation asks the endpoint registry whether a VLM is
configured (W9 review M6), not ``OP_VLM_URL``."""

from __future__ import annotations

import pytest

from curation.conftest import SCOPED
from curation.test_profile_validation import _body
from src.services.triton_control import TritonControlService


@pytest.fixture(autouse=True)
def _no_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    async def listing(_self):
        return []

    monkeypatch.setattr(TritonControlService, 'get_repository_index', listing)


def _validate(vlm_api) -> dict:
    body = _body(
        text_reader='vlm',
        ocr_pipeline_model='',
        ocr_rec_model='',
        detector_model='',
        segmenter_text_prompt='thing',
    )
    response = vlm_api.client.post(
        f'{SCOPED}/region_profiles/validate', json={'name': None, 'body': body}
    )
    assert response.status_code == 200, response.text
    return response.json()


def _codes(report: dict) -> set[str]:
    return {i['code'] for i in [*report['errors'], *report['warnings']]}


def test_a_stored_active_endpoint_counts_as_a_configured_vlm(vlm_api) -> None:
    vlm_api.ready('alpha')
    assert vlm_api.activate('alpha', expected_active=None).status_code == 200
    codes = _codes(_validate(vlm_api))
    assert 'vlm_not_configured' not in codes
    assert 'triton_unreachable' not in codes


def test_the_env_url_does_not_count_once_the_project_switched_the_vlm_off(
    vlm_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'served')
    assert 'vlm_not_configured' not in _codes(_validate(vlm_api))
    off = vlm_api.client.post(f'{SCOPED}/vlm/endpoints/deactivate', json={'expected_active': None})
    assert off.status_code == 200, off.text
    assert 'vlm_not_configured' in _codes(_validate(vlm_api))
