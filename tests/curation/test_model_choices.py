"""``build_model_choices`` (W9.8, §7.8.4): every model a user can or cannot
choose, one uniform row each, built from the live source it describes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pytest


from curation.conftest import SCOPED
from src.config.settings import TritonModelConfig
from src.services.curation.model_choices import build_model_choices
from src.services.labeling.vlm_catalog import load_catalog


ROLES = [
    'detect_model',
    'item_detector',
    'secondary_classifier',
    'region_detector',
    'region_segmenter',
    'region_ocr',
    'vlm',
    'local_vlm_model',
    'item_embedding',
    'search_embedding',
    'face_models',
    'training_size',
    'bakeoff_models',
]


def _rows(**kw: Any) -> dict[str, dict[str, Any]]:
    args = {
        'detector_ids': ['det_a', 'det_b'],
        'ocr_pipeline_ids': ['ocr_x'],
        'promoted_ids': ['prom_1'],
    }
    args.update(kw)
    return {row['role']: row for row in build_model_choices(**args)}


def test_every_role_appears_once_in_a_stable_order() -> None:
    table = build_model_choices(detector_ids=[], ocr_pipeline_ids=[], promoted_ids=[])
    assert [row['role'] for row in table] == ROLES


def test_every_row_has_the_same_shape_and_says_how_or_why_not() -> None:
    for row in _rows().values():
        assert set(row) == {
            'role',
            'label',
            'scope',
            'current',
            'dims',
            'choices',
            'settable',
            'settable_via',
            'reason',
        }
        assert row['label']
        assert row['scope'] in {
            'per_request',
            'config_store',
            'region_profile',
            'deployment',
            'per_run',
        }
        for choice in row['choices']:
            assert choice == {'id': choice['id'], 'label': choice['id']}
        if row['settable']:
            assert row['settable_via'], row['role']
        else:
            # a fixed model says why, so the UI never shows a dead control
            assert row['reason'] or row['settable_via'], row['role']


def test_choices_come_from_the_lists_the_caller_read_live() -> None:
    rows = _rows(detector_ids=['only_this'], ocr_pipeline_ids=['ocr_y'], promoted_ids=['p1', 'p2'])
    assert [c['id'] for c in rows['detect_model']['choices']] == ['only_this']
    assert [c['id'] for c in rows['region_detector']['choices']] == ['only_this']
    assert [c['id'] for c in rows['region_ocr']['choices']] == ['ocr_y']
    assert [c['id'] for c in rows['bakeoff_models']['choices']] == ['p1', 'p2']


def test_the_current_model_is_read_not_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TritonModelConfig, 'YOLO_MODEL', 'a_different_detector')
    assert _rows()['detect_model']['current'] == 'a_different_detector'
    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'my_ingest_detector')
    assert _rows()['item_detector']['current'] == 'my_ingest_detector'


def test_the_region_rows_follow_the_active_profile(reference_region_profile: None) -> None:
    from src.services.detection.profile_registry import get_active_region_profile

    profile = get_active_region_profile()
    assert profile is not None
    rows = _rows()
    assert rows['region_detector']['current'] == profile.detector_model
    assert rows['region_segmenter']['current'] == profile.segmenter_name
    assert rows['region_segmenter']['settable'] is False
    assert rows['region_ocr']['current'] == profile.ocr_pipeline_model


def test_without_a_region_profile_the_region_rows_have_no_current_model() -> None:
    rows = _rows()
    assert rows['region_detector']['current'] is None
    assert rows['region_ocr']['current'] is None


def test_embeddings_are_fixed_and_carry_their_dimensions() -> None:
    rows = _rows()
    for role in ('item_embedding', 'search_embedding'):
        assert rows[role]['settable'] is False
        assert isinstance(rows[role]['dims'], int)
        assert rows[role]['dims'] > 0
        assert 're-index' in rows[role]['reason']
    assert rows['item_embedding']['dims'] != rows['search_embedding']['dims']
    assert rows['face_models']['settable'] is False


def test_the_vlm_rows_reflect_the_registry_and_the_local_stack(
    vlm_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    vlm_api.ready('alpha')
    vlm_api.ready('beta')
    assert vlm_api.activate('beta', expected_active=None).status_code == 200
    body = vlm_api.client.get(f'{SCOPED}/config/vocabulary').json()
    by_role = {row['role']: row for row in body['model_choices']}
    assert [row['role'] for row in body['model_choices']] == ROLES
    assert by_role['vlm']['current'] == 'beta'
    assert [c['id'] for c in by_role['vlm']['choices']] == ['alpha', 'beta']
    assert by_role['vlm']['settable'] is True
    # no in-compose vlm here: the local model is not settable, and says why
    assert by_role['local_vlm_model']['settable'] is False
    assert by_role['local_vlm_model']['reason']
    assert load_catalog()[0].id in [c['id'] for c in by_role['local_vlm_model']['choices']]

    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'local-vlm')
    monkeypatch.setenv('OP_LOCAL_VLM_ENDPOINT', 'env')
    body = vlm_api.client.get(f'{SCOPED}/config/vocabulary').json()
    local = {row['role']: row for row in body['model_choices']}['local_vlm_model']
    assert local['settable'] is True
    assert local['reason'] is None


def test_the_served_vocabulary_matches_the_builder(vlm_api) -> None:
    served = vlm_api.client.get(f'{SCOPED}/config/vocabulary').json()['model_choices']
    assert [row['role'] for row in served] == ROLES
    for row in served:
        assert set(row) == {
            'role',
            'label',
            'scope',
            'current',
            'dims',
            'choices',
            'settable',
            'settable_via',
            'reason',
        }
