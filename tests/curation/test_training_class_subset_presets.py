"""``src.services.training.profiles.get_class_subset_presets``.

Presets served:

* ``all`` -- always served, generic.
* ``all_except_region`` / ``region_only`` -- only when the active region
  profile has a non-empty ``region_class_name``.
* whatever ``OP_TRAIN_PRESETS_PATH`` (a JSON list of the same shape) adds,
  appended after the built-ins. A malformed file raises.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    from pathlib import Path

from src.services.detection import profile_registry
from src.services.training.profiles import get_class_subset_presets


@pytest.fixture(autouse=True)
def _clean_region_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('OP_REGION_PROFILE', raising=False)
    monkeypatch.delenv('OP_REGION_PROFILE_PATH', raising=False)
    monkeypatch.delenv('OP_TRAIN_PRESETS_PATH', raising=False)
    import os

    for key in [
        k for k in os.environ if k.startswith(profile_registry.REGION_DETECTION_ENV_PREFIX)
    ]:
        monkeypatch.delenv(key)
    profile_registry._reset_registry_for_tests()
    yield
    profile_registry._reset_registry_for_tests()


def test_all_preset_is_always_served() -> None:
    presets = get_class_subset_presets()
    by_name = {p['name']: p for p in presets}
    assert by_name['all']['selector'] == {'kind': 'all'}


def test_no_region_presets_without_an_active_profile() -> None:
    presets = get_class_subset_presets()
    names = {p['name'] for p in presets}
    assert names == {'all'}


def test_no_region_presets_when_profile_has_no_region_class_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(f'{profile_registry.REGION_DETECTION_ENV_PREFIX}NAME', 'widget')
    profile_registry._reset_registry_for_tests()
    presets = get_class_subset_presets()
    names = {p['name'] for p in presets}
    assert names == {'all'}


def test_region_presets_appear_when_profile_names_a_region_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(f'{profile_registry.REGION_DETECTION_ENV_PREFIX}NAME', 'widget')
    monkeypatch.setenv(
        f'{profile_registry.REGION_DETECTION_ENV_PREFIX}REGION_CLASS_NAME', 'license_plate'
    )
    profile_registry._reset_registry_for_tests()

    presets = get_class_subset_presets()
    by_name = {p['name']: p for p in presets}
    assert set(by_name) == {'all', 'all_except_region', 'region_only'}
    assert by_name['all_except_region']['selector'] == {
        'kind': 'all_except',
        'names': ['license_plate'],
    }
    assert by_name['region_only']['selector'] == {'kind': 'names', 'names': ['license_plate']}
    assert by_name['region_only'].get('single_cls_default') is True


def test_op_train_presets_path_appends_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra: list[dict[str, Any]] = [
        {
            'name': 'custom',
            'label': 'Custom preset',
            'description': 'A deployment-specific preset.',
            'selector': {'kind': 'names', 'names': ['widget_a', 'widget_b']},
        }
    ]
    path = tmp_path / 'presets.json'
    path.write_text(json.dumps(extra))
    monkeypatch.setenv('OP_TRAIN_PRESETS_PATH', str(path))

    presets = get_class_subset_presets()
    names = {p['name'] for p in presets}
    assert names == {'all', 'custom'}


def test_op_train_presets_path_malformed_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / 'presets.json'
    path.write_text(json.dumps([{'name': 'bad'}]))  # missing required keys
    monkeypatch.setenv('OP_TRAIN_PRESETS_PATH', str(path))

    with pytest.raises(ValueError, match='bad'):
        get_class_subset_presets()


def test_op_train_presets_path_not_json_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / 'presets.json'
    path.write_text('not json')
    monkeypatch.setenv('OP_TRAIN_PRESETS_PATH', str(path))

    with pytest.raises(ValueError, match='JSON'):
        get_class_subset_presets()
