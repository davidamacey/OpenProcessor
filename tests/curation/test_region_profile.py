"""Neutral default region profile + env-driven selection.

An unconfigured deployment has *no* region profile: nothing is advertised
on ``GET /methods``' ``detection_profile`` axis and the detection worker's
region cascade stays off. ``OP_REGION_PROFILE_PATH`` loads a profile
from a file (e.g. ``examples/region_profiles/license_plate.json``);
``OP_REGION_PROFILE`` selects a profile a deployment's own startup code
already registered. ``OP_REGION_DETECTION_<FIELD>``
overrides fields on top of it. Whatever resolves is registered, so it is
exactly what ``GET /methods`` advertises.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from _region_profile_fixture import (
    EXAMPLE_LICENSE_PLATE_PROFILE as REFERENCE_LICENSE_PLATE_PROFILE,
    EXAMPLE_LICENSE_PLATE_PROFILE_PATH,
)

import scripts.curation.sam_worker_main as worker
from src.config import DetectionProfile
from src.services.detection import profile_registry


if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator


_ENV_PREFIX = profile_registry.REGION_DETECTION_ENV_PREFIX


@pytest.fixture
def region_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """A clean region-profile environment + registry for each test."""
    import os

    monkeypatch.delenv('OP_REGION_PROFILE', raising=False)
    monkeypatch.delenv('OP_REGION_PROFILE_PATH', raising=False)
    for key in [k for k in os.environ if k.startswith(_ENV_PREFIX)]:
        monkeypatch.delenv(key)
    profile_registry._reset_registry_for_tests()
    yield monkeypatch
    profile_registry._reset_registry_for_tests()


# =============================================================================
# Registry resolution
# =============================================================================


def test_unconfigured_default_is_neutral(region_env: pytest.MonkeyPatch) -> None:
    assert profile_registry.get_active_region_profile() is None
    assert profile_registry.get_default_profile_name() is None
    assert profile_registry.get_profiles() == {}


def test_importing_cascade_detect_registers_nothing_by_default(
    region_env: pytest.MonkeyPatch,
) -> None:
    """The reference plate profile used to self-register as the default on
    import; it must now stay an unregistered, selectable example."""
    import os
    import subprocess  # nosec B404 - this repo's interpreter on a fixed snippet
    import sys

    snippet = (
        'import src.services.detection.cascade_detect\n'
        'from src.services.detection import profile_registry as r\n'
        'print(sorted(r._REGISTRY), r._DEFAULT_NAME)\n'
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ('OP_REGION_PROFILE', 'OP_REGION_PROFILE_PATH')
        and not k.startswith(_ENV_PREFIX)
    }
    out = subprocess.run(  # nosec B603
        [sys.executable, '-c', snippet],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == '[] None'


def test_region_profile_or_neutral_has_no_detector(region_env: pytest.MonkeyPatch) -> None:
    neutral = profile_registry.region_profile_or_neutral()
    assert neutral.name == 'region'
    assert neutral.detector_model == ''
    assert neutral.sam_text_prompt == ''
    assert neutral.secondary_shape_groups == frozenset()


def test_select_builtin_reference_profile_by_name(region_env: pytest.MonkeyPatch) -> None:
    region_env.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    active = profile_registry.get_active_region_profile()
    assert active == REFERENCE_LICENSE_PLATE_PROFILE
    assert profile_registry.get_default_profile_name() == 'license_plate'
    assert set(profile_registry.get_profiles()) == {'license_plate'}


def test_env_overrides_layer_on_selected_profile(region_env: pytest.MonkeyPatch) -> None:
    region_env.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    region_env.setenv(f'{_ENV_PREFIX}SECONDARY_SHAPE_GROUPS', 'group_a, group_b')
    region_env.setenv(f'{_ENV_PREFIX}SAM_TEXT_PROMPT', 'a custom prompt')
    active = profile_registry.get_active_region_profile()
    assert active is not None
    assert active.name == 'license_plate'
    assert active.secondary_shape_groups == frozenset({'group_a', 'group_b'})
    assert active.sam_text_prompt == 'a custom prompt'
    # Everything not overridden comes from the selected base.
    assert active.detector_model == REFERENCE_LICENSE_PLATE_PROFILE.detector_model
    assert active.aspect_min == REFERENCE_LICENSE_PLATE_PROFILE.aspect_min


def test_env_only_profile_without_a_base(region_env: pytest.MonkeyPatch) -> None:
    region_env.setenv(f'{_ENV_PREFIX}NAME', 'shipping_label')
    region_env.setenv(f'{_ENV_PREFIX}DETECTOR_MODEL', 'label_detector')
    active = profile_registry.get_active_region_profile()
    assert active is not None
    assert active.name == 'shipping_label'
    assert active.detector_model == 'label_detector'
    assert set(profile_registry.get_profiles()) == {'shipping_label'}


def test_select_profile_registered_by_startup_code(region_env: pytest.MonkeyPatch) -> None:
    custom = DetectionProfile(name='shipping_label', detector_model='label_detector')
    profile_registry.register_profile(custom)
    region_env.setenv('OP_REGION_PROFILE', 'shipping_label')
    # A later env resolution re-registers it (as the default) unchanged.
    profile_registry._ENV_RESOLVED = False
    assert profile_registry.get_active_region_profile() == custom


def test_unknown_profile_name_fails_loudly(region_env: pytest.MonkeyPatch) -> None:
    region_env.setenv('OP_REGION_PROFILE', 'no_such_profile')
    with pytest.raises(ValueError, match='OP_REGION_PROFILE'):
        profile_registry.get_active_region_profile()


# =============================================================================
# GET /methods advertises exactly what is configured
# =============================================================================


def _advertised(default_id: str | None = None) -> dict[str, dict[str, object]]:
    from src.services.curation.strategy_registry import _detection_profile_strategies

    return {e['id']: e for e in _detection_profile_strategies(default_id)}  # type: ignore[misc]


def test_methods_axis_empty_when_unconfigured(region_env: pytest.MonkeyPatch) -> None:
    assert _advertised(profile_registry.get_default_profile_name()) == {}


def test_methods_axis_advertises_env_configured_profile(region_env: pytest.MonkeyPatch) -> None:
    region_env.setenv(f'{_ENV_PREFIX}NAME', 'shipping_label')
    entries = _advertised(profile_registry.get_default_profile_name())
    assert set(entries) == {'shipping_label'}
    assert entries['shipping_label']['default'] is True


# =============================================================================
# Consumers degrade cleanly with no profile
# =============================================================================


def test_models_roster_skips_region_models_when_unconfigured(
    region_env: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation.models import _core_models, _region_protected_models

    names = {name for name, *_ in _core_models()}
    assert REFERENCE_LICENSE_PLATE_PROFILE.detector_model not in names
    assert REFERENCE_LICENSE_PLATE_PROFILE.detector_model not in _region_protected_models()

    region_env.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    profile_registry._reset_registry_for_tests()
    names = {name for name, *_ in _core_models()}
    assert REFERENCE_LICENSE_PLATE_PROFILE.detector_model in names


def test_training_candidates_query_uses_neutral_profile(region_env: pytest.MonkeyPatch) -> None:
    from src.routers.curation.regions import _training_candidate_query

    query, _reason = _training_candidate_query('low_conf_correct')
    assert REFERENCE_LICENSE_PLATE_PROFILE.detector_model not in str(query)


def _patch_worker_io(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    pool = MagicMock()
    pool.initialize = AsyncMock()
    pool.close = AsyncMock()
    os_client = MagicMock()
    os_client.search = AsyncMock(return_value={'hits': {'hits': []}})
    os_client.bulk = AsyncMock()
    os_client.close = AsyncMock()
    sam3 = MagicMock()
    sam3.aclose = AsyncMock()
    gemma = MagicMock()
    gemma.aclose = AsyncMock()
    mocks = {
        'AsyncTritonPool': MagicMock(return_value=pool),
        'AsyncOpenSearch': MagicMock(return_value=os_client),
        'Sam3Client': MagicMock(return_value=sam3),
        'VlmLabeler': MagicMock(return_value=gemma),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(worker, name, mock)

    def _noop_signal_handler(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        asyncio.get_event_loop().__class__,
        'add_signal_handler',
        _noop_signal_handler,
        raising=False,
    )
    monkeypatch.setenv('SAM_WORKER_METRICS_PORT', '0')
    return mocks


def _worker_args(tmp_path: Path, sam3_url: str = 'http://sam3.local:8000') -> argparse.Namespace:
    return worker.parse_args(
        [
            '--opensearch=http://os.local:9200',
            '--triton=triton:8001',
            f'--sam3-url={sam3_url}',
            '--gemma-url=http://gemma.local:8000',
            f'--pause-sentinel={tmp_path / "pause.sentinel"}',
            '--max-iterations=1',
        ]
    )


@pytest.mark.asyncio
async def test_worker_is_a_noop_without_a_region_profile(
    region_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mocks = _patch_worker_io(region_env)
    rc = await worker.run(_worker_args(tmp_path))
    assert rc == 0
    mocks['AsyncTritonPool'].assert_not_called()
    mocks['AsyncOpenSearch'].assert_not_called()
    mocks['Sam3Client'].assert_not_called()


@pytest.mark.asyncio
async def test_worker_sends_the_profile_segmenter_prompt(
    region_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    region_env.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    region_env.setenv(f'{_ENV_PREFIX}SAM_TEXT_PROMPT', 'shipping label')
    mocks = _patch_worker_io(region_env)
    assert await worker.run(_worker_args(tmp_path)) == 0
    mocks['Sam3Client'].assert_called_once_with(
        'http://sam3.local:8000', text_prompt='shipping label'
    )


@pytest.mark.asyncio
async def test_worker_disables_segmenter_when_profile_has_no_prompt(
    region_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    region_env.setenv(f'{_ENV_PREFIX}NAME', 'shipping_label')
    mocks = _patch_worker_io(region_env)
    assert await worker.run(_worker_args(tmp_path)) == 0
    mocks['Sam3Client'].assert_called_once_with('', text_prompt='')


def test_secondary_shape_routing_follows_env_groups(
    region_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-11: group resolution comes from the class registry
    (class_name -> group), not the dead ``_ItemTask.group`` field."""
    import scripts.curation.worker.state as worker_state

    region_env.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    region_env.setenv(f'{_ENV_PREFIX}SECONDARY_SHAPE_GROUPS', 'tall_things')
    task = worker._ItemTask(
        crop_id='c',
        image_path='/x',
        vehicle_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        plate_status=None,
        class_name='audi',
    )
    monkeypatch.setattr(worker_state, '_class_group', lambda _name: 'tall_things')
    assert worker._is_secondary_shape(task)
    monkeypatch.setattr(worker_state, '_class_group', lambda _name: 'sportbikes')
    assert not worker._is_secondary_shape(task)


def test_no_secondary_shape_routing_when_profile_has_no_groups(
    region_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.curation.worker.state as worker_state

    region_env.setenv(f'{_ENV_PREFIX}NAME', 'shipping_label')
    task = worker._ItemTask(
        crop_id='c',
        image_path='/x',
        vehicle_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        plate_status=None,
        class_name='cruiserbike',
    )
    monkeypatch.setattr(worker_state, '_class_group', lambda _name: None)
    assert not worker._is_secondary_shape(task)
