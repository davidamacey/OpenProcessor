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
    REFERENCE_REGION_DETECTOR_MODEL,
)

import scripts.curation.region_worker_main as worker
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
    assert neutral.segmenter_text_prompt == ''
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
    region_env.setenv(f'{_ENV_PREFIX}SEGMENTER_TEXT_PROMPT', 'a custom prompt')
    active = profile_registry.get_active_region_profile()
    assert active is not None
    assert active.name == 'license_plate'
    assert active.secondary_shape_groups == frozenset({'group_a', 'group_b'})
    assert active.segmenter_text_prompt == 'a custom prompt'
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
    from src.services.curation.axis_copy import detection_profile_strategies

    return {e['id']: e for e in detection_profile_strategies(default_id)}  # type: ignore[misc]


def test_methods_axis_empty_when_unconfigured(region_env: pytest.MonkeyPatch) -> None:
    assert _advertised(profile_registry.get_default_profile_name()) == {}


def test_methods_axis_advertises_env_configured_profile(region_env: pytest.MonkeyPatch) -> None:
    region_env.setenv(f'{_ENV_PREFIX}NAME', 'shipping_label')
    entries = _advertised(profile_registry.get_default_profile_name())
    # W2: an explicit 'off' choice is offered once there is at least one
    # profile to turn off (any_domain_plan.md §9 W2).
    assert set(entries) == {'shipping_label', 'off'}
    assert entries['shipping_label']['default'] is True
    assert entries['off']['default'] is False


# =============================================================================
# Consumers degrade cleanly with no profile
# =============================================================================


def test_models_roster_skips_region_models_when_unconfigured(
    region_env: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation.models import _core_models
    from src.services.model_unload_guard import region_protected_models as _region_protected_models

    names = {name for name, *_ in _core_models()}
    assert REFERENCE_REGION_DETECTOR_MODEL not in names
    assert REFERENCE_REGION_DETECTOR_MODEL not in _region_protected_models()

    region_env.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    region_env.setenv(f'{_ENV_PREFIX}DETECTOR_MODEL', REFERENCE_REGION_DETECTOR_MODEL)
    profile_registry._reset_registry_for_tests()
    names = {name for name, *_ in _core_models()}
    assert REFERENCE_REGION_DETECTOR_MODEL in names


def test_training_candidates_query_uses_neutral_profile(region_env: pytest.MonkeyPatch) -> None:
    from src.routers.curation.regions import _training_candidate_query

    query, box_clause, _reason = _training_candidate_query('low_conf_correct')
    assert REFERENCE_REGION_DETECTOR_MODEL not in str((query, box_clause))


def _patch_worker_io(
    monkeypatch: pytest.MonkeyPatch, *, seed_default_project: bool = True
) -> dict[str, MagicMock]:
    """``seed_default_project``: B1 -- the worker now builds a runtime
    only for projects ``project_registry.active_projects()`` actually
    returns, so a test that wants the worker to build ANYTHING (a
    detector/segmenter for the env-resolved profile) needs a real
    ``default`` project doc for the fake registry to find. A test that
    wants to prove nothing gets built (no profile anywhere) passes
    ``False`` and skips the registry setup entirely."""
    pool = MagicMock()
    pool.initialize = AsyncMock()
    pool.close = AsyncMock()
    os_client = MagicMock()
    if seed_default_project:
        from src.config.curation import base_curation_config
        from src.config.projects import resources_for_new
        from src.services.projects.registry import (
            REVISION_DOC_ID,
            ProjectRecord,
            projects_index,
            record_to_doc,
        )

        default_record = ProjectRecord(
            slug='default',
            display_name='Default',
            description='',
            status='active',
            revision=1,
            created_at='',
            updated_at='',
            origin=None,
            resources=resources_for_new('default', base_curation_config()),
        )

        async def _search(*, index: str, body: dict) -> dict:  # type: ignore[type-arg]
            if index == projects_index():
                return {'hits': {'hits': [{'_source': record_to_doc(default_record)}]}}
            return {'hits': {'hits': []}}

        async def _get(*, index: str, id: str) -> dict:  # type: ignore[type-arg] # noqa: A002
            if index == projects_index() and id == REVISION_DOC_ID:
                return {'found': True, '_source': {'revision': 1}}
            return {'found': False}

        os_client.search = AsyncMock(side_effect=_search)
        os_client.get = AsyncMock(side_effect=_get)
    else:
        os_client.search = AsyncMock(return_value={'hits': {'hits': []}})
    os_client.bulk = AsyncMock()
    os_client.close = AsyncMock()
    segmenter = MagicMock()
    segmenter.aclose = AsyncMock()
    vlm = MagicMock()
    vlm.aclose = AsyncMock()
    mocks = {
        'AsyncTritonPool': MagicMock(return_value=pool),
        'make_script_opensearch': MagicMock(return_value=os_client),
        'SegmenterClient': MagicMock(return_value=segmenter),
        'build_vlm_labeler': MagicMock(return_value=vlm),
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
    monkeypatch.setenv('OP_REGION_WORKER_METRICS_PORT', '0')
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm.local:8000')
    return mocks


def _worker_args(
    tmp_path: Path, segmenter_url: str = 'http://segmenter.local:8000'
) -> argparse.Namespace:
    return worker.parse_args(
        [
            '--opensearch=http://os.local:9200',
            '--triton=triton:8001',
            f'--segmenter-url={segmenter_url}',
            f'--pause-sentinel={tmp_path / "pause.sentinel"}',
            '--max-iterations=1',
        ]
    )


@pytest.mark.asyncio
async def test_worker_is_a_noop_without_a_region_profile(
    region_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B1: the worker now initializes shared infra (Triton pool,
    OpenSearch client) unconditionally -- it can't know upfront whether
    ANY active project will ever have a profile, since projects are
    dynamic. What must still never happen with no profile configured
    anywhere is building a per-project runtime (SegmenterClient etc).
    """
    mocks = _patch_worker_io(region_env, seed_default_project=True)
    rc = await worker.run(_worker_args(tmp_path))
    assert rc == 0
    mocks['SegmenterClient'].assert_not_called()


@pytest.mark.asyncio
async def test_worker_sends_the_profile_segmenter_prompt(
    region_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    region_env.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    region_env.setenv(f'{_ENV_PREFIX}SEGMENTER_TEXT_PROMPT', 'shipping label')
    region_env.setenv(f'{_ENV_PREFIX}SEGMENTER_NAME', 'my_segmenter')
    mocks = _patch_worker_io(region_env)
    assert await worker.run(_worker_args(tmp_path)) == 0
    mocks['SegmenterClient'].assert_called_once_with(
        'http://segmenter.local:8000', text_prompt='shipping label', source_name='my_segmenter'
    )


@pytest.mark.asyncio
async def test_worker_disables_segmenter_when_profile_has_no_prompt(
    region_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    region_env.setenv(f'{_ENV_PREFIX}NAME', 'shipping_label')
    mocks = _patch_worker_io(region_env)
    assert await worker.run(_worker_args(tmp_path)) == 0
    mocks['SegmenterClient'].assert_called_once_with('', text_prompt='', source_name='sam3')


def test_secondary_shape_routing_follows_env_groups(
    region_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Group resolution comes from the class registry
    (class_name -> group), not the dead ``_ItemTask.group`` field."""
    import scripts.curation.worker.state as worker_state

    region_env.setenv('OP_REGION_PROFILE_PATH', EXAMPLE_LICENSE_PLATE_PROFILE_PATH)
    region_env.setenv(f'{_ENV_PREFIX}SECONDARY_SHAPE_GROUPS', 'tall_things')
    task = worker._ItemTask(
        crop_id='c',
        image_path='/x',
        item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        region_status=None,
        class_name='audi',
    )
    monkeypatch.setattr(worker_state, '_class_group', lambda _name: 'tall_things')
    assert worker._is_secondary_shape(task)
    monkeypatch.setattr(worker_state, '_class_group', lambda _name: 'unrelated_group')
    assert not worker._is_secondary_shape(task)


def test_no_secondary_shape_routing_when_profile_has_no_groups(
    region_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.curation.worker.state as worker_state

    region_env.setenv(f'{_ENV_PREFIX}NAME', 'shipping_label')
    task = worker._ItemTask(
        crop_id='c',
        image_path='/x',
        item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        region_status=None,
        class_name='class_b',
    )
    monkeypatch.setattr(worker_state, '_class_group', lambda _name: None)
    assert not worker._is_secondary_shape(task)


def test_retired_empty_class_ids_is_dropped_on_read() -> None:
    body = {'name': 'legacy', 'class_ids': []}
    assert profile_registry.region_profile_from_dict(body).name == 'legacy'


def test_non_empty_class_ids_stays_rejected() -> None:
    with pytest.raises(ValueError, match='unknown field'):
        profile_registry.region_profile_from_dict({'name': 'legacy', 'class_ids': [1]})
