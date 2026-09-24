"""Ingest detector configuration: ``OP_INGEST_PRIMARY_*`` /
``OP_INGEST_SECONDARY_*``, separate from the region profile's
``OP_REGION_PROFILE`` / ``OP_REGION_DETECTION_*``.

A real deployment runs a generic item proposer at ingest AND a different
region detector in the cascade at the same time, so the two must never read
the same env vars. The retired shared ``OP_DETECTION_*`` prefix fails loudly.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from src.config import DetectionProfile
from src.routers.curation import ingest as ingest_router
from src.services.curation.ingest_detect import WholeImageDetector
from src.services.detection import profile_registry


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    for key in list(os.environ):
        if key.startswith(('OP_INGEST_', 'OP_DETECTION_', 'OP_REGION_DETECTION_')):
            monkeypatch.delenv(key)
    monkeypatch.delenv('OP_REGION_PROFILE', raising=False)
    profile_registry._reset_registry_for_tests()
    yield monkeypatch
    profile_registry._reset_registry_for_tests()


def test_primary_profile_reads_ingest_primary_namespace(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'item_proposer')
    clean_env.setenv('OP_INGEST_PRIMARY_INPUT_SIZE', '640')
    clean_env.setenv('OP_INGEST_PRIMARY_CLASS_IDS', '2,3,5,7')
    profile = ingest_router._get_detection_profile()
    assert profile.name == 'item'
    assert profile.detector_model == 'item_proposer'
    assert profile.input_size == 640
    assert profile.class_ids == frozenset({2, 3, 5, 7})


def test_secondary_profile_off_unless_model_set(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_SECONDARY_INPUT_SIZE', '640')
    assert ingest_router._get_secondary_profile() is None


def test_secondary_profile_from_ingest_secondary_namespace(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'ensemble_classifier')
    clean_env.setenv('OP_INGEST_SECONDARY_DETECTOR_VERSION', '3')
    clean_env.setenv('OP_INGEST_SECONDARY_INPUT_SIZE', '1280')
    clean_env.setenv('OP_INGEST_SECONDARY_CONFIDENCE_FLOOR', '0.6')
    clean_env.setenv('OP_INGEST_SECONDARY_NAME', 'v2')
    profile = ingest_router._get_secondary_profile()
    assert profile is not None
    assert profile.detector_model == 'ensemble_classifier'
    assert profile.detector_version == '3'
    assert profile.input_size == 1280
    assert profile.confidence_floor == 0.6
    assert profile.name == 'v2'


def test_ingest_and_region_profiles_are_independent(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'item_proposer')
    clean_env.setenv('OP_REGION_PROFILE', 'license_plate')
    region = profile_registry.get_active_region_profile()
    assert region is not None
    assert region.detector_model != 'item_proposer'
    assert ingest_router._get_detection_profile().detector_model == 'item_proposer'


def test_retired_detection_prefix_fails_loudly(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_DETECTION_DETECTOR_MODEL', 'item_proposer')
    with pytest.raises(ValueError, match='OP_INGEST_PRIMARY_'):
        ingest_router._get_detection_profile()
    with pytest.raises(ValueError, match='OP_REGION_DETECTION_'):
        profile_registry.region_profile_from_env()


@pytest.mark.asyncio
async def test_ingest_service_receives_the_secondary_profile(
    clean_env: pytest.MonkeyPatch,
) -> None:
    import src.main as main_module

    clean_env.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'item_proposer')
    clean_env.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'ensemble_classifier')
    clean_env.setattr(main_module, 'get_async_triton_pool', lambda: object())
    clean_env.setattr(main_module.app.state, 'pe_encoder', object(), raising=False)

    service = await ingest_router._get_ingest_service(object(), object())
    assert service.profile.detector_model == 'item_proposer'
    assert service.secondary_profile is not None
    assert service.secondary_profile.detector_model == 'ensemble_classifier'
    assert service.detector.secondary_profile is service.secondary_profile


@pytest.mark.asyncio
async def test_ingest_service_503s_on_retired_prefix(clean_env: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    import src.main as main_module

    clean_env.setenv('OP_DETECTION_DETECTOR_MODEL', 'item_proposer')
    clean_env.setattr(main_module.app.state, 'pe_encoder', object(), raising=False)
    with pytest.raises(HTTPException) as info:
        await ingest_router._get_ingest_service(object(), object())
    assert info.value.status_code == 503
    assert 'OP_INGEST_PRIMARY_' in str(info.value.detail)


class _Registry:
    def get(self, _class_id: int) -> Any:
        return None


def _decode(profile: DetectionProfile, classes: list[int]) -> list[Any]:
    detector = WholeImageDetector(triton_pool=None, registry=_Registry(), profile=profile)
    n = len(classes)
    return detector.decode_primary_row(
        np.array([n]),
        np.array([[0.1, 0.1, 0.5, 0.5]] * n, dtype=np.float32),
        np.array([0.9] * n, dtype=np.float32),
        np.array(classes, dtype=np.float32),
        1.0,
        (0.0, 0.0),
        640,
    )


def test_primary_class_ids_filter_which_detections_become_items() -> None:
    narrowed = DetectionProfile(name='item', class_ids=frozenset({2, 7}))
    items = _decode(narrowed, [0, 2, 7, 15])
    assert sorted(item.class_id for item in items) == [2, 7]


def test_primary_without_class_ids_keeps_every_class() -> None:
    items = _decode(DetectionProfile(name='item'), [0, 2, 7, 15])
    assert len(items) == 4
