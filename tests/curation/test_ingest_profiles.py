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
    narrowed = DetectionProfile(name='item', assigns_class=True, class_ids=frozenset({2, 7}))
    items = _decode(narrowed, [0, 2, 7, 15])
    assert sorted(item.class_id for item in items) == [2, 7]


def test_primary_without_class_ids_keeps_every_class() -> None:
    items = _decode(DetectionProfile(name='item'), [0, 2, 7, 15])
    assert len(items) == 4


# =============================================================================
# assigns_class: a generic proposer never labels from its own class ids
# =============================================================================


class _DomainRegistry:
    """A domain taxonomy whose ids collide with the proposer's ids."""

    class _Entry:
        def __init__(self, name: str) -> None:
            self.class_name = name

    def get(self, class_id: int) -> Any:
        return self._Entry(f'domain_class_{class_id}')


def test_default_primary_does_not_assign_class(tmp_path: Any) -> None:
    labels = tmp_path / 'labels.txt'
    labels.write_text('person\nbicycle\ncar\n')
    profile = DetectionProfile(name='item', labels_path=str(labels))
    detector = WholeImageDetector(triton_pool=None, registry=_DomainRegistry(), profile=profile)
    items = detector.decode_primary_row(
        np.array([2]),
        np.array([[0.1, 0.1, 0.5, 0.5]] * 2, dtype=np.float32),
        np.array([0.99, 0.2], dtype=np.float32),
        np.array([2, 7], dtype=np.float32),
        1.0,
        (0.0, 0.0),
        640,
    )
    assert [i.class_id for i in items] == [None, None]
    assert [i.class_name for i in items] == [None, None]
    assert [i.class_source for i in items] == ['item_proposal', 'item_proposal']
    # The proposer's own label, not the registry entry sharing its id;
    # an id past the labels file falls back to the bare id.
    assert [i.proposal_name for i in items] == ['car', '7']


def test_assigning_primary_keeps_registry_labelling() -> None:
    profile = DetectionProfile(name='item', assigns_class=True)
    detector = WholeImageDetector(triton_pool=None, registry=_DomainRegistry(), profile=profile)
    items = detector.decode_primary_row(
        np.array([1]),
        np.array([[0.1, 0.1, 0.5, 0.5]], dtype=np.float32),
        np.array([0.99], dtype=np.float32),
        np.array([2], dtype=np.float32),
        1.0,
        (0.0, 0.0),
        640,
    )
    assert items[0].class_id == 2
    assert items[0].class_name == 'domain_class_2'
    assert items[0].class_source == 'item_model'


def test_assigns_class_and_labels_path_from_env(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv('OP_INGEST_PRIMARY_ASSIGNS_CLASS', 'true')
    clean_env.setenv('OP_INGEST_PRIMARY_LABELS_PATH', '/models/proposer/labels.txt')
    profile = ingest_router._get_detection_profile()
    assert profile.assigns_class is True
    assert profile.labels_path == '/models/proposer/labels.txt'
    clean_env.delenv('OP_INGEST_PRIMARY_ASSIGNS_CLASS')
    assert ingest_router._get_detection_profile().assigns_class is False


# =============================================================================
# class_source vocabulary derives from the configured profile names
# =============================================================================


def test_class_sources_follow_profile_names(clean_env: pytest.MonkeyPatch) -> None:
    from src.services.curation import ingest_class_sources as cs

    clean_env.setenv('OP_INGEST_PRIMARY_NAME', 'proposer')
    assert cs.unlabeled_proposal_class_sources() == {'proposer_proposal', 'proposer_low_conf'}
    assert cs.classifier_class_sources() == frozenset()
    assert cs.confident_class_sources() == ('human', 'vlm')

    clean_env.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'clf')
    clean_env.setenv('OP_INGEST_SECONDARY_NAME', 'clf')
    assert cs.classifier_class_sources() == {'clf_model'}
    clean_env.setenv('OP_INGEST_PRIMARY_ASSIGNS_CLASS', '1')
    assert cs.classifier_class_sources() == {'clf_model', 'proposer_model'}
    assert cs.confident_class_sources() == ('clf_model', 'human', 'proposer_model', 'vlm')


def test_worker_cohort_gate_uses_configured_names(clean_env: pytest.MonkeyPatch) -> None:
    from scripts.curation.worker.combined import _is_combined_cohort

    clean_env.setenv('OP_INGEST_PRIMARY_NAME', 'proposer')
    clean_env.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'clf')
    clean_env.setenv('OP_INGEST_SECONDARY_NAME', 'clf')
    assert _is_combined_cohort('proposer_proposal', 0.99)
    assert _is_combined_cohort('proposer_low_conf', 0.99)
    assert _is_combined_cohort('clf_model', 0.5)
    assert not _is_combined_cohort('clf_model', 0.95)
    # A name from some other deployment is not special.
    assert not _is_combined_cohort('coco_yolo11_proposal', 0.99)


def test_confident_sources_resolved_from_env_at_import(clean_env: pytest.MonkeyPatch) -> None:
    import subprocess  # nosec B404 - this repo's interpreter on a fixed snippet
    import sys
    from pathlib import Path

    snippet = (
        'from src.services.curation.clustering import embedding_reduce as e\n'
        'print(",".join(e.CONFIDENT_CLASS_SOURCES))\n'
    )
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith('OP_INGEST_')},
        'OP_INGEST_SECONDARY_DETECTOR_MODEL': 'clf',
        'OP_INGEST_SECONDARY_NAME': 'clf',
    }
    out = subprocess.run(  # nosec B603
        [sys.executable, '-c', snippet],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == 'clf_model,human,vlm'
