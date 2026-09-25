"""M2: `GET /curation/models/status` must list every configured
pipeline model that produces labels -- the primary item proposer, the
optional secondary classifier, the region detector, the segmenter and
OCR det/rec -- not just a fixed CLIP/PE-encoder subset.

Live evidence: `/models/status` omitted the primary proposer, the
secondary classifier and the segmenter entirely, even though they
produced most of the labels the labeler shows. Every entry is resolved
from config (`ingest_profiles.py` / `DetectionProfile`), never a
hardcoded id, so a different deployment gets its own roster for free.
"""

from __future__ import annotations

import pytest

from src.services.detection import profile_registry


@pytest.fixture(autouse=True)
def _reset_profiles():
    profile_registry._reset_registry_for_tests()
    yield
    profile_registry._reset_registry_for_tests()


def test_core_models_includes_primary_proposer(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'item_proposer_v9')
    names = {name for name, *_ in models_mod._core_models()}
    assert 'item_proposer_v9' in names


def test_core_models_includes_secondary_classifier_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'classifier_v6')
    names = {name for name, *_ in models_mod._core_models()}
    assert 'classifier_v6' in names


def test_core_models_omits_secondary_classifier_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.delenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', raising=False)
    names = {name for name, *_ in models_mod._core_models()}
    assert 'classifier_v6' not in names


def test_core_models_includes_the_segmenter(
    monkeypatch: pytest.MonkeyPatch, reference_region_profile: None
) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.setenv(f'{profile_registry.REGION_DETECTION_ENV_PREFIX}SEGMENTER_NAME', 'seg9')
    profile_registry._reset_registry_for_tests()
    names = {name for name, *_ in models_mod._core_models()}
    assert 'seg9' in names
