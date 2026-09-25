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

    monkeypatch.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'secondary_classifier_x1')
    names = {name for name, *_ in models_mod._core_models()}
    assert 'secondary_classifier_x1' in names


def test_core_models_omits_secondary_classifier_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.delenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', raising=False)
    names = {name for name, *_ in models_mod._core_models()}
    assert 'secondary_classifier_x1' not in names


def test_core_models_includes_the_segmenter(
    monkeypatch: pytest.MonkeyPatch, reference_region_profile: None
) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.setenv(f'{profile_registry.REGION_DETECTION_ENV_PREFIX}SEGMENTER_NAME', 'seg9')
    profile_registry._reset_registry_for_tests()
    names = {name for name, *_ in models_mod._core_models()}
    assert 'seg9' in names


# =============================================================================
# Operator-facing `role` copy must not leak code identifiers
#
# Live UI screenshot review found role strings like "(configured via
# OP_INGEST_PRIMARY_*)" and "(configured via DetectionProfile.segmenter_name)"
# served straight to the labeler UI -- env var / class names have no
# business in operator-facing text. Configuration provenance belongs in
# docs/comments, not the served `role` string.
# =============================================================================


_LEAKY_SUBSTRINGS = (
    'OP_INGEST_PRIMARY',
    'OP_INGEST_SECONDARY',
    'OP_SEGMENTER',
    'DetectionProfile',
    'os.environ',
)


def test_core_models_roles_do_not_leak_code_identifiers(
    monkeypatch: pytest.MonkeyPatch, reference_region_profile: None
) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'item_proposer_v9')
    monkeypatch.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'secondary_classifier_x1')
    profile_registry._reset_registry_for_tests()

    entries = models_mod._core_models()
    assert entries, 'expected a non-empty roster to actually exercise this check'
    for name, _friendly, role, _mtype in entries:
        for leaky in _LEAKY_SUBSTRINGS:
            assert leaky not in role, f'{name!r} role leaks a code identifier: {role!r}'
