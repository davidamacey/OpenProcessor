"""Pytest configuration for the test suite.

Some files under ``tests/`` are standalone smoke-test *scripts* meant to be
run against a live deployment (``python tests/test_full_system.py``), not
pytest suites. They define helper functions named ``test_*(name, ...)`` that
pytest would otherwise miscollect as test cases — failing with
``fixture 'name' not found``. Exclude them from collection here; run them
directly per the project README / CLAUDE.md.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest


# S7: OP_VLM_MODEL has no hardcoded default (src/services/labeling/vlm_client.py
# DEFAULT_MODEL). A deterministic test value, set before any test module
# imports vlm_client, so the whole suite doesn't have to configure it
# per-test just to get a stable, non-empty verifier/labeler string.
os.environ.setdefault('OP_VLM_MODEL', 'test-vlm-model')


if TYPE_CHECKING:
    from collections.abc import Iterator


collect_ignore = [
    'test_full_system.py',
    'test_scrfd_pipeline.py',
    'test_validate_models.py',
    'validate_visual_results.py',
]


@pytest.fixture
def reference_region_profile(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Activate the built-in ``license_plate`` reference region profile.

    The neutral default has no region profile, so the detection worker's
    cascade refuses to run. Tests that exercise the cascade opt in to the
    reference profile the same way a deployment does: ``OP_REGION_PROFILE``.
    """
    import sys

    from src.services.detection import profile_registry

    monkeypatch.setenv('OP_REGION_PROFILE', 'license_plate')
    profile_registry._reset_registry_for_tests()
    # The cascade's no-verdict count is process-wide; a test must not
    # inherit another test's count for the same crop id.
    no_verdict = sys.modules.get('scripts.curation.worker.no_verdict')
    if no_verdict is not None:
        no_verdict.reset_cascade_counter()
    yield
    no_verdict = sys.modules.get('scripts.curation.worker.no_verdict')
    if no_verdict is not None:
        no_verdict.reset_cascade_counter()
    # Lazy re-resolution: the next accessor call (after monkeypatch restores
    # the env) sees the unconfigured default again.
    profile_registry._reset_registry_for_tests()


@pytest.fixture
def reference_ingest_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingest profiles named like the reference deployment: a generic
    proposer named ``coco_yolo11`` (items it leaves unlabeled carry
    ``coco_yolo11_proposal``) and a secondary classifier named ``v6``
    (``v6_model``). The class_source vocabulary the worker and clustering
    code filter on is derived from these names."""
    monkeypatch.setenv('OP_INGEST_PRIMARY_NAME', 'coco_yolo11')
    monkeypatch.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'classifier')
    monkeypatch.setenv('OP_INGEST_SECONDARY_NAME', 'v6')
