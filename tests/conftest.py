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


# OP_VLM_MODEL has no hardcoded default (src/services/labeling/vlm_client.py
# DEFAULT_MODEL). A deterministic test value, set before any test module
# imports vlm_client, so the whole suite doesn't have to configure it
# per-test just to get a stable, non-empty verifier/labeler string.
os.environ.setdefault('OP_VLM_MODEL', 'test-vlm-model')


if TYPE_CHECKING:
    from collections.abc import Iterator


class _EnvDefaultProjectRecord:
    """The ``default`` project record, with resources recomputed from the
    env-built base config on every access -- so a test that swaps
    ``src.config.curation._default_curation_config`` (or its env) sees its
    own index names and paths, exactly as a fresh process would."""

    slug = 'default'
    display_name = 'Default'
    description = ''
    status = 'active'
    revision = 0
    created_at = ''
    updated_at = ''
    origin = None

    @property
    def resources(self) -> object:
        from src.config.curation import base_curation_config
        from src.config.projects import resources_for_default

        return resources_for_default(base_curation_config())


@pytest.fixture(autouse=True)
def _bind_default_project(request: pytest.FixtureRequest) -> Iterator[None]:
    """Bind the ``default`` project for every test (projects_plan.md §3.3).

    Project-scoped config raises ``ProjectNotBound`` when nothing is
    bound; production binds per request (route dependency) or per script
    process (``--project``). Tests that exercise binding itself, or an
    entry point that must bind on its own, opt out with
    ``@pytest.mark.unbound``."""
    from src.config.project_context import bind_process_project, bind_project

    try:
        if request.node.get_closest_marker('unbound') is not None:
            yield
        else:
            with bind_project(_EnvDefaultProjectRecord()):  # type: ignore[arg-type]
                yield
    finally:
        # A script entry point run in-process binds the whole process;
        # never let that leak into the next test.
        bind_process_project(None)


collect_ignore = [
    'test_full_system.py',
    'test_scrfd_pipeline.py',
    'test_validate_models.py',
    'validate_visual_results.py',
]


@pytest.fixture
def reference_region_profile(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Activate the example ``license_plate`` region profile
    (``examples/region_profiles/license_plate.json``).

    The neutral default has no region profile, so the detection worker's
    cascade refuses to run. Tests that exercise the cascade opt in to the
    example profile the same way a deployment does: ``OP_REGION_PROFILE_PATH``
    pointed at a profile file. No profile ships built in.

    The example runs segmenter-only (empty ``detector_model``); the fixture
    names a detector the way a deployment that exported its own would, so
    the cascade tests exercise every leg.
    """
    import sys
    from pathlib import Path

    from _region_profile_fixture import REFERENCE_REGION_DETECTOR_MODEL

    from src.services.detection import profile_registry

    example_path = (
        Path(__file__).resolve().parents[1] / 'examples' / 'region_profiles' / 'license_plate.json'
    )
    monkeypatch.setenv('OP_REGION_PROFILE_PATH', str(example_path))
    monkeypatch.setenv('OP_REGION_DETECTION_DETECTOR_MODEL', REFERENCE_REGION_DETECTOR_MODEL)
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
    ``coco_yolo11_proposal``) and a secondary classifier named
    ``classifier`` (``classifier_model``). The class_source vocabulary
    the worker and clustering code filter on is derived from these
    names."""
    monkeypatch.setenv('OP_INGEST_PRIMARY_NAME', 'coco_yolo11')
    monkeypatch.setenv('OP_INGEST_SECONDARY_DETECTOR_MODEL', 'classifier')
    monkeypatch.setenv('OP_INGEST_SECONDARY_NAME', 'classifier')
