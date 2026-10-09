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
from pathlib import Path
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
    """The ``default`` project record, with its resources recomputed on
    every access by the same ``resources_for_new`` every project uses --
    so a test that points ``OP_PROJECTS_DATA_ROOT`` / ``OP_STATE_DIR`` /
    ``OP_PROJECT_INDEX_PREFIX`` somewhere else sees its own names and
    paths, exactly as a freshly bootstrapped ``default`` would."""

    slug = 'default'
    display_name = 'Default'
    description = ''
    status = 'active'
    revision = 1
    created_at = ''
    updated_at = ''
    origin = None

    @property
    def resources(self) -> object:
        from src.config.curation import base_curation_config
        from src.config.projects import resources_for_new

        return resources_for_new('default', base_curation_config())


DEFAULT_RECORD = _EnvDefaultProjectRecord()


@pytest.fixture(autouse=True)
def _projects_data_root(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-project data (class registry, exports) lives under
    ``OP_PROJECTS_DATA_ROOT``; never let a test write into the repo's
    ``./data``. A test that sets its own value wins."""
    import os

    if 'OP_PROJECTS_DATA_ROOT' not in os.environ:
        monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path_factory.mktemp('projects')))


@pytest.fixture
def project_export_root() -> Path:
    """The bound ``default`` project's ``export_root`` (created). Training
    routes refuse a ``dataset_export_dir`` outside it
    (``export_outside_project``), so fixture exports live here."""
    from src.config.curation import get_curation_config

    root = Path(get_curation_config().export_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


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
            with bind_project(DEFAULT_RECORD):  # type: ignore[arg-type]
                yield
    finally:
        # A script entry point run in-process binds the whole process;
        # never let that leak into the next test.
        bind_process_project(None)


@pytest.fixture(autouse=True)
def _requests_start_unbound(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every ``TestClient`` request runs in a fresh ``contextvars.Context``,
    the way uvicorn serves it: the test's own binding (above) never leaks
    into the app, so a route that forgets to bind fails here too.

    Also installs a project registry that never reaches the network: it
    knows ``default`` and nothing else. A test that needs
    more projects installs its own with ``set_project_registry``."""
    import contextvars

    from fastapi.testclient import TestClient

    from src.services.projects import registry as registry_mod

    real_request = TestClient.request

    def _unbound_request(self: TestClient, *args: object, **kwargs: object) -> object:
        return contextvars.Context().run(real_request, self, *args, **kwargs)

    monkeypatch.setattr(TestClient, 'request', _unbound_request)

    # The lifespan (``with TestClient(app)``) starts unbound too, as it does
    # under uvicorn: its background tasks must never inherit a project.
    real_enter = TestClient.__enter__

    def _unbound_enter(self: TestClient) -> TestClient:
        return contextvars.Context().run(real_enter, self)

    monkeypatch.setattr(TestClient, '__enter__', _unbound_enter)

    def _no_registry_client() -> object:
        raise ConnectionError('no project registry in unit tests')

    stub = registry_mod.ProjectRegistry(_no_registry_client)
    stub._by_slug = {'default': DEFAULT_RECORD}  # type: ignore[dict-item]

    # M2 made a stale registry (``.stale`` True after a failed
    # ensure_fresh) refuse every non-GET route at bind time. This stub
    # never reaches the network on purpose -- it is a deliberately frozen,
    # authoritative-for-tests snapshot, not an instance that fell behind --
    # so a real ensure_fresh() call against it would mark it stale and
    # 409 every write-route test in the suite that doesn't install its own
    # registry. Make ensure_fresh here a no-op success instead of a real
    # (failing) refresh attempt.
    async def _never_stale(self: object) -> None:
        return None

    monkeypatch.setattr(stub, 'ensure_fresh', _never_stale.__get__(stub))
    stub._refreshed = True
    registry_mod.set_project_registry(stub)

    # Script entry points resolve ``--project`` through the registry; here
    # it holds only the env-derived ``default`` (tests/projects/
    # test_script_binding.py exercises the real read).
    from src.services.projects import script_binding

    async def _default_only_registry(_url: str | None = None) -> object:
        registry = registry_mod.ProjectRegistry(_no_registry_client)
        registry._by_slug = {'default': DEFAULT_RECORD}  # type: ignore[dict-item]
        registry._refreshed = True
        return registry

    monkeypatch.setattr(script_binding, 'load_registry', _default_only_registry)
    try:
        yield
    finally:
        registry_mod.set_project_registry(None)


# Module-level TTL caches (module, attribute). Production keys each by
# project, but every unit test binds the same ``default`` record, so a cache
# one test fills would answer the next test's first request.
_PROCESS_CACHES = (
    ('src.services.curation.strategy_registry', '_COVERAGE_CACHE'),
    ('src.routers.curation.select', '_ORDER_CACHE'),
    ('src.services.curation.clustering.outliers', '_CACHE'),
    ('src.clients.curation_opensearch', '_settings_cache'),
    ('src.clients.curation_opensearch', '_INNER_RESULT_WINDOWS'),
    ('src.routers.curation.regions_fp', '_suspected_fp_cache'),
    ('src.services.curation.eval_datasets', '_CACHE'),
    # Config-store snapshots (W2): one ConfigStore per project, keyed by
    # slug in a module-level dict -- every test binds the same ``default``
    # project, so a store one test mutates would leak into the next.
    ('src.services.config_store.store', '_STORES'),
    # The global (non-project-scoped) config store singleton (M3) -- its
    # own cache dict, never conflated with `_STORES` above.
    ('src.services.config_store.global_store', '_GLOBAL_STORE'),
    # W9: immutable VLM endpoint revision copies fetched from the registry.
    ('src.services.config_store.vlm_snapshot', '_REVISION_CACHE'),
    ('src.services.training.preflight_scan', '_scan_cache'),
    # W9: the labeler factory's LRU (one labeler per endpoint revision + pack).
    ('src.services.labeling.vlm_factory', '_LABELERS'),
)


def _clear_process_caches() -> None:
    import sys

    for module_name, attr in _PROCESS_CACHES:
        module = sys.modules.get(module_name)
        if module is not None:
            getattr(module, attr).clear()
    policy = sys.modules.get('src.services.labeling.vlm_url_policy')
    if policy is not None:
        policy.reset_policy_caches()
    capacity = sys.modules.get('src.services.projects.capacity')
    if capacity is not None:
        setattr(capacity, '_cache', None)  # noqa: B010 - module attr unknown to mypy


@pytest.fixture(autouse=True)
def _fresh_process_caches() -> Iterator[None]:
    """No test inherits (or leaves behind) a module TTL cache."""
    _clear_process_caches()
    yield
    _clear_process_caches()


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The VLM URL policy resolves hostnames; no test may hit real DNS (an
    unresolvable name is what the policy sees). A test that needs answers
    patches ``vlm_url_policy._resolve`` itself."""
    import src.services.labeling.vlm_url_policy as policy

    monkeypatch.setattr(policy, '_resolve', lambda _host: [])


@pytest.fixture(autouse=True)
def _ephemeral_worker_metrics_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that run ``worker.run()`` start its real /metrics HTTP server. On the
    default fixed port, two of them running at once (xdist workers, or two suites on
    one host) fail with "address already in use". Port 0 asks the OS for a free one."""
    monkeypatch.setenv('OP_REGION_WORKER_METRICS_PORT', '0')


@pytest.fixture
def vlm_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """An ``env`` built-in VLM endpoint (``OP_VLM_URL``), i.e. a project
    that never activated one. Returns the base URL."""
    url = 'http://vlm.invalid:8000/v1'
    monkeypatch.setenv('OP_VLM_URL', url)
    monkeypatch.setenv('OP_VLM_MODEL', 'test-vlm')
    return url


collect_ignore = [
    'test_full_system.py',
    'test_scrfd_pipeline.py',
    'test_validate_models.py',
    'validate_visual_results.py',
]


@pytest.fixture
def reference_region_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Activate the example ``license_plate`` region profile
    (``examples/region_profiles/license_plate.json``).

    The neutral default has no region profile, so the detection worker's
    cascade refuses to run. Tests that exercise the cascade opt in to the
    example profile the same way a deployment does: ``OP_REGION_PROFILE_PATH``
    pointed at a profile file. No profile ships built in.

    The example runs segmenter-only (empty ``detector_model``); the fixture
    names a detector the way a deployment that exported its own would, so
    the cascade tests exercise every leg. ``parent_classes`` is emptied (every
    item in scope) because these tests drive the cascade over arbitrary class
    names; the example's own vehicle scope is pinned by
    ``tests/test_example_region_profiles.py``.
    """
    import json

    from _region_profile_fixture import REFERENCE_REGION_DETECTOR_MODEL

    from src.services.detection import profile_registry

    example_path = (
        Path(__file__).resolve().parents[1] / 'examples' / 'region_profiles' / 'license_plate.json'
    )
    scope_free = tmp_path / 'reference_region_profile.json'
    scope_free.write_text(
        json.dumps({**json.loads(example_path.read_text()), 'parent_classes': []}),
        encoding='utf-8',
    )
    monkeypatch.setenv('OP_REGION_PROFILE_PATH', str(scope_free))
    monkeypatch.setenv('OP_REGION_DETECTION_DETECTOR_MODEL', REFERENCE_REGION_DETECTOR_MODEL)
    profile_registry._reset_registry_for_tests()
    yield
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
