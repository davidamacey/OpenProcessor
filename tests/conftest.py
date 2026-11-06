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


# The `db` tier's PostgreSQL fixtures (pg, two_sessions); inert unless a test asks for them.
pytest_plugins = ['fixtures.postgres', 'fixtures.db_marker_policy', 'fixtures.event_capture']

# OP_VLM_MODEL has no hardcoded default (src/services/labeling/vlm_client.py
# DEFAULT_MODEL). A deterministic test value, set before any test module
# imports vlm_client, so the whole suite doesn't have to configure it
# per-test just to get a stable, non-empty verifier/labeler string.
os.environ.setdefault('OP_VLM_MODEL', 'test-vlm-model')

# Tests that enter the app lifespan build a real OpenSearch client from the
# default in-cluster hostname. On a CI runner that name resolves slowly (10-15 s
# per lookup, several lookups per startup), which made lifespan setup take
# minutes and tripped the 400 s per-test timeout. A refused loopback port fails
# at once, and every test that needs OpenSearch fakes it.
os.environ.setdefault('OPENSEARCH_URL', 'http://127.0.0.1:9')


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
        from src.config.curation_accessors import base_curation_config
        from src.config.projects import resources_for_new

        return resources_for_new('default', base_curation_config())


DEFAULT_RECORD = _EnvDefaultProjectRecord()


@pytest.fixture(autouse=True)
def _control_plane_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """The startup gate finds every dependency up and the schema current.

    The lifespan refuses to start without PostgreSQL, the queue, OpenSearch and Triton, and the
    Docker-free suite has none of them. A test about the gate itself replaces these two names
    again with its own probes."""
    from src.app import startup_gate

    monkeypatch.setattr(startup_gate, 'startup_probes', lambda _settings: {})
    monkeypatch.setattr(startup_gate, 'check_control_plane', lambda: None)


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


@pytest.fixture(scope='session')
def op_weights():
    """``op_weights('yolo26n.pt')`` -> cached official file; skips locally, fails in CI, when absent."""
    from _ultralytics_weights import cache_dir, verified_path

    def get(name: str) -> Path:
        path = verified_path(name)
        if path is None:
            reason = f'{name} is not in {cache_dir()} (CI job `families` provides it)'
            if os.environ.get('CI'):
                pytest.fail(reason)
            pytest.skip(reason)
        return path

    return get


@pytest.fixture
def project_export_root() -> Path:
    """The bound ``default`` project's ``export_root`` (created). Training
    routes refuse a ``dataset_export_dir`` outside it
    (``export_outside_project``), so fixture exports live here."""
    from src.config.curation_accessors import get_curation_config

    root = Path(get_curation_config().export_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(autouse=True)
def _in_memory_class_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every project's class registry is an in-memory document store unless a test installs its
    own (the registry is an OpenSearch document; unit tests have no cluster).

    ``registry_access`` reaches the registry through the shared client; here it needs none."""
    from _registry_doc_fake import InMemoryRegistries

    from src.clients.curation_opensearch import registry as registry_module

    async def _no_client() -> None:
        return None

    monkeypatch.setattr(registry_module, '_registries', InMemoryRegistries())
    monkeypatch.setattr(
        'src.services.curation.registry_access.make_curation_opensearch', _no_client
    )


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
def _in_memory_leases(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Job leases (``src/clients/job_lease.py``) live in OpenSearch; offline tests run the
    real lease logic against an in-memory store by swapping the client the three I/O
    functions use. A test of the lease client itself opts out with ``real_lease_io``."""
    if request.node.get_closest_marker('real_lease_io') is not None:
        return
    import threading

    from _fake_lease_opensearch import FakeLeaseOpenSearch

    from src.clients import job_lease

    fake = FakeLeaseOpenSearch({}, threading.Lock())
    for name in ('_get', '_create', '_replace'):
        real = getattr(job_lease, name)

        def _bound(client: object, *args: object, _real: object = real, **kw: object) -> object:
            return _real(fake, *args, **kw)  # type: ignore[operator]

        monkeypatch.setattr(job_lease, name, _bound)


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


@pytest.fixture(scope='session', autouse=True)
def _bind_real_keymap_functions() -> None:
    """Import the keymap router before any test patches the keymap service by dotted path.

    The router binds ``get_keymap_doc`` and ``save_keymap_doc`` with ``from ... import`` at
    import time; if the first import happened inside a patched test it would keep the fakes
    for every later test in the process."""
    import src.routers.curation.keymap  # noqa: F401 - bind the real functions first


# Module-level TTL caches (module, attribute). Production keys each by
# project, but every unit test binds the same ``default`` record, so a cache
# one test fills would answer the next test's first request.
_PROCESS_CACHES = (
    ('src.services.curation.field_coverage', '_COVERAGE_CACHE'),
    ('src.services.curation.diverse_order', '_ORDER_CACHE'),
    ('src.services.curation.clustering.outliers', '_CACHE'),
    ('src.clients.curation_opensearch.settings_doc', '_settings_cache'),
    ('src.clients.curation_opensearch.ensure_overlay_fields', '_INNER_RESULT_WINDOWS'),
    ('src.routers.curation.regions_fp', '_suspected_fp_cache'),
    ('src.services.curation.eval_datasets', '_CACHE'),
    ('src.services.projects.stats', '_disk_cache'),
    # Config-store snapshots: one ConfigStore per project, keyed by
    # slug in a module-level dict -- every test binds the same ``default``
    # project, so a store one test mutates would leak into the next.
    ('src.services.config_store.store', '_STORES'),
    # The global (non-project-scoped) config store singleton (M3) -- its
    # own cache dict, never conflated with `_STORES` above.
    ('src.services.config_store.global_store', '_GLOBAL_STORE'),
    # immutable VLM endpoint revision copies fetched from the registry.
    ('src.services.config_store.vlm_snapshot', '_REVISION_CACHE'),
    ('src.services.training.preflight_scan', '_scan_cache'),
    # the labeler factory's LRU (one labeler per endpoint revision + pack).
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
def _detector_config_is_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Profile validation reads a loaded detector's output names from Triton (#362); no test
    reaches a real server, and a stubbed repository lists usable detectors. A test of the
    unusable case passes ``get_model_config`` to ``validate_profile``."""
    from src.services.triton_control import TritonControlService

    async def _config(_self: object, _model: str) -> dict[str, object]:
        return {'output': [{'name': 'output0'}]}

    monkeypatch.setattr(TritonControlService, 'get_model_config', _config)


@pytest.fixture(autouse=True)
def _ephemeral_worker_metrics_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that run ``worker.run()`` start its real /metrics HTTP server. On the
    default fixed port, two of them running at once (xdist workers, or two suites on
    one host) fail with "address already in use". Port 0 asks the OS for a free one."""
    monkeypatch.setenv('OP_REGION_WORKER_METRICS_PORT', '0')


@pytest.fixture
def vlm_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """The deployment has an ``env`` built-in VLM endpoint (``OP_VLM_URL``).
    A project calls it only once it activates it (see ``vlm_env_active``).
    Returns the base URL."""
    url = 'http://vlm.invalid:8000/v1'
    monkeypatch.setenv('OP_VLM_URL', url)
    monkeypatch.setenv('OP_VLM_MODEL', 'test-vlm')
    return url


@pytest.fixture
def vlm_env_active(vlm_env: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """``vlm_env`` plus every project snapshot without an activation doc
    reading as having activated the ``env`` endpoint: a test double for a
    project that opted in, for tests whose fake storage cannot hold the
    activation doc."""
    from curation.vlm_opt_in import project_opts_in_to_env_vlm

    project_opts_in_to_env_vlm(monkeypatch)
    return vlm_env


collect_ignore = [
    'test_full_system.py',
    'test_scrfd_pipeline.py',
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


@pytest.fixture(autouse=True)
def ingest_store_loaded(_bind_default_project: None) -> Iterator[None]:
    """The ``default`` project's config store as a worker that has already loaded it
    and found nothing active: ``project_ingest_profiles()`` resolves to the env
    profiles (it refuses on a store that never loaded or is stale). Production
    always has this because the route dependency refreshes the store; tests that
    hand a route a fake client skip that dependency. ``loaded_at`` is in the future
    so ``ensure_fresh`` does not try to read the fake. Tests of the load/refusal
    path call ``reset_config_stores()`` first."""
    from _curation_app import mark_config_store_loaded

    from src.config.project_context import try_current_project
    from src.services.config_store.store import reset_config_stores

    if try_current_project() is None:
        yield
        return
    reset_config_stores()
    mark_config_store_loaded()
    yield
    reset_config_stores()


@pytest.fixture
def no_other_projects(monkeypatch: pytest.MonkeyPatch) -> None:
    """No other project runs any ingest detector (the unload guard and the models
    list scan every project's active ingest profile)."""

    async def _none(_client: object) -> dict[str, frozenset[str]]:
        return {}

    monkeypatch.setattr('src.services.model_unload_guard.ingest_models_by_project', _none)
