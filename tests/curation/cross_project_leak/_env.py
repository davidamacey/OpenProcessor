"""The three-project environment (app, records, fake transport) the isolation tests share."""

from __future__ import annotations

import subprocess  # nosec B404 - only patched to refuse, never called
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
import pytest
from integration.ingest_fakes import FakePEEncoder, FakeTritonPool
from opensearchpy import AsyncOpenSearch

from curation.cross_project_leak._seeded_files import _docs, _seed_files, _vec
from curation.cross_project_leak._seeders import _seed_vlm
from curation.cross_project_leak._transport import (
    _FAKE_VLM_HOST,
    LeakEnv,
    _fake_vlm_reply,
    _FakeTransport,
    _RecordingTransport,
)


if TYPE_CHECKING:
    from pathlib import Path


def _record(slug: str, resources: Any) -> Any:
    from src.config.projects import ProjectRecord

    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug.title(),
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


class _LeakPEEncoder(FakePEEncoder):
    """The ingest fake, at the seed data's embedding width (so the
    clustering and scoring routes see one consistent pool)."""

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        return np.array([_vec(100 + i) for i in range(len(crops))], dtype=np.float32)

    async def embed_whole_frame_image(self, img: Any) -> np.ndarray:
        self.whole_frame_sizes.append(img.size)
        return np.array(_vec(200), dtype=np.float32)


@pytest.fixture
def leak_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_region_profile: None
) -> Any:
    """The real app, three seeded projects, a recording layer above the
    real guard, and no way out of the process (no network, no subprocess)."""
    import src.config.curation as curation_config_mod
    from src.clients.curation_opensearch import registry as curation_registry, settings_doc
    from src.config.curation import IndexRole, base_curation_config
    from src.config.projects import new_project_record, resources_for_new
    from src.core.dependencies import app_state, get_async_triton
    from src.services.curation import event_hub, index_bootstrap
    from src.services.projects import guard, registry as registry_mod
    from src.services.projects.registry import ProjectRegistry

    for name, sub in {
        'STATE_DIR': 'state',
        'CROP_CACHE_DIR': 'crop_cache',
        'TRAIN_JOBS_DIR': 'jobs/train',
        'AUTO_LABEL_STATE_DIR': 'jobs/auto_label',
        'BAKEOFF_JOBS_DIR': 'state/bakeoff_jobs',
        'PROJECTS_DATA_ROOT': 'projects',
        'PROBE_JOBS_DIR': 'jobs/probe',
        'SELECT_JOBS_DIR': 'jobs/select',
        'SCORES_STATE_DIR': 'jobs/scores',
        'VIZ_JOBS_DIR': 'jobs/viz',
        'REGION_DRAIN_STATE_DIR': 'jobs/region_drain',
        'HEARTBEAT_DIR': 'state/heartbeats',
        'TRAIN_RUNS_ROOT': 'state/training_runs',
        'SOURCE_ROOT': 'images',
        'TRITON_MODEL_REPO': 'models',
    }.items():
        monkeypatch.setenv(f'OP_{name}', str(tmp_path / sub))
    monkeypatch.setenv('OP_S3_ALLOWED', 'sweep-allowed-bucket')
    monkeypatch.setenv('OP_EVENT_BUS', 'process')
    # `ensure_fresh` skips the config read while the snapshot is under a
    # second old, so how many `configs` operations a route issues would
    # depend on how fast the previous route ran, and the sweep compares
    # those counts between projects. Pin the window to zero: every route
    # reads, for every project, in every run.
    from src.services.config_store.store import ConfigStore

    real_ensure_fresh = ConfigStore.ensure_fresh

    async def ensure_fresh_always(self: Any, client: Any, max_age_s: float = 1.0) -> Any:
        return await real_ensure_fresh(self, client, max_age_s=0.0)

    monkeypatch.setattr(ConfigStore, 'ensure_fresh', ensure_fresh_always)
    for flag in ('SCORES_ENABLED', 'SELECT_DIVERSE_ENABLED', 'VIZ_PROJECTION_ENABLED'):
        monkeypatch.setenv(f'OP_{flag}', '1')
    # Diverse selection over the job path (the one that writes); the
    # inline path answers from memory.
    monkeypatch.setenv('OP_SELECT_SYNC_MAX_OPS', '1')
    monkeypatch.setenv('OP_REGION_FIELD_EMBEDDING', 'pe_embedding')
    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'fake_item_detector')
    # The seed-from-detector route reads the detector's label list.
    (tmp_path / 'detector_labels.txt').write_text('seed probe\n')
    monkeypatch.setenv('OP_INGEST_PRIMARY_LABELS_PATH', str(tmp_path / 'detector_labels.txt'))
    # so a segmenter-only region-profile body (_region_profile_body())
    # never trips no_candidate_source in the sweep -- the segmenter health
    # probe itself still fails closed (network is disabled here), which is
    # exactly the segmenter_unreachable warning/activation-error path
    # those routes' bodies exercise (force=true on activate bypasses it).
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter-disabled-in-leak-test:8000')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    monkeypatch.setattr(curation_registry, '_registries', {})
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', set())
    # Keyed by index (audited), but its 5 s TTL would let wall-clock time
    # change which index roles a route reaches between the two passes.
    monkeypatch.setattr(settings_doc, '_SETTINGS_CACHE_TTL_SECONDS', 0.0)
    # A UMAP fit the fixture's small pool supports (the defaults need
    # more points than the fixture seeds).
    from src.services.curation.clustering import pool_size

    monkeypatch.setattr(pool_size, 'UMAP_N_COMPONENTS', 2)
    monkeypatch.setattr(pool_size, 'UMAP_N_NEIGHBORS', 3)
    # The auto-label module resolves its state dir per bound project
    # (``_state_dir()`` -> ``get_curation_config().autolabel_dir``, itself
    # ``OP_AUTO_LABEL_STATE_DIR`` + ``/projects/<slug>``) -- keep it inside
    # tmp_path so the sweep never touches the host's /jobs. Must be set
    # before the project records below are built from this env.
    al_dir = tmp_path / 'jobs' / 'auto_label'
    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(al_dir))
    # The (P2-owned) auto-label module resolves its state dir fresh per
    # bound project on every call (`_state_dir()` -> the bound project's
    # own `autolabel_dir`, a PROJECT_SCOPED_FIELDS entry) -- there is no
    # import-time module constant left to patch. Redirect the env var
    # `resources_for_new` reads instead, so the sweep never touches the
    # host's /jobs and each project's own nested dir
    # (`<this>/projects/<slug>`, `default` included per P1R §6.1/D-A) is
    # kept inside tmp_path.
    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(tmp_path / 'jobs' / 'auto_label'))

    base = base_curation_config()
    records = {'default': new_project_record('default', base)}
    for slug in ('alpha', 'beta'):
        records[slug] = _record(slug, resources_for_new(slug, base))

    store: dict[str, dict[str, dict[str, Any]]] = {}
    for slug, record in records.items():
        for role_value, docs in _docs(slug).items():
            store[record.resources.indexes[IndexRole(role_value)]] = docs
    own_dirs = {slug: _seed_files(record, tmp_path) for slug, record in records.items()}

    # M6: seed the shared op_projects registry doc for every project, so
    # the lifecycle mutations below (PATCH/archive/unarchive/clone_settings)
    # really reach a write instead of 404ing project_not_found before ever
    # touching the guard -- which is why B1 (every lifecycle write refused
    # 500 by the real guard) slipped through a sweep that runs behind the
    # real guard. ``get_record_with_seq``'s doc id is 'project:<slug>'
    # (registry.py's ``_project_doc_id``); revision starts at 1, matching
    # ``registry._revision`` below.
    from src.services.projects.registry import projects_index, record_to_doc

    store[projects_index()] = {
        f'project:{slug}': record_to_doc(record) for slug, record in records.items()
    }

    registry = ProjectRegistry(lambda: None)
    registry._by_slug = dict(records)
    registry._revision = 1

    async def _fresh(self: Any) -> None:
        """A real (if simplified) refresh instead of a hard no-op: syncs
        ``_by_slug`` from the seeded ``op_projects`` store so a project
        created mid-sweep (e.g. B2's create-then-write-own-indexes path)
        is visible to the guard on its very next check, the way a real
        ``ensure_fresh`` would pick it up after B2's ``refresh='wait_for'``.
        Frozen otherwise: no revision-counter churn, so the sweep's own
        three seeded projects never move under it."""
        from src.services.projects.registry import doc_to_record

        docs = fake.store.get(registry_mod.projects_index(), {})
        for doc_id, doc in docs.items():
            if doc_id.startswith('project:'):
                self._by_slug[doc['slug']] = doc_to_record(doc)

    monkeypatch.setattr(ProjectRegistry, 'ensure_fresh', _fresh)
    # P3F item 3 (B2(a) residual): create_project now also calls
    # refresh_strict() (raises instead of swallowing). This registry's
    # client_factory is `lambda: None` -- fine for the patched
    # ensure_fresh above (never touches it), but the real
    # refresh_strict would call client.get(...) on that None and blow up
    # with an AttributeError. Give it the same sync-from-store behavior.
    monkeypatch.setattr(ProjectRegistry, 'refresh_strict', _fresh)
    registry_mod.set_project_registry(registry)

    accesses: list[tuple[str | None, str, str, str]] = []
    fake = _FakeTransport(store)
    raw = AsyncOpenSearch(hosts=['http://127.0.0.1:9'])
    raw.transport = fake  # type: ignore[assignment]
    guard.install_project_guard(raw, registry)
    raw.transport = _RecordingTransport(raw.transport, accesses)  # type: ignore[assignment]

    from src.clients.opensearch.client import OpenSearchClient

    wrapper = OpenSearchClient(hosts=['http://127.0.0.1:9'])
    wrapper.client = raw
    monkeypatch.setattr(app_state, '_opensearch_client', wrapper)

    env = LeakEnv(
        app=None,
        records=records,
        accesses=accesses,
        transport=fake,
        own_dirs=own_dirs,
        root=tmp_path,
    )
    for slug in records:
        _seed_vlm(env, slug)
    monkeypatch.setattr(event_hub, '_HUB', None)
    hub = event_hub.get_event_hub()
    real_dispatch = hub._dispatch

    def _recording_dispatch(event: dict[str, Any]) -> None:
        env.events.append(dict(event))
        real_dispatch(event)

    monkeypatch.setattr(hub, '_dispatch', _recording_dispatch)

    def _no_network(*_a: Any, **_k: Any) -> Any:
        raise httpx.ConnectError('network disabled in the leak test')

    async def _only_the_fake_vlm(_self: Any, request: httpx.Request) -> httpx.Response:
        if request.url.host == _FAKE_VLM_HOST:
            return _fake_vlm_reply(request)
        return _no_network()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', _only_the_fake_vlm)
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', _no_network)
    # Every VLM route talks to the in-process fake VLM (a real
    # OpenAI-shaped reply), so the labeling/verify routes really write.
    # Every VLM route talks to the in-process fake VLM (a real
    # OpenAI-shaped reply) through the ``env`` built-in endpoint, so the
    # labeling/verify routes really write.
    from src.services.labeling import vlm_url_policy

    monkeypatch.setattr(
        vlm_url_policy, '_resolve', lambda host: ['10.20.30.40'] if host == _FAKE_VLM_HOST else []
    )
    monkeypatch.setenv('OP_VLM_URL', f'http://{_FAKE_VLM_HOST}/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'leak-test-vlm')

    def _no_subprocess(*_a: Any, **_k: Any) -> Any:
        raise OSError('subprocesses disabled in the leak test')

    monkeypatch.setattr(subprocess, 'Popen', _no_subprocess)
    monkeypatch.setattr(subprocess, 'run', _no_subprocess)

    import asyncio

    async def _no_exec(*_a: Any, **_k: Any) -> Any:
        raise OSError('subprocesses disabled in the leak test')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', _no_exec)
    monkeypatch.setattr(asyncio, 'create_subprocess_shell', _no_exec)

    class _DeadTriton:
        async def is_server_live(self) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            raise ConnectionError(f'triton disabled in the leak test ({name})')

    import src.app.resources as resources_module
    from src.main import app

    env.app = app
    app.dependency_overrides[get_async_triton] = lambda: _DeadTriton()  # noqa: PLW0108  # FastAPI parses a class here as a dependency with parameters
    # Ingest runs end to end: a fake detector pool and PE encoder at the
    # service boundary (the same fakes the ingest integration tests use).
    monkeypatch.setattr(resources_module, 'get_async_triton_pool', FakeTritonPool)
    had_encoder = hasattr(app.state, 'pe_encoder')
    previous_encoder = getattr(app.state, 'pe_encoder', None)
    app.state.pe_encoder = _LeakPEEncoder()
    try:
        yield env
    finally:
        app.dependency_overrides.pop(get_async_triton, None)
        registry_mod.set_project_registry(None)
        if had_encoder:
            app.state.pe_encoder = previous_encoder
        else:
            del app.state.pe_encoder
