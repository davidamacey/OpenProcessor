"""A project's active ingest profile: resolved in one place, laid over the env
profile, and seen by ingest, the detector info, class seeding, the open-vocabulary
validation, the class-source readers and the usage scan."""

from __future__ import annotations

from typing import Any

import pytest
from _registry_doc_fake import InMemoryRegistry
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._ingest_profile_world import Pool, WorldOpenSearch, body
from src.config import get_curation_config
from src.services.curation import index_bootstrap


URL = '/curation/projects/default/ingest/policy'
PROFILES = '/curation/projects/default/ingest_profiles'


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def labels(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    path = tmp_path / 'labels.txt'
    path.write_text('bird\nboat\n')
    return str(path)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, labels: str) -> Any:
    del labels
    fake = WorldOpenSearch()
    registry = InMemoryRegistry()
    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'env_model')
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', {'default'})
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)
    monkeypatch.setattr('src.routers.curation.class_seed.get_class_registry', lambda: registry)
    monkeypatch.setattr('src.routers.curation.ingest_policy.get_class_registry', lambda: registry)
    monkeypatch.setattr(
        'src.services.curation.registry_access.get_class_registry', lambda: registry
    )
    pool = Pool()
    monkeypatch.setattr('src.app.resources.get_async_triton_pool', lambda: pool)
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        yield client, fake, pool, registry


def _activate(client: TestClient, name: str, model: str, **primary: Any) -> None:
    created = client.post(PROFILES, json={'name': name, 'body': body(model, primary=primary)})
    assert created.status_code == 201, created.text
    assert client.post(f'{PROFILES}/{name}/activate', json={}).status_code == 200


# --- the resolver ------------------------------------------------------------


def test_with_nothing_active_the_env_profile_applies(world: Any) -> None:
    from src.services.curation.ingest_detector import project_ingest_profiles

    client, *_ = world
    client.get(f'{PROFILES}/active')
    resolved = project_ingest_profiles()
    assert resolved.source == 'env'
    assert resolved.primary.detector_model == 'env_model'
    assert resolved.ref is None


def test_a_stored_profile_is_laid_over_the_env_and_never_assigns_by_id(
    world: Any, labels: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation.ingest_detector import project_ingest_profiles

    client, *_ = world
    monkeypatch.setenv('OP_INGEST_PRIMARY_ASSIGNS_CLASS', 'true')
    monkeypatch.setenv('OP_INGEST_PRIMARY_BATCH_LIMIT', '7')
    _activate(client, 'mine', 'birds', version='3', input_size=1280, labels_path=labels)
    resolved = project_ingest_profiles().primary
    assert (resolved.detector_model, resolved.detector_version) == ('birds', '3')
    assert (resolved.input_size, resolved.labels_path) == (1280, labels)
    assert resolved.assigns_class is False
    assert resolved.name == 'item'
    # a field the profile leaves null keeps the deployment value
    assert resolved.batch_limit == 7
    assert project_ingest_profiles().ref == ('mine', 1)


def test_confidence_floor_and_batch_limit_come_from_the_profile(world: Any) -> None:
    from src.services.curation.ingest_detector import project_ingest_profiles

    client, *_ = world
    _activate(client, 'mine', 'birds', confidence_floor=0.8, batch_limit=3)
    primary = project_ingest_profiles().primary
    assert (primary.confidence_floor, primary.batch_limit) == (0.8, 3)


def test_a_stored_secondary_is_resolved_and_absent_otherwise(world: Any) -> None:
    from src.services.curation.ingest_detector import project_ingest_profiles

    client, _, pool, _ = world
    pool.outputs['cls'] = ['output0']
    _activate(client, 'plain', 'birds')
    assert project_ingest_profiles().secondary is None
    created = client.post(
        PROFILES, json={'name': 'dual', 'body': body('birds', secondary={'model': 'cls'})}
    )
    assert created.status_code == 201
    active = client.get(f'{PROFILES}/active').json()['active']
    client.post(f'{PROFILES}/dual/activate', json={'expected_active': active})
    secondary = project_ingest_profiles().secondary
    assert secondary is not None
    assert (secondary.detector_model, secondary.name) == ('cls', 'secondary')


def test_deactivating_returns_to_the_env_profile(world: Any) -> None:
    from src.services.curation.ingest_detector import project_ingest_profiles

    client, *_ = world
    _activate(client, 'mine', 'birds')
    active = client.get(f'{PROFILES}/active').json()['active']
    client.post(f'{PROFILES}/deactivate', json={'expected_active': active})
    assert project_ingest_profiles().primary.detector_model == 'env_model'


def test_an_activation_whose_revision_cannot_be_read_fails_closed(world: Any) -> None:
    import dataclasses

    from src.services.config_store import get_config_store
    from src.services.curation.ingest_detector import (
        IngestProfileUnresolvedError,
        project_ingest_profiles,
    )

    client, *_ = world
    _activate(client, 'mine', 'birds')
    store = get_config_store()
    store.current = dataclasses.replace(store.current, active_ingest_profile_body=None)
    with pytest.raises(IngestProfileUnresolvedError):
        project_ingest_profiles()


@pytest.mark.unbound
def test_an_unbound_project_raises_instead_of_returning_the_env_profile() -> None:
    from src.config.project_context import ProjectNotBound
    from src.services.curation.ingest_detector import project_ingest_profiles

    with pytest.raises(ProjectNotBound):
        project_ingest_profiles()


@pytest.mark.asyncio
async def test_another_workers_activation_is_seen_by_a_fresh_store(world: Any) -> None:
    from src.services.config_store.store import reset_config_stores
    from src.services.curation.ingest_detector import refreshed_ingest_profiles

    client, fake, *_ = world
    _activate(client, 'mine', 'birds')
    reset_config_stores()
    resolved = await refreshed_ingest_profiles(fake)
    assert resolved.primary.detector_model == 'birds'


# --- consumers ---------------------------------------------------------------


def test_the_project_profile_drives_the_info_and_class_seeding(world: Any, labels: str) -> None:
    client, _, _, registry = world
    info = client.get('/curation/projects/default/ingest/config').json()['detector']
    assert info['model'] == 'env_model'
    _activate(client, 'mine', 'birds', labels_path=labels)

    detector = client.get('/curation/projects/default/ingest/config').json()['detector']
    assert detector['model'] == 'birds'
    assert detector['assigns_class'] is False
    assert [label['slug'] for label in detector['labels']] == ['bird', 'boat']

    seeded = client.post(
        '/curation/projects/default/classes/seed_from_detector', json={'dry_run': False}
    ).json()
    assert [c['name'] for c in seeded['created']] == ['bird', 'boat']
    assert [c.class_name for c in registry.load().classes] == ['bird', 'boat']


@pytest.mark.asyncio
async def test_the_ingest_service_runs_the_active_profile_and_the_policy(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.app.resources as resources_module
    import src.main as main_module
    from src.routers.curation import ingest as ingest_router
    from src.services.curation.ingest_policy import DetectFilter, IngestPolicyBody
    from src.services.curation.ingest_policy_store import put_ingest_policy

    client, fake, pool, registry = world
    _activate(client, 'mine', 'birds')
    active = client.get(f'{PROFILES}/active').json()['active']
    client.post(f'{PROFILES}/deactivate', json={'expected_active': active})
    monkeypatch.setattr(resources_module, 'get_async_triton_pool', object)
    monkeypatch.setattr(main_module.app.state, 'pe_encoder', object(), raising=False)

    plain = await ingest_router.get_ingest_service(fake, registry)
    assert plain.profile.detector_model == 'env_model'

    await put_ingest_policy(
        IngestPolicyBody(detect=DetectFilter(class_resolution='by_name')),
        expected_revision=0,
    )
    monkeypatch.setattr(resources_module, 'get_async_triton_pool', lambda: pool)
    assert client.post(f'{PROFILES}/mine/activate', json={}).status_code == 200
    monkeypatch.setattr(resources_module, 'get_async_triton_pool', object)
    mine = await ingest_router.get_ingest_service(fake, registry)
    assert mine.profile.detector_model == 'birds'
    assert mine.detector.profile is mine.profile
    assert mine.policy.detect.class_resolution == 'by_name'


@pytest.mark.asyncio
async def test_the_ingest_service_refuses_when_the_active_profile_is_unreadable(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    import src.app.resources as resources_module
    import src.main as main_module
    from src.core.exceptions import RequestRejectedError
    from src.routers.curation import ingest as ingest_router
    from src.services.config_store import get_config_store

    client, fake, _, registry = world
    _activate(client, 'mine', 'birds')
    monkeypatch.setattr(resources_module, 'get_async_triton_pool', object)
    monkeypatch.setattr(main_module.app.state, 'pe_encoder', object(), raising=False)
    store = get_config_store()
    store.current = dataclasses.replace(
        store.current, active_ingest_profile_body=None, loaded_at=store.current.loaded_at + 1e9
    )
    with pytest.raises(RequestRejectedError) as info:
        await ingest_router.get_ingest_service(fake, registry)
    assert info.value.status_code == 503


@pytest.mark.asyncio
async def test_open_vocab_validation_checks_against_the_project_detector(world: Any) -> None:
    from src.services.config_store.open_vocab_inputs import validation_inputs

    client, *_ = world
    _activate(client, 'mine', 'birds')
    seen: list[str] = []

    def _names(model: str) -> dict[int, str]:
        seen.append(model)
        return {0: 'bird'}

    from src.utils import class_names

    original = class_names.get_class_names
    class_names.get_class_names = _names  # type: ignore[assignment]
    try:
        inputs = await validation_inputs()
    finally:
        class_names.get_class_names = original
    assert seen == ['birds']
    assert inputs['detector_class_names'] == frozenset({'bird'})


def test_the_class_source_catalog_names_the_project_model(world: Any) -> None:
    from src.services.curation.class_sources import class_source_catalog

    client, *_ = world
    _activate(client, 'mine', 'birds')
    labels = {e['id']: e['label'] for e in class_source_catalog()}
    assert labels['item_proposal'] == 'Proposed by birds'


def test_items_from_the_project_profile_get_their_ingest_class_source(world: Any) -> None:
    from src.services.curation.empty_vlm_answer_repair import ingest_class_source
    from src.services.curation.ingest_detector import project_ingest_profiles

    client, *_ = world
    doc = {'class_detector': 'birds', 'class_id': None}
    client.get(f'{PROFILES}/active')  # loads the store: nothing active yet
    assert ingest_class_source(doc, project_ingest_profiles()) is None
    _activate(client, 'mine', 'birds')
    assert ingest_class_source(doc, project_ingest_profiles()) == 'item_proposal'


# --- usage -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_model_a_project_runs_as_its_ingest_detector_counts_as_in_use(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.config_store import project_usage

    client, fake, pool, _ = world
    pool.outputs['cls'] = ['output0']
    client.post(PROFILES, json={'name': 'dual', 'body': body('birds', secondary={'model': 'cls'})})
    client.post(PROFILES, json={'name': 'idle', 'body': body('unused')})

    async def one_project(read: Any) -> dict[str, Any]:
        return {'alpha': await read(get_curation_config().configs_index)}

    monkeypatch.setattr(project_usage, 'read_each_project', one_project)
    # saved but never activated: not in use
    assert await project_usage.ingest_detector_users(fake, 'birds') == []
    client.post(f'{PROFILES}/dual/activate', json={})
    assert await project_usage.ingest_detector_users(fake, 'birds') == ['alpha']
    assert await project_usage.ingest_detector_users(fake, 'cls') == ['alpha']
    assert await project_usage.ingest_detector_users(fake, 'unused') == []
    assert ('alpha', 'ingest detector') in await project_usage.active_detector_users(fake, 'birds')


def test_deleting_the_model_the_active_profile_names_is_refused(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock

    from src.services.training.triton_promote import UnloadResult

    client, *_ = world
    unload = AsyncMock(return_value=UnloadResult('wheel_det', True, True))
    monkeypatch.setattr('src.routers.curation.models.unload_triton_model', unload)
    _activate(client, 'mine', 'wheel_det')

    refused = client.delete('/curation/projects/default/models/wheel_det')
    assert refused.status_code == 409, refused.text
    assert refused.json()['detail']['error'] == 'detector_in_use'
    unload.assert_not_awaited()

    forced = client.delete('/curation/projects/default/models/wheel_det', params={'force': 'true'})
    assert forced.status_code == 200, forced.text
    unload.assert_awaited_once()


# --- the policy no longer owns the detector ----------------------------------


def test_the_policy_rejects_a_detector_and_publishes_none(world: Any) -> None:
    from src.services.curation.ingest_policy import IngestPolicyBody

    client, *_ = world
    assert 'detector' not in IngestPolicyBody.model_fields
    put = client.put(URL, json={'expected_revision': 0, 'detector': {'model': 'birds'}})
    assert put.status_code == 422
    assert 'detector' not in client.get(URL).json()


def test_put_stores_the_clustering_switch(world: Any) -> None:
    client, *_ = world
    client.get(f'{PROFILES}/active')  # a worker that has loaded its store
    body_ = {'expected_revision': 0, 'clustering': {'automatic': False}}
    put = client.put(URL, json=body_)
    assert put.status_code == 200, put.text
    assert client.get(URL).json()['clustering'] == {'automatic': False}


@pytest.mark.asyncio
async def test_a_stored_policy_detector_is_reset_with_a_logged_message(
    world: Any, memory_settings: Any
) -> None:
    from structlog.testing import capture_logs

    from src.services.curation.ingest_policy import IngestPolicyBody
    from src.services.curation.ingest_policy_store import get_ingest_policy, put_ingest_policy

    stored = {
        'revision': 4,
        'detect': {'min_confidence': 0.3},
        'detector': {'model': 'legacy', 'version': '1'},
    }
    memory_settings.policies['ingest_policy'] = stored

    with capture_logs() as logs:
        policy = await get_ingest_policy()
    assert policy.revision == 4
    assert policy.detect.min_confidence == 0.3
    reset = [e for e in logs if e['event'] == 'retired_policy_field_reset']
    assert reset
    assert reset[0]['field'] == 'detector'
    assert 'ingest profile' in reset[0]['use']

    plain = IngestPolicyBody(**policy.model_dump(exclude={'revision'}))
    await put_ingest_policy(plain, expected_revision=4)
    after = memory_settings.policies['ingest_policy']
    assert 'detector' not in after


def test_an_older_revision_can_be_pinned_by_activation(world: Any) -> None:
    client, *_ = world
    _activate(client, 'mine', 'birds')
    saved = client.put(f'{PROFILES}/mine', json={'expected_revision': 1, 'body': body('boats')})
    assert saved.status_code == 200, saved.text
    assert saved.json()['revision'] == 2

    active = client.get(f'{PROFILES}/active').json()['active']
    assert (
        client.post(
            f'{PROFILES}/mine/activate', json={'revision': 2, 'expected_active': active}
        ).status_code
        == 200
    )
    active = client.get(f'{PROFILES}/active').json()['active']
    pinned = client.post(
        f'{PROFILES}/mine/activate', json={'revision': 1, 'expected_active': active}
    )
    assert pinned.status_code == 200, pinned.text
    assert client.get(f'{PROFILES}/active').json()['active']['revision'] == 1
    assert (
        client.get('/curation/projects/default/ingest/config').json()['detector']['model']
        == 'birds'
    )
