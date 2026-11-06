"""The VLM endpoint registry in PostgreSQL: deployment-wide documents, probes per revision, the
env built-in, the desired local model and a project's activation with its acknowledgement."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from _config_fixtures import make_project

from src.db.repos import config as repo
from src.services.config_store import documents, get_config_store, get_global_config_store
from src.services.config_store.documents import CloneSource
from src.services.config_store.global_store import reset_global_config_store
from src.services.config_store.store import reset_config_stores
from src.services.config_store.vlm_activation import (
    activate_vlm,
    deactivate_vlm,
    rollback_vlm,
    write_vlm_activation,
)
from src.services.config_store.vlm_endpoints import (
    clear_local_desired,
    delete_endpoint,
    record_probe,
    save_endpoint,
    set_local_desired,
)
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody, VlmProbeRecord
from src.services.labeling.vlm_endpoints import env_builtin, probe_fingerprint


pytestmark = pytest.mark.db


@pytest.fixture(scope='module', autouse=True)
def _default_project(pg):
    return make_project(pg, 'default')


@pytest.fixture(autouse=True)
def _fresh_stores(monkeypatch: pytest.MonkeyPatch):
    import src.services.labeling.vlm_url_policy as policy

    monkeypatch.setattr(policy, '_resolve', lambda host: ['172.18.0.9'] if host == 'vlm' else [])
    monkeypatch.setattr(policy, '_docker_gateway_addresses', frozenset)
    policy.reset_policy_caches()
    reset_config_stores()
    reset_global_config_store()
    yield
    reset_config_stores()
    reset_global_config_store()


async def _expected() -> dict[str, Any] | None:
    """The ``expected_active`` of the bound project's VLM, as a client would present it."""
    from src.services.config_store.vlm_activation import expected_active_for

    store = get_config_store()
    await store.refresh()
    return expected_active_for(store.current.active_vlm)


def _body(**over: Any) -> VlmEndpointBody:
    fields: dict[str, Any] = {'base_url': 'http://vlm:8000/v1', 'model': 'm', **over}
    return VlmEndpointBody(**fields)


def _probe(**over: Any) -> VlmProbeRecord:
    fields: dict[str, Any] = {
        'ok': True,
        'probed_at': '2026-09-28T12:00:00+00:00',
        'latency_ms': 12.0,
        'models_listed': ['m'],
        'model_listed': True,
        'root': 'org/real-model',
        'max_model_len': 32768,
        'vision_ok': True,
        'json_mode_supported': True,
        'reasoning_channel': False,
        'image_tokens': 260,
        'max_images_ok': True,
        'issues': [],
    }
    fields.update(over)
    return VlmProbeRecord(**fields)


def _name() -> str:
    return f'ep-{uuid.uuid4().hex[:8]}'


@pytest.mark.asyncio
async def test_a_probe_belongs_to_its_revision_only(pg_env) -> None:
    name = _name()
    first = await save_endpoint(name=name, body=_body(), expected_revision=None)
    await record_probe(name=name, revision=first.revision, body=_body(), record=_probe())
    second = await save_endpoint(
        name=name, body=_body(model='m2'), expected_revision=first.revision
    )
    registry = get_global_config_store()
    await registry.refresh()
    probes = registry.current.vlm_probes
    assert set(probes) >= {f'{name}@1'}
    assert f'{name}@2' not in probes
    assert probes[f'{name}@1']['fingerprint'] == probe_fingerprint(_body())
    assert second.revision == 2
    assert second.last_probe is None


@pytest.mark.asyncio
async def test_the_registry_is_deployment_wide_and_a_stale_save_conflicts(pg_env) -> None:
    name = _name()
    await save_endpoint(name=name, body=_body(), expected_revision=None)
    with pytest.raises(repo.RevisionConflictError):
        await save_endpoint(name=name, body=_body(), expected_revision=None)
    with pytest.raises(repo.RevisionConflictError):
        await save_endpoint(name=name, body=_body(model='x'), expected_revision=7)


@pytest.mark.asyncio
async def test_an_endpoint_a_project_runs_cannot_be_deleted(pg_env) -> None:
    name = _name()
    saved = await save_endpoint(name=name, body=_body(), expected_revision=None)
    await record_probe(name=name, revision=saved.revision, body=_body(), record=_probe())
    await activate_vlm(name=name, revision=None, expected_active=await _expected())
    with pytest.raises(repo.ConfigInUseError):
        await delete_endpoint(name=name, expected_revision=1)
    await deactivate_vlm(expected_active={'name': name, 'revision': 1})
    await delete_endpoint(name=name, expected_revision=1)


@pytest.mark.asyncio
async def test_the_desired_local_model_round_trips_and_clears(pg_env) -> None:
    await set_local_desired(catalog_id='local-model-a')
    registry = get_global_config_store()
    await registry.refresh()
    assert registry.current.local_vlm_desired is not None
    assert registry.current.local_vlm_desired['catalog_id'] == 'local-model-a'
    await set_local_desired(catalog_id='local-model-b')
    await clear_local_desired()
    await registry.refresh()
    assert registry.current.local_vlm_desired is None


@pytest.mark.asyncio
async def test_activating_the_env_builtin_stores_a_system_revision_and_keeps_the_wire_ref(
    pg_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'm')
    env = env_builtin()
    assert env is not None
    result = await write_vlm_activation(env, expected_active=await _expected(), ack_now=False)
    assert result.ref == ('env', None)
    async with documents.tx() as session:
        stored = await repo.get_config(session, project_id=None, kind='vlm_endpoint', name='env')
        assert stored is not None
        revisions = await repo.list_revisions(session, stored.id)
    assert [(r.created_by, r.description) for r in revisions] == [
        ('system', 'From OP_VLM_URL / OP_VLM_MODEL')
    ]
    # the reserved env document is not offered as a saved endpoint
    registry = get_global_config_store()
    await registry.refresh()
    assert 'env' not in registry.current.vlm_endpoints
    # a client holding the wire ref (no revision) can still switch it off
    store = get_config_store()
    await store.refresh()
    assert store.current.active_vlm == ('env', None)
    await write_vlm_activation(
        None, expected_active={'name': 'env', 'revision': None}, ack_now=False
    )
    # re-activating the unchanged env body reuses the stored revision
    await write_vlm_activation(env, expected_active=await _expected(), ack_now=False)
    async with documents.tx() as session:
        assert (
            stored.id
            == (await repo.get_config(session, project_id=None, kind='vlm_endpoint', name='env')).id
        )  # type: ignore[union-attr]
        assert len(await repo.list_revisions(session, stored.id)) == 1


@pytest.mark.asyncio
async def test_rollback_returns_to_the_previous_endpoint(pg_env) -> None:
    a, b = _name(), _name()
    for name in (a, b):
        saved = await save_endpoint(name=name, body=_body(), expected_revision=None)
        await record_probe(name=name, revision=saved.revision, body=_body(), record=_probe())
    await activate_vlm(name=a, revision=None, expected_active=await _expected())
    await activate_vlm(name=b, revision=None, expected_active={'name': a, 'revision': 1})
    result, _gate = await rollback_vlm(expected_active={'name': b, 'revision': 1})
    assert result.ref == (a, 1)


@pytest.mark.asyncio
async def test_an_acknowledgement_is_recorded_with_the_activation(pg_env) -> None:
    name = _name()
    body = _body(base_url='https://api.example.com/v1')
    saved = await save_endpoint(name=name, body=body, expected_revision=None)
    await record_probe(name=name, revision=saved.revision, body=body, record=_probe())
    await write_vlm_activation(saved, expected_active=await _expected(), ack_now=True)
    assert saved.ref in await documents.get_acked_refs()
    store = get_config_store()
    await store.refresh()
    assert saved.ref in store.current.acked_refs
    assert store.current.active_vlm_ack_at is not None


@pytest.mark.asyncio
async def test_a_clone_of_an_endpoint_keeps_its_source_revision(pg_env) -> None:
    source, target = _name(), _name()
    await save_endpoint(name=source, body=_body(), expected_revision=None)
    clone = await save_endpoint(
        name=target,
        body=_body(),
        expected_revision=None,
        cloned_from=CloneSource(None, 'vlm_endpoint', source, 1),
    )
    assert clone.cloned_from == f'{source}@1'


def _global_vlm_events(pg) -> list[tuple[str, dict[str, Any]]]:
    import psycopg

    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        return [
            (row[0], row[1])
            for row in conn.execute(
                "SELECT type, payload FROM events WHERE project_id IS NULL AND type = 'vlm.changed' "
                'ORDER BY id'
            )
        ]


@pytest.mark.asyncio
async def test_every_write_is_announced_on_the_global_stream(pg_env) -> None:
    from src.services.labeling.vlm_catalog import load_catalog

    before = len(_global_vlm_events(pg_env))
    name = _name()
    body = _body()
    saved = await save_endpoint(name=name, body=body, expected_revision=None)
    await record_probe(name=name, revision=saved.revision, body=body, record=_probe())
    await set_local_desired(catalog_id=load_catalog()[0].id)
    await delete_endpoint(name=name, expected_revision=saved.revision)
    announced = _global_vlm_events(pg_env)[before:]
    assert len(announced) == 4
    assert all(payload['topic'] == 'vlm' for _, payload in announced)
    assert {payload['axis'] for _, payload in announced} == {'registry', 'local_vlm'}
