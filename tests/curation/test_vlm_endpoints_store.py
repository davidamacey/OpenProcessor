"""The endpoint registry in ``op_global_configs`` (W9.2): revisions are
immutable and never reused, a probe belongs to the body it tested, every
write is announced, and nothing about it is stored per project."""

from __future__ import annotations

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.conftest import good_probe
from src.config.projects import ProjectRecord  # noqa: F401 - documents the bound-project contract
from src.services.config_store import get_global_config_store
from src.services.config_store.global_store import global_configs_index, reset_global_config_store
from src.services.config_store.index import RevisionConflictError
from src.services.config_store.vlm_endpoints import (
    delete_endpoint,
    list_revisions,
    record_probe,
    save_endpoint,
    set_local_desired,
)
from src.services.config_store.vlm_snapshot import fetch_revision
from src.services.curation import event_hub
from src.services.labeling.vlm_catalog import load_catalog
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody


pytestmark = pytest.mark.unbound


def _body(**over: object) -> VlmEndpointBody:
    return VlmEndpointBody(**{'base_url': 'http://vlm:8000/v1', 'model': 'm', **over})  # type: ignore[arg-type]


@pytest.fixture
def client() -> FakeConfigOpenSearch:
    reset_global_config_store()
    return FakeConfigOpenSearch()


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    seen: list[dict] = []
    monkeypatch.setattr(event_hub, '_HUB', None)
    hub = event_hub.get_event_hub()
    real = hub._dispatch

    def spy(event: dict) -> None:
        seen.append(dict(event))
        real(event)

    monkeypatch.setattr(hub, '_dispatch', spy)
    return seen


@pytest.mark.asyncio
async def test_saving_creates_revisions_and_immutable_copies(client) -> None:
    first = await save_endpoint(client, name='a', body=_body(), expected_revision=None)
    assert first.revision == 1
    second = await save_endpoint(
        client, name='a', body=_body(model='m2'), expected_revision=1, description='second'
    )
    assert second.revision == 2
    assert second.description == 'second'
    revisions = await list_revisions(client, 'a')
    assert [r['revision'] for r in revisions] == [2, 1]  # newest first
    old = await fetch_revision(client, 'a', 1)
    assert old is not None
    assert old.body['model'] == 'm'
    current = await fetch_revision(client, 'a', 2)
    assert current is not None
    assert current.body['model'] == 'm2'
    assert await fetch_revision(client, 'a', 3) is None


@pytest.mark.asyncio
async def test_the_stored_body_never_carries_a_key(client) -> None:
    body = _body(api_key_ref='secret:vendor')
    await save_endpoint(client, name='a', body=body, expected_revision=None)
    docs = client._docs[global_configs_index()]
    stored = docs['vlm:a']['_source']['body']
    assert stored['api_key_ref'] == 'secret:vendor'
    assert set(stored) == set(VlmEndpointBody.model_fields)


@pytest.mark.asyncio
async def test_a_stale_or_missing_expected_revision_is_a_conflict(client) -> None:
    await save_endpoint(client, name='a', body=_body(), expected_revision=None)
    with pytest.raises(RevisionConflictError):
        await save_endpoint(client, name='a', body=_body(), expected_revision=None)  # exists
    await save_endpoint(client, name='a', body=_body(model='x'), expected_revision=1)
    with pytest.raises(RevisionConflictError):
        await save_endpoint(client, name='a', body=_body(model='y'), expected_revision=1)  # stale
    with pytest.raises(RevisionConflictError):
        await save_endpoint(client, name='ghost', body=_body(), expected_revision=3)


@pytest.mark.asyncio
async def test_a_deleted_name_never_reuses_a_revision_number(client) -> None:
    """A project may still be pinned to ``name@1``. Re-creating the name
    must not overwrite that immutable copy."""
    await save_endpoint(client, name='a', body=_body(model='original'), expected_revision=None)
    await delete_endpoint(client, name='a', expected_revision=1)
    survivor = await fetch_revision(client, 'a', 1)
    assert survivor is not None
    assert survivor.body['model'] == 'original'

    again = await save_endpoint(
        client, name='a', body=_body(model='reborn'), expected_revision=None
    )
    assert again.revision == 2
    still = await fetch_revision(client, 'a', 1)
    assert still is not None
    assert still.body['model'] == 'original'


@pytest.mark.asyncio
async def test_deleting_needs_the_current_revision(client) -> None:
    await save_endpoint(client, name='a', body=_body(), expected_revision=None)
    await save_endpoint(client, name='a', body=_body(model='x'), expected_revision=1)
    with pytest.raises(RevisionConflictError):
        await delete_endpoint(client, name='a', expected_revision=1)
    await delete_endpoint(client, name='a', expected_revision=2)
    assert 'a' not in get_global_config_store().current.vlm_endpoints


@pytest.mark.asyncio
async def test_a_probe_belongs_to_the_body_it_tested(client) -> None:
    body = _body()
    await save_endpoint(client, name='a', body=body, expected_revision=None)
    await record_probe(client, name='a', revision=1, body=body, record=good_probe(root='org/r'))
    snapshot = get_global_config_store().current

    from src.services.labeling.vlm_endpoints import stored_endpoint

    def endpoint_at(revision_body: VlmEndpointBody, revision: int):
        item = snapshot.vlm_endpoints['a'].__class__(
            kind='vlm_endpoint',
            name='a',
            revision=revision,
            body=revision_body.model_dump(),
            description='',
            created_at=None,
            updated_at=None,
            cloned_from=None,
        )
        return stored_endpoint(item, snapshot.vlm_probes)

    assert endpoint_at(body, 1).status == 'ready'
    # a probe is of ONE revision: the same body at another revision has none
    assert endpoint_at(body, 2).status == 'unprobed'
    # fields the probe exercised: a change invalidates it
    for changed in (
        _body(base_url='http://other:8000/v1'),
        _body(model='other'),
        _body(api_key_ref='secret:vendor'),
        _body(max_images_per_call=2),
    ):
        assert endpoint_at(changed, 1).status == 'unprobed', changed
        assert endpoint_at(changed, 1).model_id == changed.model  # not the old root
    # fields it did not exercise: a change keeps it
    for unchanged in (
        _body(timeout_s=30.0),
        _body(requests_per_second=5.0),
        _body(json_mode='off'),
    ):
        assert endpoint_at(unchanged, 1).status == 'ready', unchanged


@pytest.mark.asyncio
async def test_probing_one_revision_never_touches_another_revisions_probe(client) -> None:
    """M3: the probe of the running revision survives a probe of a newer one."""
    first, second = _body(), _body(model='newer')
    await save_endpoint(client, name='a', body=first, expected_revision=None)
    await record_probe(client, name='a', revision=1, body=first, record=good_probe(root='org/r1'))
    await save_endpoint(client, name='a', body=second, expected_revision=1)
    await record_probe(client, name='a', revision=2, body=second, record=good_probe(root='org/r2'))
    probes = get_global_config_store().current.vlm_probes
    assert probes['a@1']['record']['root'] == 'org/r1'
    assert probes['a@2']['record']['root'] == 'org/r2'


@pytest.mark.asyncio
async def test_a_probe_bumps_the_config_revision_but_not_the_endpoint_revision(client) -> None:
    body = _body()
    await save_endpoint(client, name='a', body=body, expected_revision=None)
    store = get_global_config_store()
    before = store.current.config_revision
    await record_probe(client, name='a', revision=1, body=body, record=good_probe())
    assert store.current.config_revision == before + 1
    assert store.current.vlm_endpoints['a'].revision == 1
    assert [r['revision'] for r in await list_revisions(client, 'a')] == [1]


@pytest.mark.asyncio
async def test_deleting_drops_the_probe_too(client) -> None:
    body = _body()
    await save_endpoint(client, name='a', body=body, expected_revision=None)
    await record_probe(client, name='a', revision=1, body=body, record=good_probe())
    await delete_endpoint(client, name='a', expected_revision=1)
    assert not get_global_config_store().current.vlm_probes


@pytest.mark.asyncio
async def test_every_write_is_announced_on_the_global_stream(client, events) -> None:
    body = _body()
    await save_endpoint(client, name='a', body=body, expected_revision=None)
    await record_probe(client, name='a', revision=1, body=body, record=good_probe())
    await set_local_desired(client, catalog_id=load_catalog()[0].id)
    await delete_endpoint(client, name='a', expected_revision=1)
    announced = [e for e in events if e['type'] == 'vlm.changed']
    assert len(announced) == 4
    for event in announced:
        assert event['project'] is None
        assert event['topic'] == 'vlm'
    assert {e['axis'] for e in announced} == {'registry', 'local_vlm'}


@pytest.mark.asyncio
async def test_the_registry_is_one_index_shared_by_every_project(client) -> None:
    await save_endpoint(client, name='a', body=_body(), expected_revision=None)
    assert set(client._docs) == {global_configs_index()}
    kinds = {
        doc['_source'].get('doc_type') for doc in client._docs[global_configs_index()].values()
    }
    assert kinds <= {'config', 'revision', 'meta'}


@pytest.mark.asyncio
async def test_a_cold_process_reads_the_same_registry(client) -> None:
    body = _body()
    await save_endpoint(client, name='a', body=body, expected_revision=None)
    await record_probe(client, name='a', revision=1, body=body, record=good_probe(root='org/r'))
    await set_local_desired(client, catalog_id='qwen3-vl-4b')
    reset_global_config_store()
    cold = get_global_config_store()
    await cold.refresh(client)
    assert set(cold.current.vlm_endpoints) == {'a'}
    assert cold.current.vlm_probes['a@1']['record']['root'] == 'org/r'
    assert cold.current.local_vlm_desired is not None
    assert cold.current.local_vlm_desired['catalog_id'] == 'qwen3-vl-4b'


@pytest.mark.asyncio
async def test_the_immutable_revision_cache_is_not_a_second_source_of_truth(client) -> None:
    await save_endpoint(client, name='a', body=_body(model='one'), expected_revision=None)
    first = await fetch_revision(client, 'a', 1)
    assert first is not None
    # the doc is gone from the store: the cached copy is still revision 1's
    del client._docs[global_configs_index()]['vlm:a@1']
    cached = await fetch_revision(client, 'a', 1)
    assert cached is first
    # ...but a fresh process (empty cache) sees the truth: it is gone
    from src.services.config_store.vlm_snapshot import reset_vlm_revision_cache

    reset_vlm_revision_cache()
    assert await fetch_revision(client, 'a', 1) is None
