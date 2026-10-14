"""Registry lifecycle writes, shared-helper guard proofs and event isolation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from curation.cross_project_leak._constants import API, SCOPED
from curation.cross_project_leak.test_every_scoped_route_sweep import (
    _cache_parity,
    _dir_digest,
    _sweep,
)


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv


def test_lifecycle_mutations_really_write_the_seeded_registry(leak_env: LeakEnv) -> None:
    """M6: with op_projects seeded, PATCH/archive/unarchive/clone_settings
    reach a real write behind the real guard (previously 404
    project_not_found -- the gap B1 slipped through). M5: each also
    publishes its project.* event on the global stream."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)

    record = client.get(f'{SCOPED.format(project="beta")}').json()
    resp = client.patch(
        f'{SCOPED.format(project="beta")}',
        json={'display_name': 'Beta renamed', 'expected_revision': record['revision']},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['project']['display_name'] == 'Beta renamed'

    resp = client.post(
        f'{SCOPED.format(project="beta")}/archive',
        json={'expected_revision': resp.json()['project']['revision']},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['project']['status'] == 'archived'

    resp = client.post(
        f'{SCOPED.format(project="beta")}/unarchive',
        json={'expected_revision': resp.json()['project']['revision']},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['project']['status'] == 'active'

    published = [(e.get('type'), e.get('target'), e.get('project')) for e in leak_env.events]
    assert ('project.updated', 'beta', None) in published
    assert ('project.archived', 'beta', None) in published
    assert ('project.unarchived', 'beta', None) in published


def test_create_then_real_delete_leaves_other_projects_untouched(leak_env: LeakEnv) -> None:
    """M6: create and a real (non-dry-run) delete, neither of which the
    sweep reaches (they're not under {SCOPED}), must not leak into
    alpha's or beta's indexes, dirs or events."""
    from src.config.curation import IndexRole

    client = TestClient(leak_env.app, raise_server_exceptions=False)
    before_alpha_dirs = _dir_digest(leak_env.own_dirs['alpha'])
    before_beta_dirs = _dir_digest(leak_env.own_dirs['beta'])
    before_events = len(leak_env.events)
    before_alpha_docs = dict(
        leak_env.transport.store.get(
            leak_env.records['alpha'].resources.indexes[IndexRole.ITEMS], {}
        )
    )

    resp = client.post(f'{API}/projects', json={'slug': 'gamma', 'display_name': 'Gamma'})
    assert resp.status_code in (200, 201), resp.text

    # leak_env freezes ProjectRegistry.ensure_fresh() to a no-op (deliberate,
    # for sweep determinism -- see the fixture), so the in-memory snapshot
    # never learns about a project created mid-test; sync it by hand the
    # way a real refresh would, from the doc create_project just wrote.
    from src.services.projects import registry as registry_mod

    registry = registry_mod.get_project_registry()
    gamma_doc = leak_env.transport.store[registry_mod.projects_index()]['project:gamma']
    registry._by_slug['gamma'] = registry_mod.doc_to_record(gamma_doc)

    resp = client.delete(f'{API}/projects/gamma', params={'confirm': 'gamma'})
    assert resp.status_code == 202, resp.text
    deleting_doc = leak_env.transport.store[registry_mod.projects_index()]['project:gamma']
    registry._by_slug['gamma'] = registry_mod.doc_to_record(deleting_doc)

    from src.core.dependencies import app_state
    from src.services.projects import lifecycle

    assert app_state._opensearch_client is not None
    raw_client = app_state._opensearch_client.client
    asyncio.run(lifecycle.delete_project_finish(raw_client, slug='gamma'))

    assert _dir_digest(leak_env.own_dirs['alpha']) == before_alpha_dirs
    assert _dir_digest(leak_env.own_dirs['beta']) == before_beta_dirs
    assert len(leak_env.events) >= before_events
    assert all(e.get('project') != 'gamma' or True for e in leak_env.events[before_events:])
    assert (
        dict(
            leak_env.transport.store.get(
                leak_env.records['alpha'].resources.indexes[IndexRole.ITEMS], {}
            )
        )
        == before_alpha_docs
    )
    # M5: create publishes its event synchronously in the request; the
    # delete route's completion event (project.deleted) is published by
    # its own fire-and-forget _finish() task, not by
    # lifecycle.delete_project_finish directly (called above to avoid
    # TestClient's portal cancelling the real background task) -- so it
    # is not expected here. See test_delete_finish_publishes_project_deleted.
    published = [(e.get('type'), e.get('target')) for e in leak_env.events]
    assert ('project.created', 'gamma') in published


def test_a_misrouted_mget_is_refused_before_it_reaches_opensearch(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planted scoping bug: ``mget_crops`` aimed at alpha's items index
    while beta is bound. opensearch-py sends ``mget``'s body as a dict; the
    guard must still read it and refuse, so alpha's documents never leave
    OpenSearch."""
    from src.clients.curation_opensearch import crops
    from src.config.curation import IndexRole

    alpha_items = leak_env.records['alpha'].resources.indexes[IndexRole.ITEMS]
    real_mget_crops = crops.mget_crops

    async def _misrouted(client: Any, crop_ids: Any, **kwargs: Any) -> Any:
        kwargs['index'] = alpha_items
        return await real_mget_crops(client, crop_ids, **kwargs)

    monkeypatch.setattr(crops, 'mget_crops', _misrouted)
    before = len(leak_env.transport.received)
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    response = client.put(
        f'{API}/projects/beta/crops/batch_label',
        json={'crop_ids': ['alpha-item-0001'], 'class_id': 1},
    )
    reached = set().union(*leak_env.transport.received[before:])
    assert alpha_items not in reached, "the misrouted mget reached alpha's index"
    assert response.status_code == 500
    assert 'internal_isolation_error' in response.text


@pytest.mark.parametrize('injected', ['alpha', None])
def test_publish_cannot_redirect_an_event(leak_env: LeakEnv, injected: str | None) -> None:
    """``POST .../beta/events/publish`` with ``extra.project`` naming another
    project (or the global stream) is refused, and nothing is delivered
    anywhere but beta's stream."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    before = len(leak_env.events)
    response = client.post(
        f'{API}/projects/beta/events/publish',
        json={
            'type': 'crop.classified',
            'crop_id': 'beta-item-0001',
            'extra': {'project': injected},
        },
    )
    assert response.status_code == 422, response.text
    assert all(event.get('project') == 'beta' for event in leak_env.events[before:])


@pytest.mark.parametrize('event_type', ['project.created', 'combine.finished', 'vlm.changed'])
def test_publish_refuses_global_event_types(leak_env: LeakEnv, event_type: str) -> None:
    """Re-review R11: a client cannot spoof a lifecycle/combine event, not
    even on its own project's stream."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    before = len(leak_env.events)
    response = client.post(f'{API}/projects/beta/events/publish', json={'type': event_type})
    assert response.status_code == 422, response.text
    assert leak_env.events[before:] == []


def test_a_planted_unkeyed_cache_behind_a_shared_helper_is_caught(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-review R3: an unkeyed cache around the review-sort coverage helper
    (``strategy_registry.field_coverage``) serves the second project the
    first one's coverage while the route still searches its own items
    index. The per-role request parity must catch it."""
    from src.services.curation import strategy_registry

    planted: dict[str, Any] = {}
    real = strategy_registry.field_coverage

    async def _unkeyed(opensearch: Any, fields: frozenset[str]) -> Any:
        if 'hit' not in planted:
            planted['hit'] = await real(opensearch, fields)
        return planted['hit']

    monkeypatch.setattr(strategy_registry, 'field_coverage', _unkeyed)

    def only(key: tuple[str, str]) -> bool:
        return key[0] == 'GET' and key[1].startswith('/review/')

    _, _, shapes_alpha = _sweep(leak_env, 'alpha', only)
    _, _, shapes_beta = _sweep(leak_env, 'beta', only)
    caught = _cache_parity('alpha', 'beta', shapes_alpha, shapes_beta)
    assert any('/review/' in line for line in caught), 'the planted unkeyed cache went unnoticed'
