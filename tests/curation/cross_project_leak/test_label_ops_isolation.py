"""Label operations belong to the project that made them: its list, its rows, its indexes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from curation.cross_project_leak._constants import SCOPED


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv


pytestmark = pytest.mark.db


def _stored_op_ids(leak_env: LeakEnv, slug: str) -> list[str]:
    """The ids of the ``label_operations`` rows the database holds for ``slug``."""
    import asyncio

    from src.db.repos.records import list_label_operations
    from src.db.session import async_session_factory, project_tx

    project_id = leak_env.records[slug].id

    async def read() -> list[str]:
        async with async_session_factory()() as session, project_tx(session, project_id):
            rows = await list_label_operations(session, project_id, limit=100)
        return [str(r.id) for r in rows]

    return asyncio.run(read())


def _label(client: TestClient, slug: str) -> str:
    resp = client.put(
        f'{SCOPED.format(project=slug)}/crops/batch_label',
        json={'crop_ids': [f'{slug}-item-0002'], 'class_id': 1},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['updated'] == 1
    return str(resp.json()['op_id'])


def test_an_operation_is_listed_and_stored_only_in_its_own_project(leak_env: LeakEnv) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    op_id = _label(client, 'alpha')

    def listed(slug: str) -> list[str]:
        resp = client.get(f'{SCOPED.format(project=slug)}/label_ops')
        assert resp.status_code == 200, resp.text
        return [o['op_id'] for o in resp.json()['items']]

    assert listed('alpha') == [op_id]
    assert listed('beta') == []
    assert listed('default') == []
    assert _stored_op_ids(leak_env, 'alpha') == [op_id]
    for slug in ('beta', 'default'):
        assert _stored_op_ids(leak_env, slug) == [], 'the row exists only under the owning project'


def test_listing_reaches_only_the_bound_projects_indexes(leak_env: LeakEnv) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    _label(client, 'alpha')
    leak_env.accesses.clear()

    resp = client.get(f'{SCOPED.format(project="alpha")}/label_ops')

    assert resp.status_code == 200, resp.text
    assert resp.json()['items'][0]['current'] == 1
    own = {str(i) for i in leak_env.records['alpha'].resources.indexes.values()}
    touched = {idx for _b, _m, _u, idx in leak_env.accesses if idx} - {'op_global_configs'}
    assert touched, 'the route reached OpenSearch'
    assert touched <= own, f'touched foreign indexes: {touched - own}'
    assert {bound for bound, *_ in leak_env.accesses} == {'alpha'}


def test_another_projects_operation_id_is_404_and_nothing_changes(leak_env: LeakEnv) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    op_id = _label(client, 'alpha')
    alpha_history = f'{SCOPED.format(project="alpha")}/crops/alpha-item-0002/history'
    before = client.get(alpha_history).json()
    assert before['entries'][-1]['op_id'] == op_id
    leak_env.accesses.clear()

    for dry_run in (True, False):
        resp = client.post(
            f'{SCOPED.format(project="beta")}/label_ops/{op_id}/undo', json={'dry_run': dry_run}
        )
        assert resp.status_code == 404, resp.text
        assert resp.json()['detail']['error'] == 'label_op_not_found'

    own = {str(i) for i in leak_env.records['beta'].resources.indexes.values()}
    touched = {idx for _b, _m, _u, idx in leak_env.accesses if idx} - {'op_global_configs'}
    assert touched <= own, f'touched foreign indexes: {touched - own}'
    assert {bound for bound, *_ in leak_env.accesses} <= {'beta'}
    assert client.get(alpha_history).json() == before, "alpha's item is unchanged"
    # Alpha can still undo its own operation, and only alpha's item is restored.
    dry = client.post(
        f'{SCOPED.format(project="alpha")}/label_ops/{op_id}/undo', json={'dry_run': True}
    )
    assert dry.status_code == 200, dry.text
    assert dry.json()['restored'] == 1
