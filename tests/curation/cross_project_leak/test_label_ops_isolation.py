"""Label operations belong to the project that made them: its list, its record files, its indexes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from curation.cross_project_leak._constants import SCOPED


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv


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

    alpha_dir = leak_env.records['alpha'].resources.project_state_dir / 'label_ops'
    assert (alpha_dir / op_id / 'state.json').is_file()
    for slug in ('beta', 'default'):
        other = leak_env.records[slug].resources.project_state_dir / 'label_ops'
        assert not (other / op_id).exists()
    written = set(leak_env.root.rglob(op_id))
    assert written == {alpha_dir / op_id}, 'op files only under the owning project'
    assert json.loads((alpha_dir / op_id / 'state.json').read_text())['op_id'] == op_id


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
