"""The item wire reports ``embedding_states`` per space (embeddings plan 5.7), not one state.

After a space is added the same item has a state in each space; the route reads them from the
fields the registry names.
"""

from __future__ import annotations

from typing import Any

import pytest
from _embedding_registry_fake import new_space
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.domain.embedding_spaces import Target
from src.routers.curation import _common
from src.services.curation import index_bootstrap


class _OS:
    def __init__(self, docs: dict[str, dict[str, Any]]) -> None:
        self.docs = docs

    async def search(self, *, index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:  # noqa: ARG002
        hits = [{'_id': k, '_source': v} for k, v in self.docs.items()]
        return {'hits': {'total': {'value': len(hits)}, 'hits': hits}}

    async def get(self, *, index: str, id: str, **_: Any) -> dict[str, Any]:  # noqa: A002, ARG002
        return {'_id': id, '_source': self.docs[id], 'found': True}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, embedding_registry: Any) -> Any:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', {'default'})
    embedding_registry.add(new_space(Target.ITEM, 'mobileclip2_s2'), primary=True)
    fake = _OS(
        {
            'c1': {
                'crop_id': 'c1',
                'embedding_state': 'embedded',
                'vec_mobileclip2_s2_v1_state': 'deferred',
            },
            'c2': {'crop_id': 'c2', 'embedding_state': 'failed'},
        }
    )
    app = mount_curation_routers(FastAPI(), curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


P = f'{_common.config.api_prefix}/projects/default'


def test_the_list_reports_the_state_in_each_space(client: TestClient) -> None:
    r = client.get(f'{P}/crops')
    assert r.status_code == 200, r.text
    by_id = {c['crop_id']: c for c in r.json()['crops']}
    assert by_id['c1']['embedding_states'] == {
        'pe_core_l14_336_v1': 'embedded',
        'mobileclip2_s2_v1': 'deferred',
    }
    assert by_id['c2']['embedding_states'] == {'pe_core_l14_336_v1': 'failed'}
    assert 'embedding_state' not in by_id['c1']


def test_the_single_item_reports_the_state_in_each_space(client: TestClient) -> None:
    r = client.get(f'{P}/crops/c1')
    assert r.status_code == 200, r.text
    assert r.json()['embedding_states'] == {
        'pe_core_l14_336_v1': 'embedded',
        'mobileclip2_s2_v1': 'deferred',
    }
