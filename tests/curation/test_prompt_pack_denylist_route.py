"""#125: ``proposal_denylist`` survives the pack REST routes, is validated, is described
in the editor schema, and is applied by the labeler for a stored pack."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.test_proposal_denylist import _entry, _labeler
from src.services.labeling.vlm_models import ItemCrop
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, get_prompt_pack


PREFIX = '/curation/projects/default/prompt_packs'


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    fake_os = FakeConfigOpenSearch()
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def _body() -> dict[str, object]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    return body


def _create(client: TestClient, name: str, deny: object) -> None:
    body = _body()
    body['proposal_denylist'] = deny
    r = client.post(PREFIX, json={'name': name, 'description': '', 'body': body})
    assert r.status_code == 201, r.text


def test_create_get_put_get_round_trips_the_denylist(app_client: TestClient) -> None:
    _create(app_client, 'deny_pack', ['Scratch', 'blur*'])
    assert app_client.get(f'{PREFIX}/deny_pack').json()['body']['proposal_denylist'] == [
        'Scratch',
        'blur*',
    ]
    body = _body()
    body['proposal_denylist'] = ['glare', '*_scene']
    r = app_client.put(
        f'{PREFIX}/deny_pack', json={'expected_revision': 1, 'description': '', 'body': body}
    )
    assert r.status_code == 200, r.text
    got = app_client.get(f'{PREFIX}/deny_pack').json()
    assert got['revision'] == 2
    assert got['body']['proposal_denylist'] == ['glare', '*_scene']


def test_clone_carries_the_denylist(app_client: TestClient) -> None:
    r = app_client.post(
        f'{PREFIX}/{GENERIC_ITEM_PACK.name}/clone', json={'new_name': 'cloned_deny'}
    )
    assert r.status_code == 201, r.text
    got = app_client.get(f'{PREFIX}/cloned_deny').json()['body']['proposal_denylist']
    assert got == GENERIC_ITEM_PACK.proposal_denylist
    assert got


@pytest.mark.parametrize(
    ('bad', 'code'),
    [
        ('notalist', 'pack_field_missing'),
        ([1, 2], 'pack_field_missing'),
        (['ok', '  '], 'pack_field_empty'),
        (['x' * 201], 'pack_field_too_long'),
        (['p'] * 501, 'pack_field_too_long'),
    ],
)
def test_bad_denylist_is_a_validation_issue_not_accepted(
    app_client: TestClient, bad: object, code: str
) -> None:
    body = _body()
    body['proposal_denylist'] = bad
    r = app_client.post(f'{PREFIX}/validate', json={'body': body})
    assert r.status_code == 200, r.text
    report = r.json()
    assert report['ok'] is False
    assert any(e['code'] == code and e['field'] == 'proposal_denylist' for e in report['errors']), (
        report
    )
    # and the save route refuses it too
    r = app_client.post(PREFIX, json={'name': 'bad_deny', 'description': '', 'body': body})
    assert r.status_code >= 400, r.text


def test_schema_describes_the_denylist_as_an_editable_list(app_client: TestClient) -> None:
    fields = {f['field']: f for f in app_client.get(f'{PREFIX}/schema').json()['fields']}
    assert fields['proposal_denylist']['kind'] == 'list'
    assert fields['proposal_denylist']['group'] == 'vocabulary'


@pytest.mark.asyncio
async def test_labeler_applies_a_stored_packs_denylist(app_client: TestClient) -> None:
    _create(app_client, 'zzz_deny', ['zzz_*'])
    # the store the route wrote is the one a labeler resolves packs from
    pack = get_prompt_pack('zzz_deny')
    assert pack is not None
    assert pack.proposal_denylist == ['zzz_*']
    labeler = _labeler([_entry(1, 'zzz_noise'), _entry(2, 'blurry_thing')], pack)
    crops = [ItemCrop(img_id=f'c{i}', jpeg_bytes=b'j') for i in (1, 2)]
    preds = await labeler.label_or_propose_batch(crops, ['box'])
    assert preds[0].proposed_class == ''
    assert preds[1].proposed_class == 'blurry_thing'  # not in THIS pack's denylist
