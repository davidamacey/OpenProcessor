"""Pack-declared attributes: the pack round-trips them, validation guards them,
the editor schema describes them and ``/prompt_packs/test`` previews the assist
and attributes calls without writing."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.services.config_store.pack_validation import validate_pack
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack


PREFIX = '/curation/projects/default/prompt_packs'
CAR_ATTRIBUTES: list[dict[str, Any]] = [
    {'key': 'make', 'description': 'car manufacturer', 'values': ['Acme', 'borealis']},
    {'key': 'rim_material', 'description': 'wheel rim', 'applies_to': 'box'},
]


def _body(**over: Any) -> dict[str, Any]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    body.update(over)
    return body


def _codes(body: dict[str, Any]) -> list[str]:
    return [e.code for e in validate_pack(None, body).errors]


def test_the_shipped_packs_declare_no_attributes_and_have_the_prompts() -> None:
    assert GENERIC_ITEM_PACK.attributes == []
    assert GENERIC_ITEM_PACK.assist_user_template
    assert GENERIC_ITEM_PACK.attributes_user_template
    assert validate_pack(None, _body()).ok


def test_attributes_round_trip_and_normalise_their_vocabulary() -> None:
    pack = PromptPack.from_dict({**_body(attributes=CAR_ATTRIBUTES), 'name': 'cars'})
    assert [a.key for a in pack.attributes] == ['make', 'rim_material']
    assert pack.attributes[0].values == ['acme', 'borealis']
    assert pack.attributes[1].applies_to == 'box'
    again = PromptPack.from_dict(pack.to_dict())
    assert again == pack


def test_the_vehicle_example_declares_item_and_box_attributes() -> None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / 'examples' / 'prompt_packs' / 'vehicle_wheel.json'
    pack = PromptPack.from_json(path)
    by_key = {a.key: a.applies_to for a in pack.attributes}
    assert by_key == {
        'make': 'item',
        'model': 'item',
        'body_style': 'item',
        'rim_material': 'box',
    }


def test_a_pack_stored_before_attributes_existed_still_validates() -> None:
    body = _body()
    for key in (
        'attributes',
        'assist_system',
        'assist_user_template',
        'attributes_system',
        'attributes_user_template',
    ):
        body.pop(key)
    assert validate_pack(None, body).ok


def test_valid_attributes_pass() -> None:
    assert _codes(_body(attributes=CAR_ATTRIBUTES)) == []


@pytest.mark.parametrize(
    ('attributes', 'code'),
    [
        ([{'key': 'a'}, {'key': 'a'}], 'pack_attribute_duplicate'),
        ([{'key': 'Make'}], 'pack_attribute_invalid'),
        ([{'key': '1make'}], 'pack_attribute_invalid'),
        ([{'key': 'x' * 33}], 'pack_attribute_invalid'),
        ([{'key': 'a', 'applies_to': 'image'}], 'pack_attribute_invalid'),
        ([{'key': 'a', 'description': 'd' * 301}], 'pack_attribute_invalid'),
        ([{'key': 'a', 'values': ['v'] * 51}], 'pack_attribute_invalid'),
        ([{'key': 'a', 'values': ['v' * 49]}], 'pack_attribute_invalid'),
        (['make'], 'pack_attribute_invalid'),
        ({'key': 'a'}, 'pack_attribute_invalid'),
        ([{'key': f'k{i}'} for i in range(17)], 'pack_field_too_long'),
    ],
)
def test_malformed_attributes_are_reported(attributes: Any, code: str) -> None:
    assert code in _codes(_body(attributes=attributes))


def test_sixteen_attributes_are_allowed() -> None:
    assert _codes(_body(attributes=[{'key': f'k{i}'} for i in range(16)])) == []


def test_item_attributes_need_the_assist_and_attributes_prompts() -> None:
    body = _body(attributes=CAR_ATTRIBUTES, assist_user_template='', attributes_system='')
    report = validate_pack(None, body)
    fields = {e.field for e in report.errors if e.code == 'pack_field_empty'}
    assert fields == {'assist_user_template', 'attributes_system'}


def test_box_only_attributes_need_just_the_attributes_prompts() -> None:
    body = _body(
        attributes=[{'key': 'rim', 'applies_to': 'box'}], assist_system='', assist_user_template=''
    )
    assert validate_pack(None, body).ok


def test_the_attribute_block_placeholder_is_required() -> None:
    body = _body(attributes=CAR_ATTRIBUTES, assist_user_template='Classes: {class_list}')
    report = validate_pack(None, body)
    missing = [
        e.detail['placeholder'] for e in report.errors if e.code == 'pack_placeholder_missing'
    ]
    assert missing == ['attribute_block']


def test_n_images_is_optional_and_unknown_placeholders_are_refused() -> None:
    keys = 'img class confidence attributes'
    ok = _body(assist_user_template=f'{{class_list}}\n{{attribute_block}}\n{{n_images}} {keys}')
    assert validate_pack(None, ok).ok
    bad = _body(assist_user_template=f'{{class_list}}\n{{attribute_block}}\n{{colour}} {keys}')
    assert 'pack_placeholder_unknown' in _codes(bad)


def test_the_assist_reply_keys_must_be_named() -> None:
    body = _body(
        assist_system='Answer.',
        assist_user_template='{class_list}\n{attribute_block}\nAnswer as JSON.',
    )
    keys = {e.detail['key'] for e in validate_pack(None, body).errors}
    assert {'img', 'class', 'confidence'} <= keys


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    fake_os = FakeConfigOpenSearch()
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def test_attributes_survive_create_get_and_clone(client: TestClient) -> None:
    r = client.post(
        PREFIX,
        json={'name': 'cars', 'description': '', 'body': _body(attributes=CAR_ATTRIBUTES)},
    )
    assert r.status_code == 201, r.text
    got = client.get(f'{PREFIX}/cars').json()['body']['attributes']
    assert [a['key'] for a in got] == ['make', 'rim_material']
    clone = client.post(f'{PREFIX}/cars/clone', json={'new_name': 'cars2'})
    assert clone.status_code == 201, clone.text
    assert len(client.get(f'{PREFIX}/cars2').json()['body']['attributes']) == 2


def test_a_duplicate_attribute_key_is_a_422_with_the_issue(client: TestClient) -> None:
    body = _body(attributes=[{'key': 'a'}, {'key': 'a'}])
    r = client.post(PREFIX, json={'name': 'dup', 'description': '', 'body': body})
    assert r.status_code == 422
    codes = [e['code'] for e in r.json()['detail']['report']['errors']]
    assert 'pack_attribute_duplicate' in codes


def test_the_editor_schema_lists_the_new_fields_and_calls(client: TestClient) -> None:
    schema = client.get(f'{PREFIX}/schema').json()
    by_field = {f['field']: f for f in schema['fields']}
    assert by_field['attributes']['kind'] == 'attributes'
    assert by_field['assist_user_template']['required_placeholders'] == [
        'class_list',
        'attribute_block',
    ]
    assert by_field['assist_user_template']['allowed_placeholders'][-1] == 'n_images'
    assert by_field['assist_user_template']['group'] == 'assist'
    assert {c['id'] for c in schema['calls']} >= {'assist', 'attributes'}


# ---- POST /prompt_packs/test -----------------------------------------------------

TEST_URL = '/prompt_packs/test'


def _draft() -> dict[str, Any]:
    return _body(attributes=CAR_ATTRIBUTES)


def test_the_assist_call_returns_parsed_attributes_and_writes_nothing(stack: Any) -> None:
    stack.seed('car1', class_source='item_model')
    before = stack.snapshot()
    stack.upstream.vlm_content = json.dumps(
        {
            'results': [
                {
                    'img': 1,
                    'class': 'gadget',
                    'confidence': 'high',
                    'attributes': {'make': {'value': 'Borealis', 'confidence': 'high'}},
                }
            ]
        }
    )

    r = stack.post(TEST_URL, call='assist', crop_ids=['car1'], draft=_draft())

    assert r.status_code == 200, r.text
    body = r.json()
    assert body['parse_ok'] is True
    (result,) = body['results']
    assert result['parsed']['class_name'] == 'gadget'
    assert result['parsed']['attributes']['make'] == {'value': 'borealis', 'confidence': 'high'}
    assert stack.snapshot() == before


def test_the_attributes_call_returns_attributes_only(stack: Any) -> None:
    stack.seed('car1')
    stack.upstream.vlm_content = json.dumps(
        {'results': [{'img': 1, 'attributes': {'make': {'value': 'borealis'}}}]}
    )

    r = stack.post(TEST_URL, call='attributes', crop_ids=['car1'], draft=_draft())

    assert r.status_code == 200, r.text
    assert r.json()['results'][0]['parsed']['attributes']['make']['value'] == 'borealis'


def test_a_pack_without_item_attributes_is_a_422(stack: Any) -> None:
    stack.seed('car1')

    r = stack.post(TEST_URL, call='attributes', crop_ids=['car1'])

    assert r.status_code == 422
    assert r.json()['detail']['error'] == 'pack_declares_no_attributes'
    assert stack.upstream.vlm_requests == []


def test_the_general_objects_example_loads_validates_and_declares_domain_neutral_attributes() -> (
    None
):
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2] / 'examples' / 'prompt_packs' / 'general_objects.json'
    )
    pack = PromptPack.from_json(path)
    assert pack.name == 'general_objects'
    by_key = {a.key: a for a in pack.attributes}
    assert set(by_key) == {'brand', 'colour', 'material'}
    assert {a.applies_to for a in pack.attributes} == {'item'}
    assert by_key['brand'].values == []  # free text
    assert 'red' in by_key['colour'].values  # a closed vocabulary
    body = json.loads(path.read_text())
    body.pop('_comment', None)
    body.pop('name')
    assert validate_pack(None, body).ok
