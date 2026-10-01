"""W5 (§5.1, W9): which VLM endpoint a pack test sends a real crop to.

A draft endpoint is tested exactly as it would run (through the one
factory), and every way of naming an endpoint goes through the one gate
(mode ``test``): an endpoint outside this deployment is refused unless the
request acknowledges it, and nothing is sent before the refusal."""

from __future__ import annotations

import json
from typing import Any

import pytest

from curation.conftest import ACTIVE


URL = '/prompt_packs/test'
REPLY = json.dumps({'results': [{'img': 1, 'class': 'widget', 'confidence': 'high'}]})


@pytest.fixture
def item(stack: Any) -> Any:
    stack.upstream.vlm_content = REPLY
    stack.api.dns['b.vlm.test'] = ['10.0.0.2']
    return stack.seed('car1', class_source='item_model')


def _draft(**over: Any) -> dict[str, Any]:
    return {'base_url': 'http://b.vlm.test/v1', 'model': 'draft-model', **over}


def test_a_draft_endpoint_is_called_at_its_own_url_with_its_own_model(stack, item) -> None:
    response = stack.post(URL, call='classify', crop_ids=['car1'], vlm_draft=_draft())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['vlm']['draft'] is True
    assert body['vlm']['name'] is None
    assert body['vlm']['endpoint'].startswith('_draft@')
    assert body['vlm']['model'] == 'draft-model'
    (sent,) = stack.upstream.vlm_requests
    assert (sent['_host'], sent['model']) == ('b.vlm.test', 'draft-model')
    assert body['results'][0]['preview_item']['vlm_endpoint'] == body['vlm']['endpoint']


def test_a_saved_endpoint_by_name_and_the_active_default_are_called_as_named(stack, item) -> None:
    stack.api.probe_record[0] = stack.api.probe_record[0].model_copy(update={'root': 'org/b-root'})
    stack.api.ready('ub', base_url='http://b.vlm.test/v1', model='alias-b')

    named = stack.post(URL, call='classify', crop_ids=['car1'], vlm_name='ub')
    default = stack.post(URL, call='classify', crop_ids=['car1'])

    assert named.json()['vlm']['name'] == 'ub'
    assert named.json()['vlm']['draft'] is False
    assert default.json()['vlm']['name'] == 'ua'
    assert [(r['_host'], r['model']) for r in stack.upstream.vlm_requests] == [
        ('b.vlm.test', 'alias-b'),
        ('a.vlm.test', 'alias-a'),
    ]


def test_an_external_draft_needs_the_acknowledgement_and_nothing_is_sent_without_it(
    stack, item
) -> None:
    stack.upstream.extra_vlm_hosts.add('api.example.com')
    draft = _draft(base_url='https://api.example.com/v1', allow_external=True)

    refused = stack.post(URL, call='classify', crop_ids=['car1'], vlm_draft=draft)

    assert refused.status_code == 422, refused.text
    assert refused.json()['detail']['error'] == 'vlm_external_not_acknowledged'
    assert stack.upstream.vlm_requests == [], 'a refused endpoint receives no crop'

    allowed = stack.post(
        URL, call='classify', crop_ids=['car1'], vlm_draft=draft, acknowledge_external=True
    )

    assert allowed.status_code == 200, allowed.text
    assert [r['_host'] for r in stack.upstream.vlm_requests] == ['api.example.com']


def test_naming_a_saved_endpoint_and_a_draft_together_is_a_422(stack, item) -> None:
    response = stack.post(
        URL, call='classify', crop_ids=['car1'], vlm_name='ua', vlm_draft=_draft()
    )

    assert response.status_code == 422
    assert stack.upstream.vlm_requests == []


def test_an_unknown_endpoint_name_is_a_422_listing_the_known_ones(stack, item) -> None:
    response = stack.post(URL, call='classify', crop_ids=['car1'], vlm_name='nope')

    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail['error'] == 'unknown_vlm'
    assert 'ua' in detail['valid_ids']


def test_a_draft_that_may_not_be_contacted_is_refused_before_any_call(stack, item) -> None:
    stack.api.dns['meta.example.com'] = ['169.254.169.254']

    response = stack.post(
        URL,
        call='classify',
        crop_ids=['car1'],
        vlm_draft=_draft(base_url='http://meta.example.com/v1'),
    )

    assert response.status_code == 422
    assert stack.upstream.vlm_requests == []


def test_a_draft_never_becomes_the_projects_endpoint(stack, item) -> None:
    before = stack.client.get(f'{ACTIVE}/active').json()

    stack.post(URL, call='classify', crop_ids=['car1'], vlm_draft=_draft())

    assert stack.client.get(f'{ACTIVE}/active').json() == before
