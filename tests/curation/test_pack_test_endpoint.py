"""W5 (§5.1): ``POST /prompt_packs/test`` runs one call of a draft or saved
pack over stored crops through the production labeler and previews each
item's write. It writes nothing."""

from __future__ import annotations

import json
from typing import Any

import pytest

from curation.config_test_stack import profile_body
from curation.conftest import ACTIVE
from src.config import get_region_fields
from src.routers.curation._item_models import ItemDoc
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


F = get_region_fields()
URL = '/prompt_packs/test'
MARKER = 'DRAFT-MARKER-7f3a'


def _boxes(*states: str) -> dict[str, Any]:
    boxes = [
        RegionBox(
            box_id=f'b{i + 1}',
            bbox_norm=(0.2 + 0.2 * i, 0.5, 0.35 + 0.2 * i, 0.8),
            state=state,
            score=0.9,
            detector='sam3',
            detector_version='1',
            source='segmenter',
        )
        for i, state in enumerate(states)
    ]
    return {
        F.status: 'pending_verification',
        'region_detector_chain': ['sam3:hit'],
        **boxes_write_fields(boxes, current_src={}),
    }


@pytest.fixture
def wheels(stack: Any) -> str:
    """A saved region profile (segmenter-only, text-free), and an item with
    two proposed boxes."""
    created = stack.post('/region_profiles', name='wheels', body=profile_body())
    assert created.status_code == 201, created.text
    stack.seed('car1', class_name='', **_boxes('proposed', 'proposed'))
    return 'wheels'


def _combined_reply(*verdicts: bool) -> str:
    return json.dumps(
        {
            'class_id': 0,
            'class_confidence': 'high',
            'region_visible': True,
            'region_boxes': [
                {'box': i + 1, 'region_bbox_correct': v, 'region_confidence': 'high'}
                for i, v in enumerate(verdicts)
            ],
        }
    )


def _only(response: Any) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()


def test_combined_previews_the_item_after_the_verdicts_the_worker_would_write(
    stack, wheels
) -> None:
    stack.upstream.vlm_content = _combined_reply(True, False)

    body = _only(stack.post(URL, call='combined', crop_ids=['car1'], profile_name=wheels))

    (result,) = body['results']
    preview = result['preview_item']
    ItemDoc.model_validate(preview)
    assert [(b['box_id'], b['state']) for b in preview['region_boxes']] == [
        ('b1', 'accepted'),
        ('b2', 'rejected'),
    ]
    assert preview['region_status'] == 'detected'
    assert (preview['region_count'], preview['region_rejected_count']) == (1, 1)
    assert preview['region_profile'] == 'wheels'
    assert preview['vlm_endpoint'] == body['vlm']['endpoint']
    assert preview['vlm_model'] == 'org/a-root'
    assert preview['vlm_prompt_pack'].startswith(GENERIC_ITEM_PACK.name)
    assert preview['class_name'] == 'widget', 'the class the reply picked, by registry name'
    assert result['parsed']['class_id'] == 0
    assert body['parse_ok'] is True
    assert body['raw_reply'] == stack.upstream.vlm_content
    assert body['call'] == 'combined'
    assert 'model' not in body, 'the answering model is only vlm.model'
    assert body['vlm']['model'] == 'org/a-root'


def test_the_prompt_reported_is_the_prompt_the_endpoint_received(stack, wheels) -> None:
    stack.upstream.vlm_content = _combined_reply(True, True)

    body = _only(stack.post(URL, call='combined', crop_ids=['car1'], profile_name=wheels))

    (sent,) = stack.upstream.vlm_requests
    system = next(m for m in sent['messages'] if m['role'] == 'system')['content']
    user = next(m for m in sent['messages'] if m['role'] == 'user')['content']
    assert body['prompt']['system'] == system
    assert body['prompt']['user_text'] == next(p['text'] for p in user if p['type'] == 'text')
    assert sent['model'] == 'alias-a', 'the endpoint is called by its served model name'
    # The two stored boxes were drawn as a numbered overlay on the one image.
    assert sum(1 for p in user if p['type'] == 'image_url') == 1


def test_use_region_box_none_sends_no_boxes_and_changes_only_the_class(stack, wheels) -> None:
    stack.upstream.vlm_content = json.dumps({'class_id': 1, 'region_visible': True})

    body = _only(
        stack.post(
            URL, call='combined', crop_ids=['car1'], profile_name=wheels, use_region_box='none'
        )
    )

    preview = body['results'][0]['preview_item']
    assert preview['class_name'] == 'gadget'
    assert [b['state'] for b in preview['region_boxes']] == ['proposed', 'proposed']
    assert preview['region_status'] == 'pending_verification'


def test_a_reply_that_does_not_parse_is_a_200_with_parse_ok_false(stack, wheels) -> None:
    stack.upstream.vlm_content = 'I cannot help with that'

    body = _only(stack.post(URL, call='combined', crop_ids=['car1'], profile_name=wheels))

    assert body['parse_ok'] is False
    assert body['parse_error']
    assert body['raw_reply'] == 'I cannot help with that'
    assert body['results'][0]['parsed'] is None
    assert body['results'][0]['preview_item'] is None
    assert body['results'][0]['skipped'] == 'no_usable_answer'


def test_classify_previews_the_class_write_and_skips_a_locked_item(stack, wheels) -> None:
    stack.seed('free1', class_source='item_model')
    stack.seed(
        'owned1', class_id=0, class_name='widget', class_source='human', class_validated=True
    )
    stack.upstream.vlm_content = json.dumps(
        {
            'results': [
                {'img': 1, 'class': 'gadget', 'confidence': 'high'},
                {'img': 2, 'class': 'gadget', 'confidence': 'high'},
            ]
        }
    )

    body = _only(stack.post(URL, call='classify', crop_ids=['free1', 'owned1']))

    free, owned = body['results']
    assert free['preview_item']['class_name'] == 'gadget'
    assert free['preview_item']['class_source'] == 'vlm'
    assert free['preview_item']['vlm_endpoint'] == body['vlm']['endpoint']
    assert owned['preview_item'] is None
    assert owned['skipped'] == 'class_locked'
    assert owned['parsed']['class_name'] == 'gadget', 'the answer is still shown'


def test_open_classify_can_propose_a_class_the_registry_lacks(stack, wheels) -> None:
    stack.seed('free1', class_source='item_model')
    stack.upstream.vlm_content = json.dumps(
        {
            'results': [
                {'img': 1, 'class': '__new__', 'confidence': 'low', 'proposed_class': 'gizmo'}
            ]
        }
    )

    body = _only(stack.post(URL, call='open_classify', crop_ids=['free1']))

    preview = body['results'][0]['preview_item']
    assert preview['class_source'] == 'vlm_new_class_pending'
    assert preview['needs_new_class'] is True


def test_region_verify_answers_per_stored_box(stack, wheels) -> None:
    stack.upstream.vlm_content = json.dumps(
        {
            'results': [
                {'img': 1, 'is_region': True, 'confidence': 'high', 'reason': 'a wheel'},
                {'img': 2, 'is_region': False, 'confidence': 'high', 'reason': 'a sticker'},
            ]
        }
    )

    body = _only(stack.post(URL, call='region_verify', crop_ids=['car1']))

    assert [(r['crop_id'], r['box_id']) for r in body['results']] == [
        ('car1', 'b1'),
        ('car1', 'b2'),
    ]
    assert [r['parsed']['is_region'] for r in body['results']] == [True, False]
    preview = body['results'][0]['preview_item']
    assert [(b['box_id'], b['state']) for b in preview['region_boxes']] == [
        ('b1', 'accepted'),
        ('b2', 'rejected'),
    ]
    assert preview['region_verified'] is True


def test_region_verify_needs_a_box_open_to_a_machine_verdict(stack, wheels) -> None:
    stack.seed('bare1')

    response = stack.post(URL, call='region_verify', crop_ids=['bare1'])

    assert response.status_code == 422
    assert response.json()['detail']['error'] == 'no_region_box'
    assert stack.upstream.vlm_requests == []


def test_region_visible_reports_the_answer_and_no_preview(stack, wheels) -> None:
    stack.upstream.vlm_content = json.dumps({'results': [{'img': 1, 'region_visible': False}]})

    body = _only(stack.post(URL, call='region_visible', crop_ids=['car1']))

    assert body['results'][0]['parsed'] is False
    assert body['results'][0]['preview_item'] is None


def test_a_test_run_writes_nothing_anywhere(stack, wheels) -> None:
    stack.upstream.vlm_content = _combined_reply(True, False)
    before = stack.snapshot()

    for call in ('combined', 'classify', 'open_classify', 'region_verify', 'region_visible'):
        stack.post(URL, call=call, crop_ids=['car1'], profile_name=wheels)
    stack.post(URL, call='combined', crop_ids=['nope'], profile_name=wheels)

    assert stack.upstream.vlm_requests, 'the runs really reached the endpoint'
    assert stack.snapshot() == before
    assert stack.items.write_calls == 0


def test_no_vlm_is_a_409_before_anything_is_sent(stack, wheels) -> None:
    assert (
        stack.client.post(
            f'{ACTIVE}/deactivate', json={'expected_active': {'name': 'ua', 'revision': 1}}
        ).status_code
        == 200
    )

    response = stack.post(URL, call='classify', crop_ids=['car1'])

    assert response.status_code == 409
    assert response.json()['detail']['error'] == 'vlm_not_configured'
    assert stack.upstream.vlm_requests == []


def test_a_missing_crop_is_a_404_naming_it(stack, wheels) -> None:
    response = stack.post(URL, call='classify', crop_ids=['car1', 'ghost'])

    assert response.status_code == 404
    detail = response.json()['detail']
    assert detail['error'] == 'crop_not_found'
    assert detail['crop_ids'] == ['ghost']


def test_more_crops_than_one_request_carries_is_refused_before_the_call(stack, wheels) -> None:
    for i in range(9):
        stack.seed(f'many{i}')

    response = stack.post(URL, call='classify', crop_ids=[f'many{i}' for i in range(9)])

    assert response.status_code == 422
    assert response.json()['detail']['error'] == 'too_many_crops'
    assert response.json()['detail']['limit'] == 8
    assert stack.upstream.vlm_requests == []


def test_an_unknown_pack_and_an_unknown_revision_are_told_apart(stack, wheels) -> None:
    nope = stack.post(URL, call='classify', crop_ids=['car1'], pack_name='nope')
    revision = stack.post(
        URL, call='classify', crop_ids=['car1'], pack_name=GENERIC_ITEM_PACK.name, pack_revision=9
    )
    both = stack.post(
        URL,
        call='classify',
        crop_ids=['car1'],
        pack_name=GENERIC_ITEM_PACK.name,
        draft={'class_system': 'x'},
    )

    assert nope.json()['detail']['error'] == 'unknown_pack'
    assert revision.json()['detail']['error'] == 'unknown_revision'
    assert both.status_code == 422
    assert stack.upstream.vlm_requests == []


def test_an_invalid_draft_pack_is_refused_with_the_report(stack, wheels) -> None:
    response = stack.post(URL, call='classify', crop_ids=['car1'], draft={})

    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail['error'] == 'pack_invalid'
    assert detail['report']['errors']
    assert stack.upstream.vlm_requests == []


def test_a_draft_pack_is_what_the_endpoint_receives(stack, wheels) -> None:
    draft = GENERIC_ITEM_PACK.to_dict()
    draft.pop('name')
    draft['class_system'] = f'{MARKER}: ' + draft['class_system']
    stack.upstream.vlm_content = json.dumps(
        {'results': [{'img': 1, 'class': 'widget', 'confidence': 'high'}]}
    )

    body = _only(stack.post(URL, call='classify', crop_ids=['car1'], draft=draft))

    assert body['pack'] == {'name': None, 'revision': None, 'draft': True}
    assert body['prompt']['system'].startswith(MARKER)
    (sent,) = stack.upstream.vlm_requests
    assert sent['messages'][0]['content'].startswith(MARKER)
    assert body['results'][0]['preview_item']['vlm_prompt_pack'].startswith('draft@')


def test_a_busy_process_refuses_instead_of_queueing(stack, wheels, monkeypatch) -> None:
    from src.services.curation import config_test_limits as limits

    monkeypatch.setitem(limits.SLOT_CAPS, 'vlm', 0)

    response = stack.post(URL, call='classify', crop_ids=['car1'])

    assert response.status_code == 429
    assert response.json()['detail']['error'] == 'test_busy'
    assert stack.upstream.vlm_requests == []


def test_a_run_that_takes_too_long_is_stopped(stack, wheels, monkeypatch) -> None:
    import asyncio

    from src.services.curation import config_test_limits as limits

    async def slow() -> None:
        await asyncio.sleep(5)

    stack.upstream.vlm_delay = slow
    monkeypatch.setattr(limits, 'TEST_TIMEOUT_S', 0.05)

    response = stack.post(URL, call='classify', crop_ids=['car1'])

    assert response.status_code == 504
    assert response.json()['detail']['error'] == 'test_timeout'
    from src.services.curation.config_test_limits import active_runs

    assert active_runs('vlm') == 0, 'the slot is released when the run is stopped'


def test_an_upstream_outage_is_a_502_not_an_empty_answer(stack, wheels, monkeypatch) -> None:
    import httpx

    from src.services.labeling import vlm_client

    monkeypatch.setattr(vlm_client, 'RETRY_WAIT_MIN_S', 0.0)

    async def down() -> None:
        raise httpx.ConnectError('refused')

    stack.upstream.vlm_delay = down

    response = stack.post(URL, call='classify', crop_ids=['car1'])

    assert response.status_code == 502
    assert response.json()['detail']['error'] == 'vlm_transport_error'
