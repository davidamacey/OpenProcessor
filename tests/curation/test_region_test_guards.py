"""``POST /region_profiles/test`` guards: the segmenter-prompt override is
validated on every profile source, the external-VLM gate runs before
anything is sent, and the per-process slots and time limit hold."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from curation.config_test_stack import profile_body
from src.services.curation import config_test_limits as limits
from src.services.curation.config_test_limits import MAX_TEST_CROP_IDS, active_runs
from src.services.labeling import vlm_client


URL = '/region_profiles/test'
SEGMENT = {
    'bbox_norm': [0.1, 0.5, 0.3, 0.9],
    'score': 0.9,
    'mask_iou': 0.8,
    'mask_polygon': [[0.1, 0.5], [0.3, 0.5], [0.3, 0.9]],
}
COMBINED_REPLY = json.dumps(
    {
        'class_id': 0,
        'class_confidence': 'high',
        'region_visible': True,
        'region_boxes': [{'box': 1, 'region_bbox_correct': True, 'region_confidence': 'high'}],
    }
)


@pytest.fixture
def crop(stack: Any) -> dict[str, Any]:
    stack.upstream.segments = [SEGMENT]
    stack.upstream.vlm_content = COMBINED_REPLY
    return stack.seed('car1', class_id=0, class_name='widget', class_source='item_model')


@pytest.fixture
def active_wheels(stack: Any, crop: dict[str, Any]) -> str:
    assert stack.post('/region_profiles', name='wheels', body=profile_body()).status_code == 201
    activated = stack.client.post(
        '/curation/projects/default/region_profiles/wheels/activate',
        json={'expected_active': None},
    )
    assert activated.status_code == 200, activated.text
    return 'wheels'


@pytest.mark.parametrize('prompt', ['x' * 201, 'wheel\nrim'], ids=['over-200-chars', 'multi-line'])
@pytest.mark.parametrize('source', ['active', 'named', 'draft'])
def test_a_bad_prompt_override_is_a_422_whichever_profile_it_rides_on(
    stack, active_wheels, source: str, prompt: str
) -> None:
    body = {
        'active': {},
        'named': {'profile_name': 'wheels'},
        'draft': {'draft': profile_body()},
    }[source]

    response = stack.post(URL, crop_id='car1', segmenter_text_prompt=prompt, **body)

    assert response.status_code == 422, response.text
    assert response.json()['detail']['error'] == 'profile_invalid'
    assert stack.upstream.segment_requests == []


def test_a_huge_prompt_override_never_reaches_the_segmenter(stack, active_wheels) -> None:
    response = stack.post(URL, crop_id='car1', segmenter_text_prompt='x' * 500_000)

    assert response.status_code == 422
    assert stack.upstream.segment_requests == []


def test_a_valid_prompt_override_on_the_active_profile_is_what_the_segmenter_receives(
    stack, active_wheels
) -> None:
    response = stack.post(URL, crop_id='car1', segmenter_text_prompt='rim')

    assert response.status_code == 200, response.text
    assert [p['text_prompt'] for p in stack.upstream.segment_requests] == ['rim']


def test_an_external_vlm_without_the_acknowledgement_sends_nothing_at_all(stack, crop) -> None:
    stack.upstream.extra_vlm_hosts.add('api.example.com')
    stack.api.dns['api.example.com'] = ['93.184.216.34']
    vlm = {'base_url': 'https://api.example.com/v1', 'model': 'm', 'allow_external': True}

    refused = stack.post(URL, crop_id='car1', draft=profile_body(), verify=True, vlm_draft=vlm)

    assert refused.status_code == 422, refused.text
    assert refused.json()['detail']['error'] == 'vlm_external_not_acknowledged'
    assert stack.upstream.vlm_requests == []
    assert stack.upstream.segment_requests == []

    allowed = stack.post(
        URL,
        crop_id='car1',
        draft=profile_body(),
        verify=True,
        vlm_draft=vlm,
        acknowledge_external=True,
    )

    assert allowed.status_code == 200, allowed.text
    assert [r['_host'] for r in stack.upstream.vlm_requests] == ['api.example.com']


def test_a_busy_segmenter_slot_refuses_before_any_call(stack, crop, monkeypatch) -> None:
    monkeypatch.setitem(limits.SLOT_CAPS, 'segmenter', 0)

    response = stack.post(URL, crop_id='car1', draft=profile_body())

    assert response.status_code == 429
    assert response.json()['detail']['error'] == 'test_busy'
    assert stack.upstream.segment_requests == []


def test_a_busy_vlm_slot_refuses_a_verify_run_and_frees_the_segmenter_slot(
    stack, crop, monkeypatch
) -> None:
    monkeypatch.setitem(limits.SLOT_CAPS, 'vlm', 0)

    response = stack.post(URL, crop_id='car1', draft=profile_body(), verify=True)

    assert response.status_code == 429
    assert response.json()['detail']['error'] == 'test_busy'
    assert stack.upstream.vlm_requests == []
    assert (active_runs('segmenter'), active_runs('vlm')) == (0, 0)


def test_a_verify_run_holds_both_slots_while_it_runs(stack, crop) -> None:
    seen: list[tuple[int, int]] = []

    async def peek() -> None:
        seen.append((active_runs('segmenter'), active_runs('vlm')))

    stack.upstream.vlm_delay = peek

    response = stack.post(URL, crop_id='car1', draft=profile_body(), verify=True)

    assert response.status_code == 200, response.text
    assert seen == [(1, 1)]
    assert (active_runs('segmenter'), active_runs('vlm')) == (0, 0)


def test_a_run_that_takes_too_long_is_a_504_and_frees_both_slots(stack, crop, monkeypatch) -> None:
    async def slow() -> None:
        await asyncio.sleep(5)

    stack.upstream.vlm_delay = slow
    monkeypatch.setattr(limits, 'TEST_TIMEOUT_S', 0.05)

    response = stack.post(URL, crop_id='car1', draft=profile_body(), verify=True)

    assert response.status_code == 504
    assert response.json()['detail']['error'] == 'test_timeout'
    assert (active_runs('segmenter'), active_runs('vlm')) == (0, 0)


def test_a_vlm_outage_on_a_verify_run_is_a_502_not_an_empty_preview(
    stack, crop, monkeypatch
) -> None:
    monkeypatch.setattr(vlm_client, 'RETRY_WAIT_MIN_S', 0.0)

    async def down() -> None:
        raise httpx.ConnectError('refused')

    stack.upstream.vlm_delay = down

    response = stack.post(URL, crop_id='car1', draft=profile_body(), verify=True)

    assert response.status_code == 502
    assert response.json()['detail']['error'] == 'vlm_transport_error'
    assert (active_runs('segmenter'), active_runs('vlm')) == (0, 0)


def test_a_malformed_segmenter_reply_is_a_segmenter_error_not_a_500(stack, crop) -> None:
    for bad in (
        {'bbox_norm': [0.1, 0.1, 0.3, 0.3], 'score': 0.9, 'mask_polygon': [[1], [2]]},
        {'bbox_norm': [0.1, 0.1, 0.3, 0.3], 'score': 'high'},
        {'bbox_norm': ['a', 0.1, 0.3, 0.3], 'score': 0.9},
    ):
        stack.upstream.segments = [bad]

        response = stack.post(URL, crop_id='car1', draft=profile_body())

        assert response.status_code == 502, (bad, response.text)
        assert response.json()['detail']['error'] == 'segmenter_error'


def test_more_crop_ids_than_the_cap_are_refused_before_any_lookup(stack, crop) -> None:
    ids = [f'g{i}' for i in range(MAX_TEST_CROP_IDS + 1)]

    response = stack.post('/prompt_packs/test', call='classify', crop_ids=ids)

    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail['error'] == 'too_many_crop_ids'
    assert detail['limit'] == MAX_TEST_CROP_IDS
    assert 'g0' not in json.dumps(detail), 'the ids are not echoed back'


def test_region_verify_checks_capacity_before_decoding_any_box(stack, crop, monkeypatch) -> None:
    from src.services.curation import pack_test_run as run_mod

    decoded: list[str] = []

    def spy(*_a: Any, **_k: Any) -> bytes:
        decoded.append('x')
        return b''

    monkeypatch.setattr(run_mod, 'load_region_jpeg', spy)
    boxes = [
        {
            'box_id': f'b{i}',
            'bbox_norm': [0.1, 0.1, 0.2, 0.2],
            'state': 'proposed',
            'score': 0.9,
            'detector': 'sam3',
            'source': 'segmenter',
        }
        for i in range(40)
    ]
    stack.seed('many', region_boxes=boxes, region_count=40, region_status='pending_verification')

    response = stack.post('/prompt_packs/test', call='region_verify', crop_ids=['many'])

    assert response.status_code == 422, response.text
    assert response.json()['detail']['error'] == 'too_many_crops'
    assert decoded == []
