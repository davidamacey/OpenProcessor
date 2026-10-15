"""Assist on boxes: verification through assist runs (the replacement of
``/vlm/verify_regions``) and box attributes under ``assist.boxes[]``."""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any

import pytest

from curation.config_test_stack import ITEMS
from src.config import get_region_fields
from src.domain.region_boxes import RegionBox, boxes_write_fields
from src.services.curation.wire import serialize_item
from src.services.labeling.vlm_attributes import AttributeSpec
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


F = get_region_fields()
RUNS = '/assist/runs'
VERIFY_YES = json.dumps({'is_region': True, 'confidence': 'high', 'reason': 'a wheel'})
VERIFY_NO = json.dumps({'is_region': False, 'confidence': 'high', 'reason': 'not a wheel'})


@pytest.fixture
def world(stack: Any, readable_crops: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    monkeypatch.setenv('OP_ASSIST_JOBS_DIR', str(tmp_path / 'assist_jobs'))
    return stack


@pytest.fixture
def box_pack(world: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.labeling.run_labeler import get_vlm_labeler as real

    pack = dataclasses.replace(
        GENERIC_ITEM_PACK,
        name='cars',
        attributes=[AttributeSpec(key='rim', values=['alloy', 'steel'], applies_to='box')],
    )

    def build(_name: Any, _revision: Any = None, *, endpoint: Any = None) -> Any:
        labeler = real(None, None, endpoint=endpoint)
        labeler._pack = pack
        return labeler

    monkeypatch.setattr('src.services.curation.assist.prepare.resolve_pack', lambda *_a, **_k: pack)
    monkeypatch.setattr('src.services.curation.assist.runner.get_vlm_labeler', build)


def _box(box_id: str, x: float, **over: Any) -> RegionBox:
    fields: dict[str, Any] = {
        'box_id': box_id,
        'bbox_norm': (x, 0.1, x + 0.1, 0.3),
        'state': 'proposed',
        'score': 0.8,
        'detector': 'det_model',
        'source': 'detector',
    }
    fields.update(over)
    return RegionBox(**fields)


def _seed(stack: Any, crop_id: str, *boxes: RegionBox, **doc: Any) -> None:
    stack.seed(
        crop_id,
        class_source='item_model',
        **{F.status: 'pending_verification'},
        **boxes_write_fields(list(boxes), current_src={}),
        **doc,
    )


def _body(target: dict[str, Any], actions: list[str], **over: Any) -> dict[str, Any]:
    return {'target': target, 'actions': actions, 'dry_run': False, **over}


def _boxes_target(*pairs: tuple[str, str]) -> dict[str, Any]:
    return {'kind': 'boxes', 'boxes': [{'crop_id': c, 'box_id': b} for c, b in pairs]}


def _run(stack: Any, **body: Any) -> dict[str, Any]:
    with stack.client:
        started = stack.post(RUNS, **body)
        assert started.status_code == 200, started.text
        run_id = started.json()['run_id']
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            state = stack.client.get(f'/curation/projects/default{RUNS}/{run_id}').json()
            if state['status'] not in ('queued', 'running'):
                return state
            time.sleep(0.05)
    raise AssertionError('the run did not finish')


def _doc(stack: Any, crop_id: str) -> dict[str, Any]:
    return dict(stack.items.docs(ITEMS)[crop_id])


def _states(stack: Any, crop_id: str) -> dict[str, str]:
    return {b['box_id']: b['state'] for b in _doc(stack, crop_id)[F.boxes]}


def _replies(stack: Any, *contents: str) -> None:
    queue = list(contents)

    async def next_reply() -> None:
        stack.upstream.vlm_content = queue.pop(0)

    stack.upstream.vlm_delay = next_reply


def test_verify_on_named_boxes_touches_only_those_boxes(world: Any) -> None:
    _seed(world, 'c1', _box('b1', 0.1), _box('b2', 0.4), _box('b3', 0.7))
    _replies(world, VERIFY_YES)

    state = _run(world, **_boxes_target_body(('c1', 'b2')))

    assert state['status'] == 'completed', state
    assert state['result']['verify'] == {
        'boxes': 1,
        'accepted': 1,
        'rejected': 0,
        'locked_skipped': 0,
    }
    assert _states(world, 'c1') == {'b1': 'proposed', 'b2': 'accepted', 'b3': 'proposed'}
    assert len(world.upstream.vlm_requests) == 1
    assert _doc(world, 'c1')['vlm_prompt_pack'].startswith(GENERIC_ITEM_PACK.name)


def _boxes_target_body(*pairs: tuple[str, str], actions: list[str] | None = None) -> dict[str, Any]:
    return _body(_boxes_target(*pairs), actions or ['verify'])


def test_verify_on_a_crop_judges_every_verifiable_box_and_never_a_human_one(world: Any) -> None:
    human = _box('bh', 0.7, source='human', state='accepted')
    _seed(world, 'c1', _box('b1', 0.1), _box('b2', 0.4), human)
    _replies(world, VERIFY_YES, VERIFY_NO)

    state = _run(world, **_body({'kind': 'crops', 'crop_ids': ['c1']}, ['verify']))

    assert state['result']['verify']['boxes'] == 2
    assert (state['result']['verify']['accepted'], state['result']['verify']['rejected']) == (1, 1)
    assert state['result']['verify']['locked_skipped'] == 1
    assert _states(world, 'c1') == {'b1': 'accepted', 'b2': 'rejected', 'bh': 'accepted'}
    assert len(world.upstream.vlm_requests) == 2


def test_a_locked_box_named_explicitly_is_skipped_and_counted(world: Any) -> None:
    human = _box('bh', 0.7, source='human', state='accepted')
    _seed(world, 'c1', _box('b1', 0.1), human)
    _replies(world, VERIFY_NO)

    state = _run(world, **_boxes_target_body(('c1', 'bh'), ('c1', 'b1')))

    assert state['result']['verify']['locked_skipped'] == 1
    assert _states(world, 'c1') == {'b1': 'rejected', 'bh': 'accepted'}


def test_a_human_verdict_during_the_vlm_call_wins(world: Any) -> None:
    _seed(world, 'c1', _box('b1', 0.1))
    world.upstream.vlm_content = VERIFY_NO

    async def human_verifies() -> None:
        world.items.store[ITEMS]['c1'][F.verifier] = 'human'

    world.upstream.vlm_delay = human_verifies

    state = _run(world, **_boxes_target_body(('c1', 'b1')))

    assert state['status'] == 'completed'
    assert _states(world, 'c1') == {'b1': 'proposed'}


def test_an_unknown_box_or_crop_is_counted_not_sent(world: Any) -> None:
    _seed(world, 'c1', _box('b1', 0.1))

    r = world.post(RUNS, **_boxes_target_body(('c1', 'nope'), ('ghost', 'b1')) | {'dry_run': True})

    body = r.json()
    assert r.status_code == 200, r.text
    assert body['resolved']['skipped']['box_not_found'] == 1
    assert body['resolved']['skipped']['not_found'] == 1
    assert world.upstream.vlm_requests == []


@pytest.mark.parametrize(
    'body',
    [
        _boxes_target_body(('c1', 'b1'), actions=['classify']),
        _body(_boxes_target(('c1', 'b1')), ['verify'], sample={'mode': 'representative'}),
        _body({'kind': 'cluster', 'cluster_id': 1}, ['verify']),
        _body({'kind': 'filter', 'filter': {'cluster_id': 1}}, ['verify']),
    ],
)
def test_requests_that_make_no_sense_are_refused(world: Any, body: dict[str, Any]) -> None:
    r = world.post(RUNS, **body)
    assert (r.status_code, r.json()['detail']['error']) == (422, 'invalid_request')


@pytest.mark.usefixtures('box_pack')
def test_box_attributes_land_under_assist_boxes_and_never_touch_the_boxes(world: Any) -> None:
    _seed(world, 'c1', _box('b1', 0.1), _box('b2', 0.4))
    before = [dict(b) for b in _doc(world, 'c1')[F.boxes]]
    world.upstream.vlm_content = json.dumps(
        {
            'results': [
                {'img': 1, 'attributes': {'rim': {'value': 'Alloy', 'confidence': 'high'}}},
                {'img': 2, 'attributes': {'rim': {'value': 'wood'}}},
            ]
        }
    )

    state = _run(world, **_boxes_target_body(('c1', 'b1'), ('c1', 'b2'), actions=['attributes']))

    assert state['status'] == 'completed', state
    assert state['result']['attributes'] == {'items_written': 1, 'invalid_values': 1}
    doc = _doc(world, 'c1')
    assert doc[F.boxes] == before
    boxes = {b['box_id']: b['attributes'] for b in doc['assist']['boxes']}
    assert boxes['b1'] == [{'key': 'rim', 'value': 'alloy', 'confidence': 'high'}]
    assert boxes['b2'] == []
    assert doc['assist']['class_name'] is None


@pytest.mark.usefixtures('box_pack')
def test_a_box_run_keeps_the_item_suggestions_and_a_deleted_box_is_dropped_on_read(
    world: Any,
) -> None:
    _seed(world, 'c1', _box('b1', 0.1))
    world.items.store[ITEMS]['c1']['assist'] = {
        'run_id': 'asr_item',
        'run_created_at': '2026-10-15T10:00:00+00:00',
        'class_name': 'widget',
        'attributes': [{'key': 'make', 'value': 'acme', 'confidence': 'high'}],
        'boxes': [{'box_id': 'gone', 'attributes': []}],
    }
    world.upstream.vlm_content = json.dumps(
        {'results': [{'img': 1, 'attributes': {'rim': {'value': 'steel'}}}]}
    )

    _run(world, **_boxes_target_body(('c1', 'b1'), actions=['attributes']))

    slot = _doc(world, 'c1')['assist']
    assert (slot['run_id'], slot['class_name']) == ('asr_item', 'widget')
    assert slot['attributes'][0]['value'] == 'acme'
    wire = serialize_item(_doc(world, 'c1'))['assist']['boxes']
    assert [b['box_id'] for b in wire] == ['b1'], 'the box that no longer exists is not served'


def test_verify_regions_route_is_gone(world: Any) -> None:
    r = world.client.post(
        '/curation/projects/default/vlm/verify_regions', json={'crop_ids': ['c1']}
    )
    assert r.status_code in (404, 405)
