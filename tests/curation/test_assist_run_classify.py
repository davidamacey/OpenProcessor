"""Assist runs end to end (route, scope, VLM call, writes): the class goes through
the shared class write path and never lands on a locked item, answers are kept as
suggestions in the item's ``assist`` slot, and attributes never touch the labels."""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any

import pytest

from curation.config_test_stack import ITEMS
from curation.conftest import ACTIVE
from src.services.labeling.vlm_attributes import AttributeSpec
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


pytestmark = pytest.mark.usefixtures('readable_crops')

RUNS = '/assist/runs'
CLUSTER = 10000
MAKE = {'key': 'make', 'description': 'car manufacturer', 'values': []}


@pytest.fixture(autouse=True)
def _registry(stack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry', lambda: stack.registry
    )


@pytest.fixture(autouse=True)
def _jobs_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_ASSIST_JOBS_DIR', str(tmp_path / 'assist_jobs'))


def _seed(stack: Any, crop_id: str, **doc: Any) -> None:
    defaults = {
        'cluster_id': CLUSTER,
        'cluster_subid': f'{CLUSTER}a',
        'class_source': 'item_model',
        'updated_at': '2026-10-01T00:00:00+00:00',
    }
    stack.seed(crop_id, **{**defaults, **doc})


def _reply(*entries: dict[str, Any]) -> str:
    return json.dumps({'results': [{'img': i, **e} for i, e in enumerate(entries, 1)]})


def _cluster_body(**over: Any) -> dict[str, Any]:
    return {
        'target': {'kind': 'cluster', 'cluster_id': CLUSTER},
        'actions': ['classify'],
        'dry_run': False,
        **over,
    }


def _wait(stack: Any, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = stack.client.get(f'/curation/projects/default{RUNS}/{run_id}').json()
        if state['status'] not in ('queued', 'running'):
            return state
        time.sleep(0.05)
    raise AssertionError('the run did not finish')


def _run(stack: Any, **body: Any) -> dict[str, Any]:
    with stack.client:
        started = stack.post(RUNS, **_cluster_body(**body))
        assert started.status_code == 200, started.text
        assert started.json()['status'] == 'queued'
        return _wait(stack, started.json()['run_id'])


def _doc(stack: Any, crop_id: str) -> dict[str, Any]:
    return dict(stack.items.docs(ITEMS)[crop_id])


@pytest.fixture
def car_pack(stack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The active pack declares one item attribute."""
    pack = dataclasses.replace(
        GENERIC_ITEM_PACK,
        name='cars',
        attributes=[AttributeSpec(**MAKE)],
    )
    monkeypatch.setattr('src.services.curation.assist.prepare.resolve_pack', lambda *_a, **_k: pack)
    monkeypatch.setattr('src.services.curation.assist.runner.get_vlm_labeler', _labeler_for(pack))


def _labeler_for(pack: Any) -> Any:
    from src.services.labeling.run_labeler import get_vlm_labeler as real

    def build(_name: Any, _revision: Any = None, *, endpoint: Any = None) -> Any:
        labeler = real(None, None, endpoint=endpoint)
        labeler._pack = pack
        return labeler

    return build


def test_unlocked_items_get_a_class_suggestion_and_an_assist_slot(stack: Any) -> None:
    _seed(stack, 'free1')
    _seed(stack, 'free2')
    stack.upstream.vlm_content = _reply(
        {'class': 'gadget', 'confidence': 'high'}, {'class': 'widget', 'confidence': 'medium'}
    )

    state = _run(stack)

    assert state['status'] == 'completed', state
    assert state['result']['classify'] == {
        'predicted': 2,
        'updated': 2,
        'failed': 0,
        'unreadable': 0,
        'locked_suggest_only': 0,
    }
    one, two = _doc(stack, 'free1'), _doc(stack, 'free2')
    assert (one['class_name'], one['class_source']) == ('gadget', 'vlm')
    assert (two['class_name'], two['class_source']) == ('widget', 'vlm')
    assert one['assist']['class_name'] == 'gadget'
    assert one['assist']['run_id'] == state['run_id']
    assert one['assist']['class_confidence'] == 'high'
    assert one['vlm_prompt_pack'].startswith(GENERIC_ITEM_PACK.name)


def test_a_human_label_and_a_validated_import_keep_their_class(stack: Any) -> None:
    _seed(stack, 'free1')
    _seed(
        stack, 'human1', class_id=0, class_name='widget', class_source='human', class_validated=True
    )
    _seed(
        stack,
        'import1',
        class_id=0,
        class_name='widget',
        class_source='external_label',
        class_validated=True,
    )
    stack.upstream.vlm_content = _reply(*[{'class': 'gadget', 'confidence': 'high'}] * 3)

    state = _run(stack)

    assert state['result']['classify']['locked_suggest_only'] == 2
    assert state['result']['classify']['updated'] == 1
    for crop_id in ('human1', 'import1'):
        doc = _doc(stack, crop_id)
        assert (doc['class_name'], doc['class_source']) == ('widget', doc['class_source'])
        assert doc['class_source'] != 'vlm'
        assert doc['assist']['class_name'] == 'gadget', 'the disagreement is visible'
    assert _doc(stack, 'free1')['class_name'] == 'gadget'


def test_the_frozen_holdout_is_never_sent_to_the_vlm(stack: Any) -> None:
    _seed(stack, 'free1')
    _seed(stack, 'held', test_holdout=True)
    stack.upstream.vlm_content = _reply({'class': 'gadget', 'confidence': 'high'})

    state = _run(stack)

    assert state['resolved']['skipped']['holdout'] == 1
    (sent,) = stack.upstream.vlm_requests
    images = [p for p in sent['messages'][1]['content'] if p['type'] == 'image_url']
    assert len(images) == 1
    assert 'assist' not in _doc(stack, 'held')
    assert _doc(stack, 'held')['class_source'] == 'item_model'


def test_a_human_edit_during_the_vlm_call_wins(stack: Any) -> None:
    _seed(stack, 'free1')
    stack.upstream.vlm_content = _reply({'class': 'gadget', 'confidence': 'high'})

    async def human_relabels() -> None:
        doc = stack.items.store[ITEMS]['free1']
        doc.update(class_id=0, class_name='widget', class_source='human', class_validated=True)

    stack.upstream.vlm_delay = human_relabels

    state = _run(stack)

    assert state['status'] == 'completed'
    doc = _doc(stack, 'free1')
    assert (doc['class_name'], doc['class_source']) == ('widget', 'human')
    assert doc['assist']['class_name'] == 'gadget'


def test_label_cluster_is_gone(stack: Any) -> None:
    r = stack.client.post(f'/curation/projects/default/vlm/label_cluster/{CLUSTER}')
    assert r.status_code in (404, 405)


def test_a_dry_run_counts_and_writes_nothing(stack: Any) -> None:
    for i in range(3):
        _seed(stack, f'c{i}')
    _seed(stack, 'held', test_holdout=True)
    before = stack.snapshot()

    r = stack.post(RUNS, **_cluster_body(dry_run=True, sample={'mode': 'representative'}))

    assert r.status_code == 200, r.text
    body = r.json()
    assert body['status'] == 'dry_run'
    assert body['run_id'] is None
    assert body['resolved']['scope_count'] == 3
    assert body['resolved']['sampled'] == 3
    assert body['resolved']['skipped']['holdout'] == 1
    assert body['resolved']['estimated_vlm_calls'] == 1
    assert body['resolved']['scope_digest'].startswith('sha256:')
    assert body['resolved']['vlm']['name'] == 'ua'
    assert stack.snapshot() == before
    assert stack.upstream.vlm_requests == []


def test_a_project_with_its_vlm_off_is_refused(stack: Any) -> None:
    _seed(stack, 'free1')
    active = stack.client.get(f'{ACTIVE}/active').json()
    off = stack.client.post(f'{ACTIVE}/deactivate', json={'expected_active': active['active']})
    assert off.status_code == 200, off.text

    r = stack.post(RUNS, **_cluster_body(dry_run=True))

    assert (r.status_code, r.json()['detail']['error']) == (409, 'vlm_not_configured')


def test_a_second_run_while_one_is_live_is_busy(stack: Any) -> None:
    _seed(stack, 'free1')
    stack.upstream.vlm_content = _reply({'class': 'gadget', 'confidence': 'high'})
    gate: dict[str, Any] = {}

    async def hold() -> None:
        import asyncio

        gate['reached'] = True
        while not gate.get('release'):
            await asyncio.sleep(0.01)

    stack.upstream.vlm_delay = hold
    with stack.client:
        first = stack.post(RUNS, **_cluster_body())
        assert first.status_code == 200, first.text
        deadline = time.monotonic() + 10
        while not gate.get('reached') and time.monotonic() < deadline:
            time.sleep(0.01)
        second = stack.post(RUNS, **_cluster_body())
        gate['release'] = True
        _wait(stack, first.json()['run_id'])
    assert (second.status_code, second.json()['detail']['error']) == (409, 'assist_busy')
    assert second.json()['detail']['run_id'] == first.json()['run_id']


def test_an_unknown_cluster_is_a_404_and_an_empty_filter_a_422(stack: Any) -> None:
    missing = stack.post(RUNS, **_cluster_body(target={'kind': 'cluster', 'cluster_id': 99}))
    assert (missing.status_code, missing.json()['detail']['error']) == (404, 'not_found')
    empty = stack.post(RUNS, **_cluster_body(target={'kind': 'filter', 'filter': {}}))
    assert (empty.status_code, empty.json()['detail']['error']) == (422, 'invalid_request')


@pytest.mark.usefixtures('car_pack')
def test_attributes_are_stored_in_the_slot_and_never_touch_the_labels(stack: Any) -> None:
    _seed(stack, 'free1')
    _seed(
        stack, 'human1', class_id=0, class_name='widget', class_source='human', class_validated=True
    )
    stack.upstream.vlm_content = _reply(
        {'attributes': {'make': {'value': 'Acme', 'confidence': 'high'}}},
        {'attributes': {'make': {'value': 'Borealis'}, 'colour': {'value': 'red'}}},
    )

    state = _run(stack, actions=['attributes'])

    assert state['status'] == 'completed', state
    assert state['result']['attributes'] == {'items_written': 2, 'invalid_values': 0}
    assert state['result']['classify'] is None
    free, human = _doc(stack, 'free1'), _doc(stack, 'human1')
    assert free['assist']['attributes'] == [{'key': 'make', 'value': 'acme', 'confidence': 'high'}]
    assert human['assist']['attributes'] == [
        {'key': 'make', 'value': 'borealis', 'confidence': 'low'}
    ]
    assert free['class_source'] == 'item_model'
    assert (human['class_name'], human['class_source']) == ('widget', 'human')


@pytest.mark.usefixtures('car_pack')
def test_classify_with_attributes_makes_one_assist_call(stack: Any) -> None:
    _seed(stack, 'free1')
    stack.upstream.vlm_content = _reply(
        {
            'class': 'gadget',
            'confidence': 'high',
            'attributes': {'make': {'value': 'acme', 'confidence': 'medium'}},
        }
    )

    state = _run(stack, actions=['classify', 'attributes'])

    assert state['status'] == 'completed', state
    assert len(stack.upstream.vlm_requests) == 1
    doc = _doc(stack, 'free1')
    assert doc['class_name'] == 'gadget'
    assert doc['assist']['attributes'][0]['value'] == 'acme'


def test_attributes_with_a_pack_that_declares_none_is_a_422(stack: Any) -> None:
    _seed(stack, 'free1')
    r = stack.post(RUNS, **_cluster_body(actions=['attributes']))
    assert (r.status_code, r.json()['detail']['error']) == (422, 'pack_declares_no_attributes')
