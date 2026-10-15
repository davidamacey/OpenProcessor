"""Re-running an assist run after a re-refine must not clobber locked labels, and an
older run's write never replaces a newer run's suggestions."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from curation.config_test_stack import ITEMS
from src.services.curation.assist.slot_write import RunMeta, build_slot, write_slots


RUNS = '/assist/runs'
CLUSTER = 10000


@pytest.fixture
def world(stack: Any, readable_crops: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry', lambda: stack.registry
    )
    monkeypatch.setenv('OP_ASSIST_JOBS_DIR', str(tmp_path / 'assist_jobs'))
    return stack


def _run(stack: Any) -> dict[str, Any]:
    body = {
        'target': {'kind': 'cluster', 'cluster_id': CLUSTER},
        'actions': ['classify'],
        'dry_run': False,
    }
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


def test_a_rerun_after_a_refine_keeps_human_labels_and_updates_the_suggestions(world: Any) -> None:
    for i in range(5):
        world.seed(
            f'c{i}',
            cluster_id=CLUSTER,
            cluster_subid=f'{CLUSTER}a',
            class_source='item_model',
            updated_at='2026-10-01T00:00:00+00:00',
        )
    world.upstream.vlm_content = json.dumps(
        {'results': [{'img': i, 'class': 'gadget', 'confidence': 'high'} for i in (1, 2, 3)]}
    )
    first = _run(world)
    assert first['status'] == 'completed', first

    # the user labels three members, then re-refines: every sub-id is rewritten
    store = world.items.store[ITEMS]
    for crop_id in ('c0', 'c1', 'c2'):
        store[crop_id].update(
            class_id=0, class_name='widget', class_source='human', class_validated=True
        )
    for i in range(5):
        store[f'c{i}']['cluster_subid'] = f'{CLUSTER}{"b" if i % 2 else "a"}'

    second = _run(world)

    assert second['status'] == 'completed', second
    for crop_id in ('c0', 'c1', 'c2'):
        doc = _doc(world, crop_id)
        assert (doc['class_name'], doc['class_source']) == ('widget', 'human')
        assert doc['assist']['run_id'] == second['run_id']
        assert doc['assist']['class_name'] == 'gadget', 'the VLM disagreement is the suggestion'
    assert second['result']['classify']['locked_suggest_only'] == 3


@pytest.mark.asyncio
async def test_an_older_runs_write_after_a_newer_one_is_skipped(world: Any) -> None:
    world.seed('c0', cluster_id=CLUSTER, class_source='item_model')
    newer = RunMeta('asr_new', '2026-10-15T12:00:00+00:00', 'env@1', 'm', 'p', None)
    older = RunMeta('asr_old', '2026-10-15T11:00:00+00:00', 'env@1', 'm', 'p', None)

    def slot(meta: RunMeta, name: str) -> dict[str, Any]:
        return build_slot(meta, class_name=name, class_confidence='high', attributes={})

    assert await write_slots(world.items, {'c0': slot(newer, 'gadget')}) == 1
    assert await write_slots(world.items, {'c0': slot(older, 'widget')}) == 0
    assert _doc(world, 'c0')['assist']['run_id'] == 'asr_new'
    assert _doc(world, 'c0')['assist']['class_name'] == 'gadget'

    newest = RunMeta('asr_newest', '2026-10-15T13:00:00+00:00', 'env@1', 'm', 'p', None)
    assert await write_slots(world.items, {'c0': slot(newest, 'widget')}) == 1
    assert _doc(world, 'c0')['assist']['class_name'] == 'widget'


@pytest.mark.asyncio
async def test_a_slot_replaces_everything_an_older_run_wrote_except_box_suggestions(
    world: Any,
) -> None:
    world.seed('c0', cluster_id=CLUSTER, class_source='item_model')
    world.items.store[ITEMS]['c0']['assist'] = {
        'run_id': 'asr_old',
        'run_created_at': '2026-10-15T10:00:00+00:00',
        'class_name': 'widget',
        'class_confidence': 'high',
        'attributes': [{'key': 'make', 'value': 'acme', 'confidence': 'high'}],
        'boxes': [{'box_id': 'b1', 'attributes': []}],
    }
    meta = RunMeta('asr_new', '2026-10-15T12:00:00+00:00', 'env@1', 'm', 'p', None)

    await write_slots(
        world.items,
        {'c0': build_slot(meta, class_name=None, class_confidence=None, attributes={})},
    )

    slot = _doc(world, 'c0')['assist']
    assert slot['run_id'] == 'asr_new'
    assert slot['class_name'] is None
    assert slot['attributes'] == []
    assert slot['boxes'] == [{'box_id': 'b1', 'attributes': []}]
