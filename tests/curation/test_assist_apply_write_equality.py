"""One writer for a carried-over class: an explicit assist apply and the auto-label loop's
propagation call the same ``write_class_to_ids``, so they write the same document.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from typing import Any

import pytest

from curation.config_test_stack import ITEMS
from src.services.curation.assist.apply_write import write_class_to_ids
from src.services.curation.class_label import ItemLabel
from src.services.curation.class_write_guard import class_state_token


RUNS = '/assist/runs'
CLUSTER = 10000
# What legitimately differs between two writes of the same label.
VOLATILE = {'updated_at', 'class_labeled_at'}


@pytest.fixture
def world(stack: Any, readable_crops: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry', lambda: stack.registry
    )
    monkeypatch.setenv('OP_ASSIST_JOBS_DIR', str(tmp_path / 'assist_jobs'))
    return stack


def _seed(stack: Any, crop_id: str, **doc: Any) -> None:
    defaults = {
        'cluster_id': CLUSTER,
        'cluster_subid': f'{CLUSTER}a',
        'class_source': 'item_model',
        'updated_at': '2026-10-01T00:00:00+00:00',
    }
    stack.seed(crop_id, **{**defaults, **doc})


def _run(stack: Any) -> str:
    content = json.dumps(
        {'results': [{'img': i, 'class': 'gadget', 'confidence': 'high'} for i in (1, 2, 3)]}
    )
    stack.upstream.vlm_content = content
    with stack.client:
        started = stack.post(
            RUNS,
            target={'kind': 'cluster', 'cluster_id': CLUSTER},
            actions=['classify'],
            dry_run=False,
        )
        run_id = started.json()['run_id']
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            state = stack.client.get(f'/curation/projects/default{RUNS}/{run_id}').json()
            if state['status'] not in ('queued', 'running'):
                return run_id
            time.sleep(0.05)
    raise AssertionError('the run did not finish')


def _normalized(doc: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in copy.deepcopy(doc).items() if k not in VOLATILE}
    for entry in out.get('class_id_history') or []:
        entry.pop('at', None)
        entry.pop('writer', None)
    return out


def _label(run_id: str, class_id: int) -> ItemLabel:
    return ItemLabel.propagated(
        class_id=class_id, class_name='widget', run_id=run_id, writer=f'assist_loop:{run_id}'
    )


def test_apply_and_the_shared_writer_write_the_same_document(world: Any) -> None:
    for i in range(3):
        _seed(world, f'c{i}')
    run_id = _run(world)
    before = copy.deepcopy(world.items.store[ITEMS])

    applied = world.post(
        f'{RUNS}/{run_id}/apply', class_name='widget', label_source='propagated', dry_run=False
    )
    assert applied.status_code == 200, applied.text
    via_apply = copy.deepcopy(world.items.store[ITEMS])

    world.items.store[ITEMS].clear()
    world.items.store[ITEMS].update(copy.deepcopy(before))
    class_id = next(c.class_id for c in world.registry.load().classes if c.class_name == 'widget')
    result = asyncio.run(
        write_class_to_ids(world.items, ['c0', 'c1', 'c2'], _label(run_id, class_id))
    )

    assert result.updated_ids == ['c0', 'c1', 'c2']
    for crop_id in ('c0', 'c1', 'c2'):
        assert _normalized(world.items.store[ITEMS][crop_id]) == _normalized(via_apply[crop_id])
        assert world.items.store[ITEMS][crop_id]['class_source'] == 'assist_propagated'


def test_a_class_locked_at_write_time_is_skipped_and_reported(world: Any) -> None:
    _seed(world, 'free')
    _seed(
        world, 'human', class_id=0, class_name='widget', class_source='human', class_validated=True
    )
    _seed(
        world,
        'validated',
        class_source='vlm',
        class_validated=True,
        class_id=1,
        class_name='gadget',
    )
    _seed(world, 'frozen', test_holdout=True)
    before = copy.deepcopy(world.items.store[ITEMS])

    result = asyncio.run(
        write_class_to_ids(world.items, ['free', 'human', 'validated', 'frozen'], _label('run1', 0))
    )

    assert result.updated_ids == ['free']
    assert sorted(result.skipped_locked) == ['frozen', 'human', 'validated']
    for crop_id in ('human', 'validated', 'frozen'):
        assert world.items.store[ITEMS][crop_id] == before[crop_id]


def test_an_item_whose_class_state_moved_since_it_was_read_is_skipped(world: Any) -> None:
    _seed(world, 'steady')
    _seed(world, 'moved')
    tokens = {cid: class_state_token(world.items.store[ITEMS][cid]) for cid in ('steady', 'moved')}
    world.items.store[ITEMS]['moved']['class_name'] = 'gadget'  # someone relabeled it

    result = asyncio.run(
        write_class_to_ids(world.items, ['steady', 'moved'], _label('run1', 0), tokens=tokens)
    )

    assert result.updated_ids == ['steady']
    assert result.skipped_changed == ['moved']
    assert world.items.store[ITEMS]['moved']['class_name'] == 'gadget'


def test_a_missing_token_is_not_a_license_to_write(world: Any) -> None:
    _seed(world, 'unknown')

    result = asyncio.run(write_class_to_ids(world.items, ['unknown'], _label('run1', 0), tokens={}))

    assert result.updated_ids == []
    assert result.skipped_changed == ['unknown']
