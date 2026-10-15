"""The `segment` action: an assist run re-queues items for the region worker with a
verification choice, waits in `waiting_region`, then continues with `verify` and the
rest. Refusals are specific 409 codes."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from curation.config_test_stack import ITEMS, profile_body
from src.config import get_region_fields
from src.domain.region_boxes import RegionBox, boxes_write_fields


F = get_region_fields()
RUNS = '/assist/runs'
SCOPED = '/curation/projects/default'
VERIFY_YES = json.dumps({'is_region': True, 'confidence': 'high', 'reason': 'a wheel'})


@pytest.fixture
def world(stack: Any, readable_crops: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    monkeypatch.setenv('OP_ASSIST_JOBS_DIR', str(tmp_path / 'assist_jobs'))
    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr('src.services.curation.assist.segment.POLL_S', 0.05)
    return stack


def _activate(stack: Any, **profile: Any) -> None:
    created = stack.post('/region_profiles', name='wheels', body=profile_body(**profile))
    assert created.status_code == 201, created.text
    activated = stack.client.post(
        f'{SCOPED}/region_profiles/wheels/activate', json={'expected_active': None}
    )
    assert activated.status_code == 200, activated.text


def _seed(stack: Any, crop_id: str, **doc: Any) -> None:
    stack.seed(
        crop_id,
        cluster_id=10000,
        class_source='item_model',
        updated_at='2026-10-01T00:00:00+00:00',
        **{F.status: 'detection_failed'},
        **doc,
    )


def _body(*actions: str, verify: str = 'none', **over: Any) -> dict[str, Any]:
    return {
        'target': {'kind': 'cluster', 'cluster_id': 10000},
        'actions': list(actions),
        'segment': {'verify': verify},
        'dry_run': False,
        **over,
    }


def _state(stack: Any, run_id: str) -> dict[str, Any]:
    return stack.client.get(f'{SCOPED}{RUNS}/{run_id}').json()


def _wait(stack: Any, run_id: str, *statuses: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = _state(stack, run_id)
        if state['status'] in statuses:
            return state
        time.sleep(0.03)
    raise AssertionError(f'never reached {statuses}: {_state(stack, run_id)}')


def _fake_region_worker(stack: Any, stop: threading.Event) -> threading.Thread:
    """Stands in for the region worker: every item stamped with a run id and pending is
    detected with one accepted box, and the per-item choice is cleared as the worker does."""

    def work() -> None:
        while not stop.is_set():
            for doc in stack.items.store[ITEMS].values():
                if doc.get('region_run_id') and doc.get(F.status) == 'pending_detection':
                    box = RegionBox(
                        box_id=f'b-{doc["crop_id"]}',
                        bbox_norm=(0.2, 0.2, 0.4, 0.4),
                        state='accepted',
                        source='segmenter',
                    )
                    doc.update(boxes_write_fields([box], current_src={}))
                    doc[F.status] = 'detected'
                    doc['region_run_verify'] = None
            time.sleep(0.02)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread


def test_segment_without_an_active_region_profile_is_refused(world: Any) -> None:
    _seed(world, 'c1')
    r = world.post(RUNS, **_body('segment'))
    assert (r.status_code, r.json()['detail']['error']) == (409, 'region_profile_required')


def test_segment_with_the_region_stage_paused_is_refused(world: Any) -> None:
    from src.services.curation.region_stage_control import set_region_stage_paused

    _activate(world)
    _seed(world, 'c1')
    set_region_stage_paused(True)
    try:
        r = world.post(RUNS, **_body('segment'))
    finally:
        set_region_stage_paused(False)
    assert (r.status_code, r.json()['detail']['error']) == (409, 'region_stage_paused')


def test_segment_with_a_profile_that_has_no_segmenter_prompt_is_refused(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.triton_control import TritonControlService

    async def _repo() -> list[dict[str, str]]:
        return [{'name': 'wheel_det', 'state': 'READY', 'version': '1'}]

    monkeypatch.setattr(TritonControlService, 'get_repository_index', lambda _s: _repo())
    _activate(world, segmenter_text_prompt='', detector_model='wheel_det')
    _seed(world, 'c1')
    r = world.post(RUNS, **_body('segment'))
    assert (r.status_code, r.json()['detail']['error']) == (409, 'no_segmenter_leg')


def test_vlm_verification_with_another_endpoint_than_the_projects_is_refused(world: Any) -> None:
    _activate(world)
    _seed(world, 'c1')
    world.api.dns['b.vlm.test'] = ['10.0.0.2']
    world.api.ready('ub', base_url='http://b.vlm.test/v1', model='alias-b')

    r = world.post(RUNS, **_body('segment', verify='vlm', vlm='ub'))

    assert (r.status_code, r.json()['detail']['error']) == (409, 'verify_endpoint_mismatch')


@pytest.mark.parametrize(
    'body',
    [
        {'target': {'kind': 'cluster', 'cluster_id': 1}, 'actions': ['segment']},
        {
            'target': {'kind': 'cluster', 'cluster_id': 1},
            'actions': ['classify'],
            'segment': {'verify': 'none'},
        },
        {
            'target': {'kind': 'boxes', 'boxes': [{'crop_id': 'a', 'box_id': 'b'}]},
            'actions': ['segment'],
            'segment': {'verify': 'none'},
        },
        {
            'target': {'kind': 'cluster', 'cluster_id': 1},
            'actions': ['segment', 'verify'],
            'segment': {'verify': 'vlm'},
        },
    ],
    ids=['no spec', 'spec without segment', 'boxes', 'double verification'],
)
def test_senseless_segment_requests_are_a_422(world: Any, body: dict[str, Any]) -> None:
    r = world.post(RUNS, **body)
    assert (r.status_code, r.json()['detail']['error']) == (422, 'invalid_request')


def test_segment_requeues_with_the_choice_waits_and_completes(world: Any) -> None:
    _activate(world)
    _seed(world, 'c1')
    _seed(world, 'c2')
    _seed(
        world,
        'locked',
        **{F.validated: True, F.verifier: 'human'},
    )
    stop = threading.Event()
    with world.client:
        started = world.post(RUNS, **_body('segment'))
        assert started.status_code == 200, started.text
        run_id = started.json()['run_id']
        waiting = _wait(world, run_id, 'waiting_region')
        stamped = {
            cid: (d.get('region_run_id'), d.get('region_run_verify'))
            for cid, d in world.items.store[ITEMS].items()
            if d.get('region_run_id')
        }
        assert stamped == {'c1': (run_id, 'none'), 'c2': (run_id, 'none')}, (
            'a validated set is not requeued'
        )
        assert waiting['progress']['processed'] == 0
        worker = _fake_region_worker(world, stop)
        done = _wait(world, run_id, 'completed', 'completed_with_errors', 'failed')
        stop.set()
        worker.join(timeout=2)

    assert done['status'] == 'completed', done
    assert done['result']['segment'] == {'queued': 2, 'done': 2, 'pending': 0, 'failed': 0}
    assert world.items.store[ITEMS]['locked'].get('region_run_id') is None


def test_segment_then_verify_judges_the_new_boxes_with_the_runs_endpoint(world: Any) -> None:
    _activate(world)
    _seed(world, 'c1')
    world.upstream.vlm_content = VERIFY_YES
    stop = threading.Event()
    with world.client:
        started = world.post(
            RUNS, **_body('segment', 'verify', target={'kind': 'crops', 'crop_ids': ['c1']})
        )
        assert started.status_code == 200, started.text
        run_id = started.json()['run_id']
        _wait(world, run_id, 'waiting_region')
        worker = _fake_region_worker(world, stop)
        done = _wait(world, run_id, 'completed', 'completed_with_errors', 'failed')
        stop.set()
        worker.join(timeout=2)

    assert done['status'] == 'completed', done
    assert done['result']['verify']['boxes'] == 1
    assert len(world.upstream.vlm_requests) == 1
    assert world.items.store[ITEMS]['c1']['region_run_verify'] is None


def test_cancelling_while_waiting_for_the_region_worker_ends_the_run(world: Any) -> None:
    _activate(world)
    _seed(world, 'c1')
    with world.client:
        run_id = world.post(RUNS, **_body('segment')).json()['run_id']
        _wait(world, run_id, 'waiting_region')
        assert world.post(f'{RUNS}/{run_id}/cancel').status_code == 200
        done = _wait(world, run_id, 'cancelled')
    assert done['status'] == 'cancelled'
