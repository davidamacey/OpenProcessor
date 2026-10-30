"""``/assist/loops`` end to end over the real app: the dry-run plan, the refusals, the job the
start writes for the worker, and, once the worker's code has run the loop against the faked VLM,
the state, nodes, members and rollback it serves."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from curation.config_test_stack import ITEMS
from curation.conftest import ACTIVE
from src.domain.loop_stats import required_sample
from src.services.curation.assist_loop import job, runtime
from src.services.labeling.vlm_client import DEFAULT_OPEN_IMAGES_PER_CALL


pytestmark = pytest.mark.usefixtures('readable_crops')

LOOPS = '/assist/loops'
CLUSTER = 10000


@pytest.fixture(autouse=True)
def _env(stack: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry', lambda: stack.registry
    )
    monkeypatch.setenv('OP_ASSIST_LOOP_JOBS_DIR', str(tmp_path / 'loops'))


def _seed(stack: Any, count: int = 60, *, cluster: int = CLUSTER, prefix: str = 'c', **doc: Any):
    for i in range(count):
        defaults = {
            'cluster_id': cluster,
            'class_source': 'item_model',
            'updated_at': '2026-10-01T00:00:00+00:00',
        }
        stack.seed(f'{prefix}{i:03d}', **{**defaults, **doc})


def _body(**over: Any) -> dict[str, Any]:
    return {'budget': {'max_vlm_crops': 500}, 'scope': {'min_root_size': 10}, **over}


def _start(stack: Any, **over: Any) -> Any:
    return stack.post(LOOPS, **_body(dry_run=False, **over))


def _gadget(stack: Any) -> None:
    results = [
        {'img': i, 'class': 'gadget', 'confidence': 'high'}
        for i in range(1, DEFAULT_OPEN_IMAGES_PER_CALL + 1)
    ]
    stack.upstream.vlm_content = json.dumps({'results': results})


def _error(response: Any) -> tuple[int, str]:
    return response.status_code, response.json()['detail']['error']


def _get(stack: Any, path: str = '', **params: Any) -> Any:
    return stack.client.get(f'/curation/projects/default{LOOPS}{path}', params=params)


def _opensearch(stack: Any) -> Any:
    from src.routers.curation import _raw_opensearch_dep

    return stack.client.app.dependency_overrides[_raw_opensearch_dep]()


def _run_worker(stack: Any, loop_id: str) -> Any:
    """What the loop worker does with a queued loop."""
    assert job.claim_trigger() == loop_id
    return asyncio.run(runtime.run_job(_opensearch(stack), loop_id))


# ---------------------------------------------------------------- the plan


def test_a_dry_run_is_the_default_and_sizes_the_loop_without_writing(stack: Any) -> None:
    _seed(stack, 60)
    before = stack.snapshot()

    r = stack.post(LOOPS, scope={'min_root_size': 10})

    assert r.status_code == 200, r.text
    body = r.json()
    assert (body['status'], body['loop_id']) == ('dry_run', None)
    assert body['plan']['roots'] == 1
    assert body['plan']['eligible_items'] == 60
    crops = required_sample(60, 0.9, 0.05 / 2, 0)
    assert body['plan']['estimated_vlm_crops_root_level'] == crops
    assert body['plan']['estimated_vlm_requests_root_level'] == -(
        -crops // DEFAULT_OPEN_IMAGES_PER_CALL
    )
    assert body['plan']['fits_budget'] is None
    (row,) = body['plan']['required_sample_table']
    assert row['n_members'] == 60
    assert row['required_by_dissent'] == [required_sample(60, 0.9, 0.05 / 2, d) for d in (0, 1)]
    assert body['pinned']['vlm']['name'] == 'ua'
    assert body['pinned']['audit_vlm'] is None
    assert stack.snapshot() == before
    assert stack.upstream.vlm_requests == []
    assert job.live_loop_ids() == []


def test_the_plan_says_whether_its_estimate_fits_the_budget(stack: Any) -> None:
    _seed(stack, 60)

    small = stack.post(LOOPS, **_body(budget={'max_vlm_crops': 3}))
    large = stack.post(LOOPS, **_body())

    assert small.json()['plan']['fits_budget'] is False
    assert large.json()['plan']['fits_budget'] is True


def test_the_plan_counts_only_what_the_lock_rule_leaves_eligible(stack: Any) -> None:
    _seed(stack, 60)
    _seed(stack, 5, prefix='h', class_source='human', class_name='widget', class_validated=True)
    _seed(stack, 5, prefix='f', test_holdout=True)

    plan = stack.post(LOOPS, scope={'min_root_size': 10}).json()['plan']

    assert plan['eligible_items'] == 60


# ---------------------------------------------------------------- refusals


def test_a_start_needs_a_budget(stack: Any) -> None:
    _seed(stack)

    r = stack.post(LOOPS, scope={'min_root_size': 10}, dry_run=False)

    assert _error(r) == (422, 'invalid_request')
    assert 'max_vlm_crops' in r.json()['detail']['message']
    assert job.live_loop_ids() == []


def test_a_misspelt_option_is_a_422(stack: Any) -> None:
    _seed(stack)

    assert stack.post(LOOPS, **_body(gaet={'theta': 0.9})).status_code == 422
    assert stack.post(LOOPS, **_body(gate={'theta': 0.2})).status_code == 422


def test_no_cluster_big_enough_is_no_roots(stack: Any) -> None:
    _seed(stack, 5)

    assert _error(stack.post(LOOPS, **_body())) == (422, 'no_roots')


def test_a_project_without_item_classes_is_refused(
    stack: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.clients.curation_opensearch.registry import ClassRegistry

    _seed(stack)
    empty = ClassRegistry(path=tmp_path / 'empty_registry.json')
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry', lambda: empty
    )

    assert _error(stack.post(LOOPS, **_body())) == (422, 'no_class_names')


def test_a_project_with_its_vlm_off_is_refused(stack: Any) -> None:
    _seed(stack)
    active = stack.client.get(f'{ACTIVE}/active').json()
    off = stack.client.post(f'{ACTIVE}/deactivate', json={'expected_active': active['active']})
    assert off.status_code == 200, off.text

    assert _error(stack.post(LOOPS, **_body())) == (409, 'vlm_not_configured')


def test_a_second_loop_is_busy_and_names_the_first(stack: Any) -> None:
    _seed(stack)
    first = _start(stack).json()

    second = _start(stack)

    assert _error(second) == (409, 'assist_loop_busy')
    assert second.json()['detail']['loop_id'] == first['loop_id']


def test_a_start_is_refused_while_an_auto_label_job_is_live(
    stack: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation.autolabel import job as auto_label_job

    _seed(stack)
    monkeypatch.setattr(auto_label_job, 'is_busy', lambda: True)

    r = _start(stack)

    assert _error(r) == (409, 'clustering_running')
    assert job.live_loop_ids() == []


def test_an_external_audit_endpoint_needs_the_consent_rule(stack: Any) -> None:
    _seed(stack)
    stack.api.dns['api.vendor.com'] = ['93.184.216.34']
    stack.api.ready('ext', base_url='https://api.vendor.com/v1', allow_external=True)

    refused = stack.post(LOOPS, **_body(audit={'vlm': 'ext'}))
    allowed = stack.post(LOOPS, **_body(audit={'vlm': 'ext', 'acknowledge_external': True}))

    assert _error(refused) == (422, 'vlm_external_not_acknowledged')
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()['pinned']['audit_vlm']['name'] == 'ext'
    assert job.live_loop_ids() == []


# ---------------------------------------------------------------- start, list, signals


def test_a_start_queues_the_loop_for_the_worker(stack: Any) -> None:
    _seed(stack, 60)
    before = stack.snapshot()

    r = _start(stack, seed=7, gate={'theta': 0.95})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body['status'] == 'queued'
    assert body['loop_id'].startswith('alp_')
    assert body['request']['seed'] == 7
    assert body['request']['gate']['theta'] == 0.95
    assert body['request']['dry_run'] is False
    assert body['pinned']['vlm'] == {'name': 'ua', 'revision': 1, 'model': 'org/a-root'}
    assert body['plan']['roots'] == 1
    assert stack.snapshot() == before, 'the API writes no label'
    stored = json.loads((job.open_job(body['loop_id']).directory / 'request.json').read_text())
    assert stored['seed'] == 7
    assert stored['budget'] == {
        'max_vlm_crops': 500,
        'max_vlm_requests': None,
        'max_wall_s': None,
    }
    assert job.claim_trigger() == body['loop_id']
    assert _get(stack, f'/{body["loop_id"]}').json()['status'] == 'queued'


def test_the_list_is_newest_first_and_can_be_limited_to_live_loops(stack: Any) -> None:
    _seed(stack)
    first = _start(stack).json()['loop_id']
    job.open_job(first).update(status='completed')
    second = _start(stack).json()['loop_id']

    everything = _get(stack).json()['loops']
    live = _get(stack, active='true').json()['loops']
    one = _get(stack, limit=1).json()['loops']

    assert [x['loop_id'] for x in everything] == [second, first]
    assert [x['loop_id'] for x in live] == [second]
    assert [x['loop_id'] for x in one] == [second]


def test_an_unknown_loop_is_a_404_everywhere(stack: Any) -> None:
    for path in ('', '/nodes', '/nodes/r1/members'):
        assert _get(stack, f'/alp_nope{path}').status_code == 404, path
    for action in ('cancel', 'resume', 'rollback'):
        assert stack.post(f'{LOOPS}/alp_nope/{action}').status_code == 404, action


def test_cancel_stops_a_queued_loop_and_resume_queues_it_again(stack: Any) -> None:
    _seed(stack)
    loop_id = _start(stack).json()['loop_id']

    cancelled = stack.post(f'{LOOPS}/{loop_id}/cancel')
    assert cancelled.json()['status'] == 'cancelled'
    resumed = stack.post(f'{LOOPS}/{loop_id}/resume')

    assert resumed.status_code == 200, resumed.text
    assert resumed.json()['status'] == 'queued'
    assert job.claim_trigger() == loop_id


def test_a_loop_that_is_not_stopped_cannot_be_resumed(stack: Any) -> None:
    _seed(stack)
    loop_id = _start(stack).json()['loop_id']

    r = stack.post(f'{LOOPS}/{loop_id}/resume')

    assert _error(r) == (409, 'assist_loop_not_resumable')


# ---------------------------------------------------------------- a finished loop


@pytest.fixture
def finished(stack: Any) -> dict[str, Any]:
    """A loop over one 60-item cluster the faked VLM labels 'gadget', run by the worker's code."""
    _seed(stack, 60)
    _gadget(stack)
    loop_id = _start(stack).json()['loop_id']
    outcome = _run_worker(stack, loop_id)
    assert outcome.status == 'completed', outcome
    return {'loop_id': loop_id}


def test_the_state_of_a_finished_loop_carries_the_record(stack: Any, finished: dict) -> None:
    state = _get(stack, f'/{finished["loop_id"]}').json()

    assert state['status'] == 'completed'
    assert state['stop_reason'] == 'done'
    assert state['partition']['roots'] == 1
    assert state['partition']['eligible_items'] == 60
    assert state['totals']['items_propagated'] > 0
    assert state['totals']['clusters_left_for_review'] == 0
    (level,) = state['levels']
    assert (level['level'], level['nodes'], level['passed']) == (0, 1, 1)
    assert state['budget']['vlm']['crops_spent'] == level['crops_spent'] > 0
    assert state['budget']['vlm']['max_crops'] == 500
    assert state['audit'] == {
        'status': 'off',
        'sampled': 0,
        'agreed': 0,
        'agreement': None,
        'lower_bound': None,
        'crops_spent': 0,
    }


def test_the_nodes_are_listed_with_their_votes(stack: Any, finished: dict) -> None:
    page = _get(stack, f'/{finished["loop_id"]}/nodes').json()

    (node,) = page['nodes']
    assert page['next_after'] is None
    assert (node['path'], node['level'], node['parent']) == (f'r{CLUSTER}', 0, None)
    assert node['decision'] == 'pass'
    assert node['class_name'] == 'gadget'
    assert node['n'] == 60
    assert node['answered'] == node['sampled'] == node['distribution'][0]['votes']
    assert node['items_propagated'] == 60 - node['answered']
    assert node['bound'] > 0.9 - 0.2


def test_nodes_filter_by_level_and_decision_and_page(stack: Any, finished: dict) -> None:
    loop_id = finished['loop_id']

    assert _get(stack, f'/{loop_id}/nodes', level=1).json()['nodes'] == []
    assert _get(stack, f'/{loop_id}/nodes', decision='split').json()['nodes'] == []
    assert len(_get(stack, f'/{loop_id}/nodes', decision='pass').json()['nodes']) == 1
    assert _get(stack, f'/{loop_id}/nodes', after=f'r{CLUSTER}').json()['nodes'] == []
    assert _get(stack, f'/{loop_id}/nodes', decision='nope').status_code == 422


def test_a_nodes_members_are_paged_in_id_order(stack: Any, finished: dict) -> None:
    loop_id = finished['loop_id']
    path = f'/{loop_id}/nodes/r{CLUSTER}/members'

    first = _get(stack, path, limit=25).json()
    second = _get(stack, path, limit=25, after=first['next_after']).json()
    rest = _get(stack, path, limit=25, after=second['next_after']).json()

    ids = first['crop_ids'] + second['crop_ids'] + rest['crop_ids']
    assert ids == sorted(ids) == [f'c{i:03d}' for i in range(60)]
    assert rest['next_after'] is None
    assert _get(stack, f'/{loop_id}/nodes/r1/members').status_code == 404


def test_rollback_is_a_dry_run_unless_asked_and_restores_what_the_loop_wrote(
    stack: Any, finished: dict
) -> None:
    loop_id = finished['loop_id']
    propagated = sorted(
        c
        for c, d in stack.items.docs(ITEMS).items()
        if d.get('class_source') == 'assist_propagated'
    )
    assert propagated

    dry = stack.post(f'{LOOPS}/{loop_id}/rollback')
    assert dry.status_code == 200, dry.text
    assert dry.json() == {
        'dry_run': True,
        'matched': len(propagated),
        'restored': len(propagated),
        'kept_changed': 0,
        'conflicts': 0,
    }
    assert all(
        stack.items.docs(ITEMS)[c]['class_source'] == 'assist_propagated' for c in propagated
    )

    done = stack.post(f'{LOOPS}/{loop_id}/rollback', dry_run=False)
    again = stack.post(f'{LOOPS}/{loop_id}/rollback', dry_run=False)

    assert done.json()['restored'] == len(propagated)
    assert all(
        stack.items.docs(ITEMS)[c]['class_source'] != 'assist_propagated' for c in propagated
    )
    assert (again.json()['matched'], again.json()['restored']) == (0, 0)


def test_a_rollback_can_name_nodes(stack: Any, finished: dict) -> None:
    r = stack.post(f'{LOOPS}/{finished["loop_id"]}/rollback', node_paths=['r99999'], dry_run=True)

    assert r.status_code == 200
    assert r.json()['matched'] == 0


def test_a_live_loop_cannot_be_rolled_back(stack: Any) -> None:
    _seed(stack)
    loop_id = _start(stack).json()['loop_id']

    r = stack.post(f'{LOOPS}/{loop_id}/rollback', dry_run=False)

    assert _error(r) == (409, 'assist_loop_running')
