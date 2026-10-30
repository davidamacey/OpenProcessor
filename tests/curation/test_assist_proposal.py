"""Cluster proposals from an assist run: votes, the agreement gate, the scope
check after a re-refine, and the explicit apply (the only path to labels)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from curation.config_test_stack import ITEMS
from src.services.curation.assist.proposal import ClassAnswer, compute_proposal


REGISTRY = frozenset({'widget', 'gadget'})


def _answers(*names: str | None, **kw: Any) -> list[ClassAnswer]:
    return [ClassAnswer(class_name=n, **kw) for n in names]


def test_the_top_class_is_proposed_when_enough_answers_agree() -> None:
    p = compute_proposal(
        _answers('widget', 'widget', 'widget', 'gadget'), min_agreement=0.6, registry_names=REGISTRY
    )
    assert (p.class_name, p.in_registry, p.passes_gate) == ('widget', True, True)
    assert (p.answered, p.agreement) == (4, 0.75)
    assert [(d.class_name, d.votes) for d in p.distribution] == [('widget', 3), ('gadget', 1)]


def test_below_the_agreement_floor_no_class_is_proposed_but_the_votes_are_shown() -> None:
    p = compute_proposal(
        _answers('widget', 'widget', 'gadget', 'gadget'), min_agreement=0.6, registry_names=REGISTRY
    )
    assert (p.class_name, p.passes_gate) == (None, False)
    assert p.agreement == 0.5
    assert len(p.distribution) == 2


def test_fewer_than_three_answers_never_pass_however_unanimous() -> None:
    p = compute_proposal(_answers('widget', 'widget'), min_agreement=0.5, registry_names=REGISTRY)
    assert (p.answered, p.agreement, p.passes_gate, p.class_name) == (2, 1.0, False, None)


def test_a_class_outside_the_registry_is_proposed_but_flagged() -> None:
    p = compute_proposal(
        _answers('spoiler', 'spoiler', 'spoiler'), min_agreement=0.6, registry_names=REGISTRY
    )
    assert (p.class_name, p.in_registry) == ('spoiler', False)


def test_new_class_proposals_count_by_name_and_are_not_votes() -> None:
    answers = [
        *_answers('widget', 'widget', 'widget'),
        ClassAnswer(class_name=None, proposed='rear_wing'),
        ClassAnswer(class_name=None, proposed='rear_wing'),
        ClassAnswer(class_name=None, proposed='bumper'),
    ]
    p = compute_proposal(answers, min_agreement=0.6, registry_names=REGISTRY)
    assert [(n.name, n.votes) for n in p.new_class_proposals] == [('rear_wing', 2), ('bumper', 1)]
    assert p.answered == 6
    assert p.agreement == 0.5
    assert p.class_name is None


def test_no_answers_is_an_empty_proposal() -> None:
    p = compute_proposal([], min_agreement=0.6, registry_names=REGISTRY)
    assert (p.answered, p.agreement, p.passes_gate, p.distribution) == (0, 0.0, False, [])


def test_attribute_summary_counts_values_per_key() -> None:
    answers = [
        ClassAnswer(class_name='widget', attributes={'make': 'acme', 'colour': 'red'}),
        ClassAnswer(class_name='widget', attributes={'make': 'acme'}),
        ClassAnswer(class_name='widget', attributes={'make': 'borealis'}),
    ]
    p = compute_proposal(answers, min_agreement=0.6, registry_names=REGISTRY)
    by_key = {s.key: s for s in p.attribute_summary}
    assert [(v.value, v.count) for v in by_key['make'].values] == [('acme', 2), ('borealis', 1)]
    assert by_key['make'].answered == 3
    assert by_key['colour'].answered == 1


# ---- the run, the scope check and apply -------------------------------------------

RUNS = '/assist/runs'
CLUSTER = 10000


@pytest.fixture
def world(stack: Any, readable_crops: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry', lambda: stack.registry
    )
    monkeypatch.setenv('OP_ASSIST_JOBS_DIR', str(tmp_path / 'assist_jobs'))
    return stack


def _seed(stack: Any, crop_id: str, subid: str = f'{CLUSTER}a', **doc: Any) -> None:
    defaults = {
        'cluster_id': CLUSTER,
        'cluster_subid': subid,
        'class_source': 'item_model',
        'updated_at': '2026-10-01T00:00:00+00:00',
    }
    stack.seed(crop_id, **{**defaults, **doc})


def _wait(stack: Any, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = stack.client.get(f'/curation/projects/default{RUNS}/{run_id}').json()
        if state['status'] not in ('queued', 'running'):
            return state
        time.sleep(0.05)
    raise AssertionError('the run did not finish')


def _reply(*names: str) -> str:
    import json

    return json.dumps(
        {'results': [{'img': i, 'class': n, 'confidence': 'high'} for i, n in enumerate(names, 1)]}
    )


def _run_cluster(stack: Any, **body: Any) -> dict[str, Any]:
    payload = {
        'target': {'kind': 'cluster', 'cluster_id': CLUSTER},
        'actions': ['classify'],
        'dry_run': False,
        **body,
    }
    with stack.client:
        started = stack.post(RUNS, **payload)
        assert started.status_code == 200, started.text
        return _wait(stack, started.json()['run_id'])


def _doc(stack: Any, crop_id: str) -> dict[str, Any]:
    return dict(stack.items.docs(ITEMS)[crop_id])


def test_the_run_result_carries_the_proposal(world: Any) -> None:
    for i in range(3):
        _seed(world, f'c{i}')
    world.upstream.vlm_content = _reply('gadget', 'gadget', 'gadget')

    state = _run_cluster(world)

    proposal = state['result']['proposal']
    assert (proposal['class_name'], proposal['in_registry'], proposal['passes_gate']) == (
        'gadget',
        True,
        True,
    )
    assert proposal['answered'] == 3
    assert state['scope_now'] == {'count': 3, 'overlap': 3, 'jaccard': 1.0, 'changed': False}


def test_a_re_refine_marks_the_proposal_stale_with_the_overlap(world: Any) -> None:
    for i in range(3):
        _seed(world, f'c{i}')
    world.upstream.vlm_content = _reply('gadget', 'gadget', 'gadget')
    state = _run_cluster(world, target={'kind': 'cluster', 'cluster_id': CLUSTER})

    # a re-refine: one member leaves the cluster, another joins it
    docs = world.items.store[ITEMS]
    docs['c2']['cluster_id'] = 77
    _seed(world, 'newcomer')

    now = world.client.get(f'/curation/projects/default{RUNS}/{state["run_id"]}').json()
    assert now['scope_now']['changed'] is True
    assert (now['scope_now']['overlap'], now['scope_now']['count']) == (2, 3)
    assert now['scope_now']['jaccard'] == pytest.approx(0.5)


def test_apply_refuses_a_changed_scope_and_applies_when_allowed(world: Any) -> None:
    for i in range(3):
        _seed(world, f'c{i}')
    world.upstream.vlm_content = _reply('gadget', 'gadget', 'gadget')
    state = _run_cluster(world)
    world.items.store[ITEMS]['c2']['cluster_id'] = 77
    _seed(world, 'newcomer')
    url = f'{RUNS}/{state["run_id"]}/apply'

    refused = world.post(url, class_name='widget', dry_run=False)
    assert (refused.status_code, refused.json()['detail']['error']) == (409, 'scope_changed')

    allowed = world.post(url, class_name='widget', dry_run=False, allow_changed_scope=True)
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()['updated'] == 3  # c0, c1 and the newcomer: the CURRENT members
    assert _doc(world, 'newcomer')['class_name'] == 'widget'
    assert _doc(world, 'c2')['class_source'] == 'vlm', 'it left the scope: untouched'


def test_apply_dry_run_counts_and_skips_the_locked(world: Any) -> None:
    _seed(world, 'free1')
    _seed(
        world, 'human1', class_id=0, class_name='widget', class_source='human', class_validated=True
    )
    _seed(
        world, 'valid1', class_source='vlm', class_validated=True, class_id=1, class_name='gadget'
    )
    world.upstream.vlm_content = _reply('gadget', 'gadget', 'gadget')
    state = _run_cluster(world)
    before = world.snapshot()

    dry = world.post(f'{RUNS}/{state["run_id"]}/apply', class_name='widget')

    body = dry.json()
    assert dry.status_code == 200, dry.text
    assert (body['dry_run'], body['selected'], body['skipped_locked'], body['updated']) == (
        True,
        1,
        2,
        0,
    )
    assert world.snapshot() == before

    done = world.post(f'{RUNS}/{state["run_id"]}/apply', class_name='widget', dry_run=False)
    assert done.json()['updated'] == 1
    assert (_doc(world, 'free1')['class_name'], _doc(world, 'free1')['class_source']) == (
        'widget',
        'human',
    )
    assert _doc(world, 'free1')['class_validated'] is True
    assert _doc(world, 'human1')['class_name'] == 'widget'
    assert _doc(world, 'valid1')['class_name'] == 'gadget', 'a validated class is not overwritten'


def test_apply_as_propagated_writes_the_machine_kind_and_leaves_the_locked(world: Any) -> None:
    _seed(world, 'free1')
    _seed(
        world, 'human1', class_id=0, class_name='widget', class_source='human', class_validated=True
    )
    world.upstream.vlm_content = _reply('gadget', 'gadget', 'gadget')
    state = _run_cluster(world)
    url = f'{RUNS}/{state["run_id"]}/apply'

    refused = world.post(url, class_name='widget', label_source='human_confirmed')
    assert refused.status_code == 422

    done = world.post(url, class_name='widget', label_source='propagated', dry_run=False)

    assert done.status_code == 200, done.text
    assert done.json()['updated'] == 1
    free = _doc(world, 'free1')
    assert (free['class_name'], free['class_source'], free['class_validated']) == (
        'widget',
        'assist_propagated',
        False,
    )
    assert free['class_detector_version'] == state['run_id']
    assert free['cluster_id'] == free['class_id']
    assert _doc(world, 'human1')['class_source'] == 'human'


def test_apply_resolves_the_class_by_name(world: Any) -> None:
    _seed(world, 'free1')
    world.upstream.vlm_content = _reply('gadget')
    state = _run_cluster(world)
    url = f'{RUNS}/{state["run_id"]}/apply'

    missing = world.post(url, class_name='spoiler', dry_run=False)
    assert (missing.status_code, missing.json()['detail']['error']) == (
        409,
        'class_not_in_registry',
    )

    spaced = world.post(url, class_name='Gadget', dry_run=False)
    assert spaced.status_code == 200, spaced.text
    assert _doc(world, 'free1')['class_name'] == 'gadget'


def test_a_run_without_a_cluster_scope_cannot_be_applied(world: Any) -> None:
    _seed(world, 'free1')
    world.upstream.vlm_content = _reply('gadget')
    with world.client:
        started = world.post(
            RUNS,
            target={'kind': 'crops', 'crop_ids': ['free1']},
            actions=['classify'],
            dry_run=False,
        )
        state = _wait(world, started.json()['run_id'])

    r = world.post(f'{RUNS}/{state["run_id"]}/apply', class_name='gadget', dry_run=False)

    assert (r.status_code, r.json()['detail']['error']) == (409, 'run_has_no_scope')


def test_a_selection_with_a_context_proposes_for_that_sub_cluster(world: Any) -> None:
    for i in range(3):
        _seed(world, f'a{i}')
    _seed(world, 'b0', f'{CLUSTER}b')
    world.upstream.vlm_content = _reply('gadget', 'gadget', 'gadget')
    with world.client:
        started = world.post(
            RUNS,
            target={
                'kind': 'crops',
                'crop_ids': ['a0', 'a1', 'a2'],
                'context': {'cluster_id': CLUSTER, 'cluster_subid': f'{CLUSTER}a'},
            },
            actions=['classify'],
            dry_run=False,
        )
        state = _wait(world, started.json()['run_id'])
    assert state['scope_now']['count'] == 3, 'the context sub-cluster, not the selection'

    listed = world.client.get(
        f'/curation/projects/default{RUNS}',
        params={'cluster_id': CLUSTER, 'cluster_subid': f'{CLUSTER}a'},
    ).json()
    assert [r['run_id'] for r in listed['runs']] == [state['run_id']]
    other = world.client.get(
        f'/curation/projects/default{RUNS}',
        params={'cluster_id': CLUSTER, 'cluster_subid': f'{CLUSTER}b'},
    ).json()
    assert other['runs'] == []
