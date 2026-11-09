"""Live auto-label loop: the whole path against a real OpenSearch, the loop worker and a fake VLM.

What this proves: the routes, the worker's job-file hand-off, the budget log, the lock rule at
write time, the propagated label kind on stored documents, and rollback. It proves nothing about
label quality (the fake VLM answers by a rule, see ``docker/test/fake_vlm.py``).

The loop runs in its own project (a fresh slug per run), seeded with the harness cohorts, so the cohorts
the other live modules assert on are never touched. Roots are the three seeded candidate
clusters; seven of their members are labeled by a person (class ``box``, the class the fake VLM
answers with, so the validated-member veto does not fire) and the freeze takes some of the first
four into the test holdout.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from .conftest import API_PREFIX, compose, refresh, search, wait_until


if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.live

# A deleted slug is retired for good, so each run takes a fresh one.
PROJECT = f'looplive{uuid.uuid4().hex[:6]}'
PREFIX = f'{API_PREFIX}/projects/{PROJECT}'
ITEMS = f'verify_prj_{PROJECT}__items'
COHORTS = ['cnd10000', 'cnd10001', 'cnd10002']
TERMINAL = ('completed', 'cancelled', 'failed', 'interrupted', 'paused')
ITEM_FIELDS = [
    'class_id',
    'class_name',
    'class_source',
    'class_validated',
    'class_detector_version',
    'test_holdout',
    'cluster_uid',
    'cluster_partition_id',
    'cluster_state',
    'pe_embedding',
]


def _request(loop_project: dict[str, Any], **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        'scope': {'cluster_uids': loop_project['root_uids'], 'min_root_size': 15},
        'budget': {'max_vlm_crops': 400},
        'recursion': {'max_depth': 1, 'branching': 2, 'min_node_size': 8},
        'seed': 7,
        'dry_run': False,
    }
    body.update(over)
    return body


def _docs(opensearch: Any, query: dict[str, Any], size: int = 500) -> dict[str, dict[str, Any]]:
    refresh(opensearch, ITEMS)
    body = {
        'size': size,
        'query': query,
        '_source': {'excludes': ['pe_embedding']},
        'sort': [{'crop_id': {'order': 'asc'}}],
    }
    hits = search(opensearch, ITEMS, body)['hits']['hits']
    return {h['_id']: h['_source'] for h in hits}


def _roots(opensearch: Any) -> dict[str, dict[str, Any]]:
    """Every item seeded into the three candidate cohorts."""
    return _docs(opensearch, {'prefix': {'crop_id': 'cnd1000'}})


def _serve_partition(api: Any, opensearch: Any) -> list[str]:
    """Cluster the project's items, serve the partition and return the uid of the cluster each
    seeded candidate cohort mostly landed in."""
    run = api.post(f'{PREFIX}/clusters/runs', json={'target': 'item'})
    assert run.status_code == 202, run.text

    def ready() -> dict[str, Any] | None:
        listed = api.get(f'{PREFIX}/clusters/partitions', params={'target': 'item'}).json()
        found = [p for p in listed['partitions'] if p['status'] in ('ready', 'failed')]
        return found[0] if found else None

    partition = wait_until(ready, timeout=300)
    assert partition is not None, 'no partition was recorded'
    assert partition['status'] == 'ready', partition
    if not partition['serving']:
        activated = api.put(
            f'{PREFIX}/clusters/activation',
            json={
                'partition_id': partition['id'],
                'expected_serving_partition_id': None,
                'expected_revision': None,
            },
        )
        assert activated.status_code == 202, activated.text

    def projected() -> bool:
        docs = _docs(opensearch, {'prefix': {'crop_id': 'cnd1000'}})
        return bool(docs) and all(
            d.get('cluster_partition_id') == partition['id'] for d in docs.values()
        )

    assert wait_until(projected, timeout=300), 'the partition was not projected onto the items'
    uids: list[str] = []
    for cohort in COHORTS:
        found = _docs(opensearch, {'prefix': {'crop_id': cohort}})
        counts: dict[str, int] = {}
        for doc in found.values():
            if doc.get('cluster_state') == 'assigned':
                counts[doc['cluster_uid']] = counts.get(doc['cluster_uid'], 0) + 1
        assert counts, f'no member of {cohort} was assigned to a cluster'
        uids.append(max(counts, key=lambda uid: counts[uid]))
    assert len(set(uids)) == len(COHORTS), uids
    return uids


@pytest.fixture(scope='module')
def loop_project(api: Any, opensearch: Any, seeded: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """A fresh project holding the seeded cohorts, with seven human labels, some of them frozen."""
    created = api.post(
        f'{API_PREFIX}/projects', json={'slug': PROJECT, 'display_name': 'Loop live'}
    )
    assert created.status_code in (200, 201), created.text
    proc = compose(
        'exec',
        '-T',
        'api',
        'python3',
        '/app/scripts/curation/seed_live_harness.py',
        '--project',
        PROJECT,
        check=False,
        timeout=900,
    )
    assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'
    # The endpoint registry is deployment-wide and the project needs its own activation.
    name = f'fake_{PROJECT}'
    made = api.post(
        f'{API_PREFIX}/vlm/endpoints',
        json={
            'name': name,
            'description': 'live loop test',
            'body': {'base_url': 'http://fake-vlm:8000/v1', 'model': 'fake-vlm'},
        },
    )
    assert made.status_code == 201, made.text
    active = api.post(
        f'{PREFIX}/vlm/endpoints/{name}/activate',
        json={'revision': 1, 'force': True, 'acknowledge_external': True},
    )
    assert active.status_code == 200, active.text
    root_uids = _serve_partition(api, opensearch)
    classes = api.get(f'{PREFIX}/classes').json()
    person_class = next(c['class_id'] for c in classes['classes'] if c['class_name'] == 'box')

    ids = sorted(
        crop_id
        for crop_id, doc in _docs(opensearch, {'prefix': {'crop_id': COHORTS[0]}}).items()
        if doc['cluster_uid'] == root_uids[0]
    )
    assert len(ids) >= 15, ids
    humans = ids[:7]
    for crop_id in humans[:4]:
        r = api.put(
            f'{PREFIX}/crops/{crop_id}/label',
            json={'class_id': person_class, 'label_source': 'human'},
        )
        assert r.status_code == 200, r.text
    # The freeze takes a share of the labels that exist now; the three below are labeled
    # afterwards, so the cluster holds both locked-and-held-out and only-locked items.
    frozen = api.post(f'{PREFIX}/test_holdout/freeze', json={'percent': 50})
    assert frozen.status_code == 200, frozen.text
    for crop_id in humans[4:]:
        r = api.put(
            f'{PREFIX}/crops/{crop_id}/label',
            json={'class_id': person_class, 'label_source': 'human'},
        )
        assert r.status_code == 200, r.text
    before = _roots(opensearch)
    held = [i for i in humans if before[i].get('test_holdout')]
    assert 1 <= len(held) <= 4, held
    yield {'humans': humans, 'held': held, 'before': before, 'root_uids': root_uids}
    deleted = api.delete(f'{API_PREFIX}/projects/{PROJECT}', params={'confirm': PROJECT})
    assert deleted.status_code == 202, deleted.text
    wait_until(
        lambda: all(
            p['slug'] != PROJECT for p in api.get(f'{API_PREFIX}/projects').json()['projects']
        ),
        timeout=60,
    )
    gone = api.delete(f'{API_PREFIX}/vlm/endpoints/{name}', params={'expected_revision': 1})
    assert gone.status_code == 204, gone.text


def _wait_terminal(api: Any, loop_id: str, timeout: float = 180.0) -> dict[str, Any]:
    def fetch() -> dict[str, Any]:
        resp = api.get(f'{PREFIX}/assist/loops/{loop_id}')
        assert resp.status_code == 200, resp.text
        return resp.json()

    deadline = time.monotonic() + timeout
    state = fetch()
    while state['status'] not in TERMINAL and time.monotonic() < deadline:
        time.sleep(1.0)
        state = fetch()
    return state


def _record(label: str, state: dict[str, Any]) -> None:
    """One line per loop in the run output (`-s`, or `-rP` for a passing run)."""
    vlm = state['budget']['vlm']
    print(
        f'LOOP-RECORD {label}: status={state["status"]} stop={state["stop_reason"]} '
        f'crops={vlm["crops_spent"]}/{vlm["max_crops"]} requests={vlm["requests_spent"]}/'
        f'{vlm["max_requests"]} propagated={state["totals"]["items_propagated"]} '
        f'direct={state["totals"]["items_direct"]} '
        f'review_clusters={state["totals"]["clusters_left_for_review"]} '
        f'review_items={state["totals"]["items_left_for_review"]} '
        f'skipped={state["totals"]["skipped"]}'
    )


def _assert_budget_respected(state: dict[str, Any]) -> None:
    _record(state['loop_id'], state)
    vlm = state['budget']['vlm']
    assert vlm['crops_spent'] + vlm['crops_reserved'] <= vlm['max_crops'], vlm
    assert vlm['requests_spent'] + vlm['requests_reserved'] <= vlm['max_requests'], vlm


def _assert_locked_items_untouched(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]], loop: dict[str, Any]
) -> None:
    for crop_id in loop['humans']:
        for field in ('class_id', 'class_name', 'class_source', 'class_validated', 'test_holdout'):
            assert after[crop_id].get(field) == before[crop_id].get(field), (crop_id, field)
    for crop_id in loop['held']:
        assert after[crop_id]['test_holdout'] is True
    for crop_id, doc in before.items():
        if doc.get('class_validated') or doc.get('test_holdout') or doc.get('class_excluded'):
            assert after[crop_id].get('class_name') == doc.get('class_name'), crop_id
            assert after[crop_id].get('class_source') == doc.get('class_source'), crop_id


def test_a_dry_run_sizes_the_loop_and_writes_nothing(
    api: Any, opensearch: Any, loop_project: dict[str, Any], fake_vlm: Any
) -> None:
    resp = api.post(f'{PREFIX}/assist/loops', json=_request(loop_project, dry_run=True))
    assert resp.status_code == 200, resp.text
    state = resp.json()
    assert state['status'] == 'dry_run'
    assert state['loop_id'] is None
    assert state['plan']['roots'] == 3
    # The members of the three clusters, minus the seven a person labeled (and what another
    # cluster of the run took from the cohorts).
    assert 0 < state['plan']['eligible_items'] <= 58, state['plan']
    assert state['plan']['fits_budget'] is True
    assert api.get(f'{PREFIX}/assist/loops').json() == {'loops': []}
    assert _roots(opensearch) == loop_project['before']
    assert fake_vlm.get('/__stats').json()['counts'] == {}


def test_a_loop_propagates_within_budget_leaves_locks_alone_and_rolls_back(
    api: Any, opensearch: Any, loop_project: dict[str, Any], fake_vlm: Any
) -> None:
    fake_vlm.post('/__reset')
    start = api.post(f'{PREFIX}/assist/loops', json=_request(loop_project))
    assert start.status_code == 200, start.text
    loop_id = start.json()['loop_id']
    assert loop_id
    # One loop per project: a second start names the live one.
    again = api.post(f'{PREFIX}/assist/loops', json=_request(loop_project))
    assert again.status_code == 409, again.text

    state = _wait_terminal(api, loop_id)
    assert state['status'] == 'completed', state
    assert state['stop_reason'] == 'done'
    _assert_budget_respected(state)
    seen = fake_vlm.get('/__seen').json()['hashes']
    assert state['budget']['vlm']['crops_spent'] == len(seen), 'every crop asked is a crop spent'
    assert state['totals']['items_propagated'] > 0, state['totals']
    assert state['audit']['status'] == 'off'

    after = _roots(opensearch)
    propagated = {i: d for i, d in after.items() if d.get('class_source') == 'assist_propagated'}
    assert len(propagated) == state['totals']['items_propagated']
    for doc in propagated.values():
        assert doc['class_detector_version'] == loop_id
        assert doc['class_name'] == 'box'
        assert doc['class_validated'] is False
    _assert_locked_items_untouched(loop_project['before'], after, loop_project)

    summary = {
        n['path']: n['decision']
        for n in api.get(f'{PREFIX}/assist/loops/{loop_id}/nodes').json()['nodes']
    }
    assert len(summary) == 3, summary
    assert set(summary.values()) == {'pass'}, summary
    holders = [
        path
        for path in summary
        if set(loop_project['humans'])
        <= set(api.get(f'{PREFIX}/assist/loops/{loop_id}/nodes/{path}/members').json()['crop_ids'])
    ]
    assert len(holders) == 1, 'members list every member, locked included'

    # The audit is off, said in the record rather than left blank.
    assert state['pinned']['audit_vlm'] is None

    dry = api.post(
        f'{PREFIX}/assist/loops/{loop_id}/rollback', json={'node_paths': None, 'dry_run': True}
    )
    assert dry.status_code == 200, dry.text
    assert dry.json()['restored'] == len(propagated)
    assert dry.json()['kept_changed'] == 0
    assert _roots(opensearch) == after, 'a rollback dry run writes nothing'

    # A person edits one propagated item after the loop: the rollback keeps it.
    edited = next(iter(propagated))
    r = api.put(f'{PREFIX}/crops/{edited}/label', json={'class_id': 1, 'label_source': 'human'})
    assert r.status_code == 200, r.text
    real = api.post(
        f'{PREFIX}/assist/loops/{loop_id}/rollback', json={'node_paths': None, 'dry_run': False}
    )
    assert real.status_code == 200, real.text
    assert real.json()['restored'] == len(propagated) - 1
    assert real.json()['kept_changed'] == 1

    restored = _roots(opensearch)
    for crop_id in propagated:
        if crop_id == edited:
            assert restored[crop_id]['class_source'] == 'human'
            continue
        assert restored[crop_id].get('class_source') != 'assist_propagated'
        was = loop_project['before'][crop_id]
        for field in ('class_id', 'class_name', 'class_source', 'class_validated'):
            assert restored[crop_id].get(field) == was.get(field), (crop_id, field)
    _assert_locked_items_untouched(loop_project['before'], restored, loop_project)
    nothing = api.post(
        f'{PREFIX}/assist/loops/{loop_id}/rollback', json={'node_paths': None, 'dry_run': True}
    )
    # What a person changed stays matched but is never restored; nothing else is left to restore.
    assert nothing.json()['restored'] == 0
    assert nothing.json()['kept_changed'] == 1


def test_a_cluster_that_fails_the_gate_is_split_then_left_for_review(
    api: Any, opensearch: Any, loop_project: dict[str, Any], fake_vlm: Any
) -> None:
    fake_vlm.post('/__reset')
    fake_vlm.post('/__control', json={'class_cycle': ['box', 'box', 'box', 'envelope']})
    before = _roots(opensearch)
    start = api.post(f'{PREFIX}/assist/loops', json=_request(loop_project, seed=11))
    assert start.status_code == 200, start.text
    loop_id = start.json()['loop_id']
    state = _wait_terminal(api, loop_id)
    assert state['status'] == 'completed', state
    _assert_budget_respected(state)

    nodes = api.get(f'{PREFIX}/assist/loops/{loop_id}/nodes').json()['nodes']
    decisions = {n['decision'] for n in nodes}
    assert 'split' in decisions, decisions
    assert decisions & {'review_leaf', 'review_small'}, decisions
    roots = {n['path']: n['decision'] for n in nodes if n['level'] == 0}
    assert len(roots) == 3, roots
    assert 'pass' not in roots.values(), roots
    assert state['totals']['clusters_left_for_review'] > 0
    assert state['totals']['items_left_for_review'] > 0
    after = _roots(opensearch)
    written = [d for d in after.values() if d.get('class_source') == 'assist_propagated']
    assert len(written) == state['totals']['items_propagated']
    _assert_locked_items_untouched(before, after, loop_project)
    # A review leaf lists its members for a person.
    leaf = next(n for n in nodes if n['decision'] in ('review_leaf', 'review_small'))
    members = api.get(f'{PREFIX}/assist/loops/{loop_id}/nodes/{leaf["path"]}/members').json()
    assert members['crop_ids']
    fake_vlm.post('/__reset')


def test_the_budget_is_a_hard_cap(
    api: Any, opensearch: Any, loop_project: dict[str, Any], fake_vlm: Any
) -> None:
    fake_vlm.post('/__reset')
    start = api.post(
        f'{PREFIX}/assist/loops',
        json=_request(loop_project, seed=13, budget={'max_vlm_crops': 40}),
    )
    assert start.status_code == 200, start.text
    state = _wait_terminal(api, start.json()['loop_id'])
    assert state['status'] == 'completed', state
    assert state['stop_reason'] == 'budget_crops', state['stop_reason']
    _assert_budget_respected(state)
    assert 0 < state['budget']['vlm']['crops_spent'] <= 40
    assert len(fake_vlm.get('/__seen').json()['hashes']) <= 40
