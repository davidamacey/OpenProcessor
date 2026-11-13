"""Job recovery with REAL handlers (audit: recovery was only proven with the noop jobs).

Every check holds a real handler mid-run by freezing the one service it is waiting on
(``docker compose pause``: the cgroup freezer, so connections hang instead of failing), then hurts
the worker or the API and asserts on the ``jobs`` / ``job_attempts`` rows and on the side effects in
OpenSearch:

* ``class_merge`` (real handler, OpenSearch relabel, not auto-resume) held on OpenSearch;
* ``class_projection_sync`` (real handler, OpenSearch rewrite, auto-resume), started by the reaper
  after a class rename, held on OpenSearch;
* ``vlm_label_batch`` (real handler, VLM calls) held on the fake VLM.

Policy under test (02b section 4): a lease that expires moves ``running`` to ``interrupted``
(reaper rule a); an ``interrupted`` job of an auto-resume kind is requeued (rule b) and finished by
the next claim; any other kind stays ``interrupted`` and a stale message does not run it again; a
cancelled queued job is never claimed. Docs: https://docs.docker.com/reference/cli/docker/compose/pause/
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from tests.live import _compose
from tests.live._jobs import cancel_job_in_container
from tests.live.conftest import INDEXES, JOB_TERMINAL, get_doc, refresh, search, wait_until


if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.live

INTERRUPT_WITHIN_S = 150
REAPER_TICK_S = 30


def _job(api_client: Any, job_id: str) -> dict[str, Any]:
    resp = api_client.get(f'/jobs/{job_id}')
    assert resp.status_code == 200, resp.text
    job: dict[str, Any] = resp.json()
    return job


def _status(api_client: Any, job_id: str) -> str:
    return str(_job(api_client, job_id)['status'])


def _wait_status(
    api_client: Any, job_id: str, wanted: set[str], timeout_s: float
) -> dict[str, Any]:
    wait_until(lambda: _status(api_client, job_id) in wanted, timeout=timeout_s, interval=2.0)
    return _job(api_client, job_id)


def _attempts(job_id: str) -> list[tuple[str, str]]:
    """``(owner, outcome)`` per attempt, oldest first."""
    rows = _compose.psql(
        f"SELECT owner, coalesce(outcome, '') FROM job_attempts WHERE job_id = '{job_id}' "
        'ORDER BY attempt'
    )
    return [tuple(line.split('|'))[:2] for line in rows.splitlines() if line]  # type: ignore[misc]


def _all_healthy() -> bool:
    states = _compose.service_states()
    return all(
        states[s]['State'] == 'running' and states[s].get('Health') == 'healthy'
        for s in ('api', 'jobs-worker', 'periodic-worker', 'jobs-beat')
    )


@pytest.fixture
def worker_stack() -> Iterator[None]:
    """Healthy before and after: every service this file hurts is running and unfrozen."""

    def heal() -> None:
        for service in ('opensearch', 'fake-vlm', 'jobs-worker'):
            _compose.compose('unpause', service, check=False)
        _compose.compose(
            'up', '-d', '--no-deps', '--no-build',
            'api', 'jobs-worker', 'periodic-worker', 'jobs-beat',
            timeout=240,
        )  # fmt: skip
        # A thawed worker reports unhealthy until its next probe, so `up --wait` would refuse it.
        assert wait_until(_all_healthy, timeout=120, interval=3.0), _compose.service_states()
        # A finished merge leaves an outbox row; the reaper turns it into a projection-sync job on
        # a later tick, in the same exclusive group. Wait for the row to be done as well as for the
        # jobs table to be idle, or a job can appear after the idle check and block the next merge.
        assert wait_until(
            lambda: (
                _compose.psql(
                    'SELECT (SELECT count(*) FROM jobs '
                    "WHERE status IN ('queued', 'running', 'cancelling')) "
                    '+ (SELECT count(*) FROM search_projection_tasks WHERE done_at IS NULL)'
                )
                == '0'
            ),
            timeout=120,
            interval=3.0,
        ), 'jobs or projection tasks from an earlier check are still active'

    heal()
    yield
    heal()


def _crops(opensearch: Any, count: int) -> list[str]:
    """Unlabelled-by-a-person, non-holdout noise crops (each check takes its own)."""
    body = {
        'size': count,
        '_source': False,
        'query': {
            'bool': {
                'filter': [{'prefix': {'crop_id': 'noise'}}],
                'must_not': [{'term': {'test_holdout': True}}, {'term': {'class_validated': True}}],
            }
        },
        'sort': [{'crop_id': {'order': 'desc'}}],
    }
    hits = search(opensearch, INDEXES['items'], body)['hits']['hits']
    assert len(hits) == count, 'the harness has too few unlabelled noise crops left'
    return [h['_id'] for h in hits]


def _class_of(opensearch: Any, crop_id: str) -> dict[str, Any]:
    refresh(opensearch, INDEXES['items'])
    return dict(get_doc(opensearch, INDEXES['items'], crop_id)['_source'])


@pytest.fixture
def merge_pair(api_client: Any, opensearch: Any) -> dict[str, Any]:
    """Two new classes and three crops a person labelled into the source class."""
    names = [f'recov_{kind}_{uuid.uuid4().hex[:6]}' for kind in ('src', 'dst')]
    ids = []
    for name in names:
        created = api_client.post('/classes', json={'name': name, 'group': 'harness'})
        assert created.status_code == 201, created.text
        ids.append(created.json()['class_id'])
    crops = _crops(opensearch, 3)
    for crop in crops:
        put = api_client.put(f'/crops/{crop}/label', json={'class_id': ids[0]})
        assert put.status_code == 200, put.text
    return {'source': ids[0], 'target': ids[1], 'names': names, 'crops': crops}


def _queue_merge_while_the_worker_is_frozen(api_client: Any, pair: dict[str, Any]) -> str:
    """Freeze the (booted) worker so the job waits in the queue, then queue a merge."""
    _compose.compose('pause', 'jobs-worker')
    resp = api_client.post(
        '/classes/merge', json={'source_id': pair['source'], 'target_id': pair['target']}
    )
    assert resp.status_code == 202, resp.text
    job_id = str(resp.json()['id'])
    assert _status(api_client, job_id) == 'queued'
    return job_id


def _run_held_on(service: str, api_client: Any, job_id: str) -> None:
    """Freeze ``service``, thaw the worker, and return once the job is running (and stuck on it)."""
    _compose.compose('pause', service)
    _compose.compose('unpause', 'jobs-worker')
    job = _wait_status(api_client, job_id, {'running'}, 120)
    assert job['status'] == 'running', job


def test_sigkill_mid_class_merge_marks_it_interrupted_and_a_retry_finishes(
    api_client: Any, opensearch: Any, merge_pair: dict[str, Any], worker_stack: None
) -> None:
    job_id = _queue_merge_while_the_worker_is_frozen(api_client, merge_pair)
    _run_held_on('opensearch', api_client, job_id)
    _compose.kill('jobs-worker')
    killed_at = time.monotonic()
    _compose.compose('unpause', 'opensearch')

    job = _wait_status(api_client, job_id, set(JOB_TERMINAL), INTERRUPT_WITHIN_S)
    waited = time.monotonic() - killed_at
    assert job['status'] == 'interrupted', f'after {waited:.0f} s: {job}'
    assert [outcome for _, outcome in _attempts(job_id)] == ['lost']

    _compose.up('jobs-worker')  # the stale message may be redelivered: it must not run the job
    time.sleep(REAPER_TICK_S * 2)
    again = _job(api_client, job_id)
    assert again['status'] == 'interrupted', again
    assert len(_attempts(job_id)) == 1
    for crop in merge_pair['crops']:
        assert _class_of(opensearch, crop)['class_id'] == merge_pair['source'], (
            'an interrupted merge must not leave a partly relabelled item behind'
        )

    retry = api_client.post(
        '/classes/merge',
        json={'source_id': merge_pair['source'], 'target_id': merge_pair['target']},
    )
    assert retry.status_code == 202, retry.text
    done = _wait_status(api_client, str(retry.json()['id']), set(JOB_TERMINAL), 120)
    assert done['status'] == 'completed', done
    for crop in merge_pair['crops']:
        assert _class_of(opensearch, crop)['class_id'] == merge_pair['target']


def test_cancelling_a_queued_job_means_it_never_runs(
    api_client: Any, opensearch: Any, merge_pair: dict[str, Any], worker_stack: None
) -> None:
    job_id = _queue_merge_while_the_worker_is_frozen(api_client, merge_pair)
    cancelled = cancel_job_in_container(job_id)
    assert cancelled['status'] == 'cancelled'

    _compose.compose(
        'unpause', 'jobs-worker'
    )  # the message is still in the queue: the claim must refuse it
    time.sleep(REAPER_TICK_S)
    job = _job(api_client, job_id)
    assert job['status'] == 'cancelled', job
    assert _attempts(job_id) == []
    for crop in merge_pair['crops']:
        assert _class_of(opensearch, crop)['class_id'] == merge_pair['source']
    state = api_client.get('/classes').json()['classes']
    source = next(c for c in state if c['class_id'] == merge_pair['source'])
    assert not source.get('deprecated')


@pytest.mark.parametrize('how', ['sigkill', 'cold_stop'])
def test_an_auto_resume_job_is_requeued_and_finished_after_the_worker_dies(
    how: str, api_client: Any, opensearch: Any, merge_pair: dict[str, Any], worker_stack: None
) -> None:
    """A rename queues a ``class_projection_sync`` (auto-resume); the worker dies mid-sync."""
    _compose.compose('pause', 'jobs-worker')
    renamed = f'{merge_pair["names"][0]}_renamed'
    began = _compose.psql('SELECT now()')
    put = api_client.put(f'/classes/{merge_pair["source"]}', json={'name': renamed})
    assert put.status_code == 200, put.text

    def sync_job() -> str:
        return _compose.psql(
            "SELECT id FROM jobs WHERE kind = 'class_projection_sync' "
            f"AND created_at > '{began}' ORDER BY created_at DESC LIMIT 1"
        )

    job_id = wait_until(sync_job, timeout=REAPER_TICK_S * 3, interval=2.0)
    assert job_id, 'reaper rule e never started the projection sync of the rename'
    assert _status(api_client, job_id) == 'queued'
    _run_held_on('opensearch', api_client, job_id)

    if how == 'sigkill':
        _compose.kill('jobs-worker')
    else:
        # SIGTERM starts Celery's warm shutdown, which waits for the running task; Docker's SIGKILL
        # ends it at the grace. The grace must be shorter than the handler's 30 s OpenSearch client
        # timeout: with OpenSearch frozen, a longer one lets that timeout fail the job first
        # (measured: failed at 32-34 s with handler_error, 3 of 3 runs at a 75 s grace).
        _compose.stop('jobs-worker', grace_s=20)
    _compose.compose('unpause', 'opensearch')
    _compose.up('jobs-worker')

    finished = {'completed', 'completed_with_errors', 'failed', 'cancelled'}  # not `interrupted`
    job = _wait_status(api_client, job_id, finished, INTERRUPT_WITHIN_S + 120)
    assert job['status'] in ('completed', 'completed_with_errors'), job
    outcomes = [outcome for _, outcome in _attempts(job_id)]
    assert len(outcomes) == 2, outcomes
    assert outcomes[0] in ('lost', 'interrupted'), outcomes
    assert job['attempt'] == 2
    for crop in merge_pair['crops']:
        assert _class_of(opensearch, crop)['class_name'] == renamed


def test_an_api_restart_mid_job_leaves_the_job_running_and_readable(
    api_client: Any, opensearch: Any, worker_stack: None
) -> None:
    crops = _crops(opensearch, 2)
    _compose.compose('pause', 'fake-vlm')
    resp = api_client.post('/vlm/label_batch', json={'crop_ids': crops})
    assert resp.status_code == 202, resp.text
    job_id = str(resp.json()['job_id'])
    job = _wait_status(api_client, job_id, {'running'}, 120)
    assert job['status'] == 'running', job

    _compose.compose('restart', 'api', timeout=240)
    _compose.up('api')
    during = _job(api_client, job_id)
    assert during['status'] == 'running', during
    assert during['attempt'] == 1

    _compose.compose('unpause', 'fake-vlm')
    done = _wait_status(api_client, job_id, set(JOB_TERMINAL), 180)
    assert done['status'] == 'completed', done
    assert len(_attempts(job_id)) == 1
    for crop in crops:
        assert _class_of(opensearch, crop)['class_source'] == 'vlm'
