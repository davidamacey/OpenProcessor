"""The control-plane live checks: the stage-3 exit and the PR1 smoke (overview section 9, T2).

Needs the live harness with its Postgres, Valkey, db-migrate, jobs-worker, periodic-worker and
jobs-beat services (``docker/test/compose.yml``). Run, from the repo root::

    docker compose -p op-live-verify -f docker/test/compose.yml up -d --wait
    .venv/bin/python -m pytest tests/live/test_control_plane_smoke.py -v --no-cov -m live

``VERIFY_COMPOSE_PROJECT`` names another harness project. The destructive checks (worker kill,
broker loss, database restart) put the stack back and prove it consumes jobs again before the next
one starts. ``/ready`` is judged per service (``services.database.ok``, ``services.queue.ok``): the
harness has no Triton, so its overall status is 503 by design.

The infra jobs are deployment-wide, so their ``job.state`` events ride the global stream
(``/curation/events``); the project stream carries only the project's own events.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from tests.live import _compose
from tests.live._jobs import (
    StartFailedError,
    get_job,
    start_infra_job,
    start_infra_job_undispatched,
    wait_for_job,
)
from tests.live._sse import SseCollector
from tests.live.conftest import API_PREFIX, API_URL, PROJECT_PREFIX, wait_until


if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.live

EVENTS_URL = f'{API_URL}{API_PREFIX}/events'
TERMINAL = {'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted'}
INTERRUPT_WITHIN_S = 150
REAPER_TICK_S = 30
SSE_TICK_S = 2


def _ready(api: httpx.Client) -> tuple[int, dict[str, Any]]:
    resp = api.get('/ready', timeout=20.0)
    return resp.status_code, resp.json()


def _service_ok(api: httpx.Client, name: str) -> bool | None:
    try:
        return _ready(api)[1]['services'][name]['ok']
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def _wait_service(api: httpx.Client, name: str, ok: bool, timeout_s: float) -> bool:
    return bool(wait_until(lambda: _service_ok(api, name) is ok, timeout=timeout_s, interval=1.0))


def _running_with_progress(api: httpx.Client, job_id: str) -> dict[str, Any]:
    def view() -> dict[str, Any] | None:
        job = get_job(api, job_id)
        return job if job and job['status'] == 'running' and job['progress'].get('done') else None

    job = wait_until(view, timeout=90, interval=1.0)
    assert job, f'job {job_id} never reported progress while running: {get_job(api, job_id)}'
    return job


def _assert_status(job: dict[str, Any] | None, expected: str, note: str = '') -> None:
    assert job is not None, f'the job is not in GET /jobs. {note}'
    assert job['status'] == expected, f'{note} {job}'


def _states(events: list[dict[str, Any]]) -> list[str]:
    return [e['data']['status'] for e in events if e.get('event') == 'job.state']


def _ensure_stack(api: httpx.Client) -> None:
    """Every service up, database and queue ready, and a noop completes (the worker consumes)."""
    for service in ('postgres', 'valkey', 'jobs-worker', 'periodic-worker', 'jobs-beat'):
        _compose.up(service)
    assert _wait_service(api, 'database', True, 90), 'database not ready'
    assert _wait_service(api, 'queue', True, 90), 'queue not ready'
    job_id = start_infra_job('noop')
    job = wait_for_job(api, job_id, TERMINAL, timeout_s=90)
    _assert_status(job, 'completed', 'the stack does not run jobs')


@pytest.fixture
def intact_stack(api: httpx.Client) -> Iterator[None]:
    """For the destructive checks: healthy before, and healed afterwards even when one fails."""
    _ensure_stack(api)
    yield
    _ensure_stack(api)


# --- stage 3 exit (03c "Stage 3 exit criteria") ---------------------------------------------


def test_stage3_exit_a_noop_through_start_job_completes(api: httpx.Client) -> None:
    """Stage-3 exit A: op.infra.noop started through start_job ends `completed`."""
    job_id = start_infra_job('noop')
    job = wait_for_job(api, job_id, TERMINAL, timeout_s=60)
    assert job is not None, 'the job is not in GET /jobs'
    assert job['status'] == 'completed', job
    assert job['result'] == {'noop': True}
    assert job['attempt'] == 1


def test_stage3_exit_b_job_state_events_arrive_on_the_stream(api: httpx.Client) -> None:
    """Stage-3 exit B: subscribed first, the noop's job.state events arrive through `completed`.

    The noop is deployment-wide, so the stream is the global one (``project: null``).
    """
    sse = SseCollector(EVENTS_URL).start()
    try:
        job_id = start_infra_job('noop')
        assert sse.wait_for(
            lambda ev: any(
                e.get('event') == 'job.state'
                and e['data']['job_id'] == job_id
                and e['data']['status'] == 'completed'
                for e in ev
            ),
            timeout_s=60,
        ), f'no completed job.state for {job_id}: {sse.for_job(job_id)} error={sse.error}'
        mine = sse.for_job(job_id)
        assert _states(mine) == ['queued', 'running', 'completed']
        assert all(e['data']['project'] is None for e in mine)
    finally:
        sse.close()


def test_stage3_exit_c_killed_worker_leaves_the_job_interrupted(
    api: httpx.Client, intact_stack: None
) -> None:
    """Stage-3 exit C: kill the jobs-worker mid sleep_noop; the row is `interrupted` in 150 s."""
    job_id = start_infra_job('sleep_noop', seconds=120)
    _running_with_progress(api, job_id)
    _compose.kill('jobs-worker')
    killed_at = time.monotonic()
    job = wait_for_job(api, job_id, TERMINAL, timeout_s=INTERRUPT_WITHIN_S, interval_s=2.0)
    waited = time.monotonic() - killed_at
    _assert_status(job, 'interrupted', f'after {waited:.0f} s')
    assert waited <= INTERRUPT_WITHIN_S
    _compose.start('jobs-worker')


# --- PR1 smoke (overview section 9, T2 table) -----------------------------------------------


def test_step1_bring_up_database_and_queue_ready_and_migration_exited_zero(
    api: httpx.Client,
) -> None:
    """T2 step 1: /ready shows database ok and queue ok; db-migrate exited 0."""
    status, body = _ready(api)
    assert body['services']['database']['ok'] is True, body['services']['database']
    assert body['services']['queue']['ok'] is True, body['services']['queue']
    assert status in (200, 503)  # 503 only for the Triton the harness does not run
    migrate = _compose.service_states()['db-migrate']
    assert (migrate['State'], migrate['ExitCode']) == ('exited', 0), migrate


def test_step8_broker_loss_is_survived_and_recovered(api: httpx.Client, intact_stack: None) -> None:
    """T2 step 8: stop valkey and delete its volume, start it again.

    /ready names the queue (503) during and is fine after; a job committed `queued` during the
    outage is dispatched by the reaper (rule c); a running sleep_noop keeps its lease; SSE events
    still arrive (the 2 s safety tick).
    """
    running_id = start_infra_job('sleep_noop', seconds=100)
    _running_with_progress(api, running_id)
    sse = SseCollector(EVENTS_URL).start()
    try:
        _compose.remove_with_volumes('valkey')
        assert _wait_service(api, 'queue', False, 30), 'the queue stayed ready without valkey'
        status, body = _ready(api)
        assert status == 503
        assert body['status'] == 'not_ready'
        assert body['services']['queue']['ok'] is False

        queued_id = start_infra_job_undispatched('noop')
        assert sse.wait_for(
            lambda ev: 'queued' in _states([e for e in ev if e['data'].get('job_id') == queued_id]),
            timeout_s=SSE_TICK_S * 5,
        ), f'no SSE event while valkey is down: error={sse.error}'
        during = get_job(api, running_id)
        _assert_status(during, 'running', 'the lease was lost during the broker outage')

        _compose.up('valkey')
        assert _wait_service(api, 'queue', True, 60), 'queue not ready after valkey came back'

        job = wait_for_job(api, queued_id, TERMINAL, timeout_s=REAPER_TICK_S * 4)
        _assert_status(job, 'completed', 'reaper rule c did not redispatch the queued job')
        after = wait_for_job(api, running_id, TERMINAL, timeout_s=150)
        _assert_status(after, 'completed')
        assert after is not None
        assert after['attempt'] == 1, 'the running job lost its lease and was re-run'
    finally:
        sse.close()


def test_step9_postgres_restart_mid_job(api: httpx.Client, intact_stack: None) -> None:
    """T2 step 9: restart Postgres under a running sleep_noop.

    During the outage a start fails, the DB-backed routes answer 503, /ready names the database
    and /live stays 200; afterwards the database is ready and the job completes.
    """
    job_id = start_infra_job('sleep_noop', seconds=90)
    _running_with_progress(api, job_id)
    _compose.stop('postgres')
    try:
        assert _wait_service(api, 'database', False, 30), 'database stayed ready while stopped'
        status, body = _ready(api)
        assert status == 503
        assert body['services']['database']['ok'] is False
        assert api.get('/live', timeout=20.0).status_code == 200
        jobs = api.get(f'{PROJECT_PREFIX}/jobs', timeout=30.0)
        assert jobs.status_code == 503, jobs.text
        assert 'database' in jobs.json().get('code', ''), jobs.text
        with pytest.raises(StartFailedError):
            start_infra_job('noop')
    finally:
        _compose.start('postgres')
    assert _wait_service(api, 'database', True, 90), 'database not ready after restart'
    assert api.get('/live', timeout=20.0).status_code == 200
    job = wait_for_job(api, job_id, TERMINAL, timeout_s=180)
    _assert_status(job, 'completed', 'neither completed nor resumed after the restart')


def test_step11_sse_resume_without_loss_or_duplicates_then_reset(api: httpx.Client) -> None:
    """T2 step 11: disconnect mid-job, resume with Last-Event-ID, lose and repeat nothing.

    A cursor older than the retained minimum then gets exactly one `reset` event.
    """
    first = SseCollector(EVENTS_URL).start()
    try:
        job_id = start_infra_job('sleep_noop', seconds=12)
        assert first.wait_for(
            lambda ev: (
                sum(
                    1
                    for e in ev
                    if e.get('event') == 'job.progress' and e['data']['job_id'] == job_id
                )
                >= 3
            ),
            timeout_s=60,
        ), f'no progress events: {first.for_job(job_id)} error={first.error}'
    finally:
        first.close()
    seen_first = first.for_job(job_id)
    last_id = first.events[-1]['id']  # the stream's last frame; the resume starts after it

    second = SseCollector(EVENTS_URL, last_event_id=last_id).start()
    try:
        assert second.wait_for(
            lambda ev: 'completed' in _states([e for e in ev if e['data'].get('job_id') == job_id]),
            timeout_s=60,
        ), f'terminal state not delivered after resume: {second.for_job(job_id)}'
    finally:
        second.close()
    seen_second = second.for_job(job_id)

    ids = [e['id'] for e in seen_first + seen_second]
    assert len(ids) == len(set(ids)), f'duplicated events across the resume: {ids}'
    assert not any(e.get('event') == 'reset' for e in second.events)
    stored = _compose.psql(f"SELECT id FROM events WHERE job_id = '{job_id}' ORDER BY id")
    stored_ids = [int(i) for i in stored.splitlines()]
    assert sorted(int(i.split('-')[1]) for i in ids) == stored_ids, (
        f'events lost or invented: stream {ids} vs stored {stored_ids}'
    )

    ancient = SseCollector(EVENTS_URL, last_event_id='1-1').start()
    try:
        assert ancient.wait_for(bool, timeout_s=15)
        time.sleep(SSE_TICK_S * 2)  # a second reset, if the server were to send one, would be here
        events = list(ancient.events)
    finally:
        ancient.close()
    resets = [e for e in events if e.get('event') == 'reset']
    assert len(resets) == 1, events
    assert events[0].get('event') == 'reset'
    assert resets[0]['data']['reason'] == 'cursor_expired'
