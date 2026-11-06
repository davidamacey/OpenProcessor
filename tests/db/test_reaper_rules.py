"""Each reaper rule acts on what it selects and on nothing else (02b section 4)."""

from __future__ import annotations

import datetime as dt
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest
from fixtures.job_rows import insert_row, make_project, sql
from fixtures.reaper_support import Broker, environment, reaper, row, status

from src.clients import task_queue
from src.config.job_kinds import KINDS
from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


if TYPE_CHECKING:
    import uuid


pytestmark = pytest.mark.db

PAST = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)


@pytest.fixture
def broker() -> Broker:
    return Broker()


@pytest.fixture
def env(pg, pg_env, monkeypatch):
    yield from environment(pg, monkeypatch)


# --- a clean database -----------------------------------------------------------------------


def test_a_clean_database_gives_no_writes(env, broker) -> None:
    project = make_project(env)
    fresh = insert_row(env, project, 'running')
    queued = insert_row(env, project, 'queued', last_dispatched_at=dt.datetime.now(dt.UTC))
    before = sql(env, 'SELECT id, updated_at, status FROM jobs ORDER BY id')
    report = reaper(broker).tick()
    assert not report.acted()
    assert sql(env, 'SELECT id, updated_at, status FROM jobs ORDER BY id') == before
    assert (status(env, fresh), status(env, queued)) == ('running', 'queued')
    assert broker.sent == []


# --- rule a ---------------------------------------------------------------------------------


def test_rule_a_interrupts_an_expired_running_job_and_marks_the_attempt_lost(env, broker) -> None:
    project = make_project(env)
    job = insert_row(env, project, 'running', lease_expires_at=PAST, attempt=1)
    sql(
        env,
        "INSERT INTO job_attempts (job_id, attempt, fencing_token, owner) VALUES (%s, 1, 1, 'w:1')",
        job,
    )
    stale = insert_row(env, project, 'cancelling', lease_expires_at=PAST, attempt=1)
    report = reaper(broker).tick()
    assert report.lost == 2
    assert row(env, job)[:2] == ('interrupted', 1)
    assert row(env, job)[4] is None
    assert status(env, stale) == 'cancelled'
    assert sql(env, 'SELECT outcome FROM job_attempts WHERE job_id = %s', job) == [('lost',)]
    events = sql(env, "SELECT payload->>'status' FROM events WHERE job_id = %s", job)
    assert events == [('interrupted',)]


def test_a_fresh_lease_is_untouched(env, broker) -> None:
    project = make_project(env)
    job = insert_row(env, project, 'running')
    reaper(broker).tick()
    assert status(env, job) == 'running'


# --- rule b ---------------------------------------------------------------------------------


def test_rule_b_requeues_an_interrupted_auto_resume_job_and_dispatches_it(env, broker) -> None:
    project = make_project(env)
    job = insert_row(env, project, 'interrupted', auto_resume=True, updated_at=PAST, retry_count=1)
    report = reaper(broker).tick()
    assert report.resumed == 1
    assert row(env, job)[0] == 'queued'
    assert broker.sent == [('noop', job)] if 'noop' in KINDS else True
    assert row(env, job)[5] == 1  # dispatch recorded


def test_rule_b_leaves_the_others_alone(env, broker) -> None:
    project = make_project(env)
    manual = insert_row(env, project, 'interrupted', auto_resume=False, updated_at=PAST)
    recent = insert_row(env, project, 'interrupted', auto_resume=True)
    reaper(broker).tick()
    assert (status(env, manual), status(env, recent)) == ('interrupted', 'interrupted')
    assert broker.sent == []


def test_rule_b_fails_an_exhausted_job_without_the_trigger_refusing(env, broker) -> None:
    project = make_project(env)
    job = insert_row(env, project, 'interrupted', auto_resume=True, updated_at=PAST, retry_count=3)
    report = reaper(broker).tick()
    assert report.exhausted == 1
    assert row(env, job)[:4:3] == ('failed', 'attempts_exhausted')
    assert broker.sent == []


def test_an_interrupted_job_whose_group_is_busy_does_not_block_the_batch(env, broker) -> None:
    project = make_project(env)
    insert_row(env, project, 'running', exclusive_group='g', scope='')  # the group is busy
    blocked = insert_row(
        env, project, 'interrupted', auto_resume=True, updated_at=PAST, exclusive_group='g'
    )
    free = insert_row(env, project, 'interrupted', auto_resume=True, updated_at=PAST)
    exhausted = insert_row(
        env,
        project,
        'interrupted',
        auto_resume=True,
        updated_at=PAST,
        retry_count=3,
        exclusive_group='g',
    )
    reaper(broker).tick()
    assert status(env, blocked) == 'interrupted'
    assert status(env, free) == 'queued'
    assert status(env, exhausted) == 'failed'  # exhausted fails even while its group is busy


def test_two_interrupted_jobs_of_one_group_requeue_one_and_skip_the_other(env, broker) -> None:
    """Both pass the busy test; the second hits the singleton index inside its savepoint (R-5)."""
    project = make_project(env)
    first = insert_row(
        env, project, 'interrupted', auto_resume=True, updated_at=PAST, exclusive_group='g'
    )
    second = insert_row(
        env,
        project,
        'interrupted',
        auto_resume=True,
        updated_at=PAST + dt.timedelta(seconds=1),
        exclusive_group='g',
    )
    other = insert_row(env, project, 'interrupted', auto_resume=True, updated_at=PAST)
    report = reaper(broker).tick()
    states = sorted([status(env, first), status(env, second)])
    assert states == ['interrupted', 'queued']  # one requeued, one left for a later tick
    assert status(env, other) == 'queued'  # the batch was not rolled back
    assert report.resumed == 2


def test_two_reapers_at_once_never_requeue_one_job_twice(env) -> None:
    project = make_project(env)
    jobs = {
        insert_row(env, project, 'interrupted', auto_resume=True, updated_at=PAST)
        for _ in range(30)
    }
    brokers = [Broker(), Broker()]
    barrier = threading.Barrier(2)

    def run(broker: Broker) -> None:
        barrier.wait(10)
        reaper(broker).tick()

    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(run, b) for b in brokers]:
            future.result(60)
    sent = [job for b in brokers for _kind, job in b.sent]
    assert sorted(sent) == sorted(jobs)  # every job dispatched exactly once, none twice
    assert all(status(env, job) == 'queued' for job in jobs)
    assert all(row(env, job)[5] == 1 for job in jobs)


# --- rule c ---------------------------------------------------------------------------------


def test_rule_c_sends_a_queued_job_again_when_its_marker_is_gone(env, broker) -> None:
    project = make_project(env)
    never = insert_row(env, project, 'queued')
    old = insert_row(env, project, 'queued', last_dispatched_at=PAST, dispatch_count=1)
    recent = insert_row(env, project, 'queued', last_dispatched_at=dt.datetime.now(dt.UTC))
    report = reaper(broker).tick()
    assert report.redispatched == 2
    assert sorted(job for _kind, job in broker.sent) == sorted([never, old])
    assert row(env, never)[5] == 1
    assert row(env, old)[5] == 2
    assert row(env, recent)[5] == 0


@pytest.mark.parametrize('marker', [task_queue.QueuedState.PRESENT, task_queue.QueuedState.UNKNOWN])
def test_rule_c_does_nothing_with_a_marker_present_or_unknown(env, broker, marker) -> None:
    project = make_project(env)
    job = insert_row(env, project, 'queued')
    broker.marker = marker
    reaper(broker).tick()
    assert broker.sent == []
    assert row(env, job)[5] == 0


def test_a_broker_that_is_down_leaves_the_row_queued(env, broker) -> None:
    project = make_project(env)
    job = insert_row(env, project, 'queued')
    broker.down = True
    reaper(broker).tick()
    assert (status(env, job), row(env, job)[5]) == ('queued', 0)


# --- rule d ---------------------------------------------------------------------------------


def test_rule_d_fails_a_job_that_waited_past_its_limit(env, broker) -> None:
    project = make_project(env)
    late = insert_row(
        env,
        project,
        'queued',
        queued_at=PAST,
        max_queue_wait=dt.timedelta(minutes=5),
        last_dispatched_at=dt.datetime.now(dt.UTC),
    )
    waiting = insert_row(env, project, 'queued', last_dispatched_at=dt.datetime.now(dt.UTC))
    report = reaper(broker).tick()
    assert report.queue_wait_failed == 1
    assert row(env, late)[:4:3] == ('failed', 'queue_wait_exceeded')
    assert status(env, waiting) == 'queued'


# --- rules e and i --------------------------------------------------------------------------


@pytest.fixture
def sync_kinds(monkeypatch):
    for name, group in (
        ('class_projection_sync', 'class_registry_write'),
        ('cluster_projection_sync', 'cluster_projection'),
    ):
        kind = JobKind(
            name=name,
            task_name=f'op.test.{name}',
            queue=TaskQueues.JOBS,
            project_scope='project',
            soft_limit_s=3600,
            hard_limit_s=4500,
            max_queue_wait_s=3600,
            exclusive_group=group,
        )
        monkeypatch.setitem(KINDS, name, kind)


def projection_task(env, project, kind='class_rename') -> uuid.UUID:
    [(task,)] = sql(
        env,
        "INSERT INTO search_projection_tasks (project_id, kind, payload) VALUES (%s, %s, '{}') RETURNING id",
        project,
        kind,
    )
    return task


def test_rule_e_starts_the_sync_job_of_a_projection_task_once(env, broker, sync_kinds) -> None:
    project = make_project(env)
    task = projection_task(env, project)
    report = reaper(broker).tick()
    assert report.projection_started == 1
    [(job_id, kind, key, requested_by)] = sql(
        env,
        'SELECT j.id, j.kind, j.idempotency_key, j.requested_by FROM search_projection_tasks t '
        'JOIN jobs j ON j.id = t.job_id WHERE t.id = %s',
        task,
    )
    assert (kind, key, requested_by) == ('class_projection_sync', f'e:{task}', 'reaper')
    assert broker.sent == [('class_projection_sync', job_id)]
    reaper(broker).tick()  # a second tick finds the task linked: no second job
    assert sql(env, "SELECT count(*) FROM jobs WHERE kind = 'class_projection_sync'") == [(1,)]


def test_rule_e_waits_while_the_group_is_busy(env, broker, sync_kinds) -> None:
    project = make_project(env)
    insert_row(
        env,
        project,
        'running',
        kind='class_projection_sync',
        exclusive_group='class_registry_write',
    )
    task = projection_task(env, project)
    reaper(broker).tick()
    assert sql(env, 'SELECT job_id FROM search_projection_tasks WHERE id = %s', task) == [(None,)]


def test_rule_e_does_nothing_for_a_kind_nobody_registered(env, broker) -> None:
    project = make_project(env)
    task = projection_task(env, project)
    reaper(broker).tick()
    assert sql(env, 'SELECT job_id FROM search_projection_tasks WHERE id = %s', task) == [(None,)]
    assert broker.sent == []


def test_rule_i_starts_the_cluster_projection_sync(env, broker, sync_kinds) -> None:
    project = make_project(env)
    task = projection_task(env, project, 'cluster_partition')
    reaper(broker).tick()
    [(kind, key)] = sql(
        env,
        'SELECT j.kind, j.idempotency_key FROM search_projection_tasks t JOIN jobs j ON j.id = t.job_id '
        'WHERE t.id = %s',
        task,
    )
    assert (kind, key) == ('cluster_projection_sync', f'i:{task}')


# --- rule g ---------------------------------------------------------------------------------


def test_rule_g_deletes_expired_stream_worker_claims(env, broker) -> None:
    project, other = make_project(env), make_project(env)
    for p, expires in ((project, '-1 minute'), (other, '+1 minute')):
        sql(
            env,
            'INSERT INTO stream_worker_claims (worker_kind, project_id, worker_id, lease_expires_at) '
            f"VALUES ('detection', %s, 'w', now() + interval '{expires}')",
            p,
        )
    report = reaper(broker).tick()
    assert report.claims_deleted == 1
    assert sql(env, 'SELECT project_id FROM stream_worker_claims') == [(other,)]


# --- rule f ---------------------------------------------------------------------------------


def test_rule_f_deletes_old_events_and_keeps_new_ones(env, broker) -> None:
    project = make_project(env)
    for age in ('25 hours', '2 hours'):
        sql(
            env,
            'INSERT INTO events (project_id, type, payload, created_at) '
            f"VALUES (%s, 'job.state', '{{}}', now() - interval '{age}')",
            project,
        )
    report = reaper(broker).tick()
    assert report.deleted['events'] == 1
    assert sql(env, 'SELECT count(*) FROM events WHERE project_id = %s', project) == [(1,)]


def test_rule_f_deletes_old_terminal_jobs_but_not_those_a_run_still_references(env, broker) -> None:
    project = make_project(env)
    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=100)
    plain = insert_row(env, project, 'completed', finished_at=old)
    kept = insert_row(env, project, 'completed', finished_at=old)
    young = insert_row(env, project, 'completed')
    live = insert_row(env, project, 'queued', last_dispatched_at=dt.datetime.now(dt.UTC))
    sql(
        env,
        'INSERT INTO training_runs (project_id, job_id, spec, class_names, family, base_model, '
        "artifact_prefix) VALUES (%s, %s, '{}', '{}', 'yolo', 'base', 'p/')",
        project,
        kept,
    )
    report = reaper(broker).tick()
    assert report.deleted['jobs'] == 1
    survivors = {r[0] for r in sql(env, 'SELECT id FROM jobs WHERE project_id = %s', project)}
    assert survivors == {kept, young, live}
    assert plain not in survivors


def test_rule_f_deletes_workers_and_devices_not_seen_for_a_day(env, broker) -> None:
    for name in ('old', 'new'):
        age = '2 days' if name == 'old' else '1 hour'
        sql(
            env,
            'INSERT INTO worker_capabilities (worker_id, queue, host, capabilities, seen_at) '
            f"VALUES (%s, 'jobs', 'h', '{{}}', now() - interval '{age}')",
            name,
        )
        sql(
            env,
            'INSERT INTO gpu_devices (gpu_uuid, host, nvml_index, name, compute_capability, '
            'memory_total_mib, hw_jpeg_decoder, driver_version, cuda_version, seen_at) '
            f"VALUES (%s, 'h', 0, 'g', '8.6', 1, false, 'd', 'c', now() - interval '{age}')",
            name,
        )
    report = reaper(broker).tick()
    assert report.deleted['worker_capabilities'] == 1
    assert report.deleted['gpu_devices'] == 1
    assert sql(env, 'SELECT worker_id FROM worker_capabilities') == [('new',)]


# --- the tick lock --------------------------------------------------------------------------


def test_a_tick_that_cannot_take_the_lock_stops(env, broker) -> None:
    import psycopg

    from src.config.advisory_locks import REAPER

    project = make_project(env)
    job = insert_row(env, project, 'running', lease_expires_at=PAST)
    with psycopg.connect(env.conninfo('super'), autocommit=True) as holder:
        holder.execute('SELECT pg_advisory_lock(%s, 0)', (REAPER,))
        report = reaper(broker).tick()
    assert report.skipped_busy
    assert status(env, job) == 'running'
    assert reaper(broker).tick().lost == 1  # the lock is free again
