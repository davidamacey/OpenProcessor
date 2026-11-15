"""The source file queue on PostgreSQL: idempotent enqueue, disjoint claims, retry and dead rows."""

from __future__ import annotations

import threading
import uuid
from typing import TYPE_CHECKING

import pytest
from fixtures.job_rows import insert_row, make_project, project_session
from sqlalchemy import text

from src.db.repos import ingest_sources as repo


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture
def source(pg: PgDatabase):
    """A project, one source and a job to claim for; returns ``(project, source id, job)``."""
    project = make_project(pg)
    job = insert_row(pg, project, 'running', kind='bulk_ingest_run', owner='w:1')
    with project_session(pg.engine('app'), project) as session:
        row = repo.insert(
            session,
            project,
            name=f's{uuid.uuid4().hex[:8]}',
            kind='folder',
            config={'path': '/x'},
            schedule={},
        )
    return project, row.id, job


def enqueue(pg: PgDatabase, project, source_id, keys):
    with project_session(pg.engine('app'), project) as session:
        return repo.enqueue(session, source_id, keys)


def test_enqueue_is_idempotent_by_key(pg, source) -> None:
    project, source_id, _ = source
    assert enqueue(pg, project, source_id, ['a', 'b', 'c']) == (3, 0)
    assert enqueue(pg, project, source_id, ['b', 'c', 'd', 'd']) == (1, 3)
    with project_session(pg.engine('app'), project) as session:
        assert repo.counts(session, source_id)['pending'] == 4


def test_two_runners_claim_disjoint_batches_without_waiting(pg, source) -> None:
    project, source_id, job = source
    enqueue(pg, project, source_id, [f'f{i:04}' for i in range(100)])
    first, second = pg.engine('app'), pg.engine('app')
    claimed: dict[str, list[str]] = {}
    held = threading.Event()
    release = threading.Event()

    def runner(name: str, engine, hold: bool) -> None:
        with project_session(engine, project) as session:
            claimed[name] = [c.file_key for c in repo.claim(session, source_id, job, 30)]
            if hold:  # keep the first claim's transaction (and its row locks) open
                held.set()
                release.wait(10)

    a = threading.Thread(target=runner, args=('a', first, True))
    a.start()
    assert held.wait(10)
    runner('b', second, False)  # the rows a holds locked are skipped, not waited for
    release.set()
    a.join(10)
    assert len(claimed['a']) == 30
    assert len(claimed['b']) == 30
    assert set(claimed['a']).isdisjoint(claimed['b'])
    with project_session(pg.engine('app'), project) as session:
        counts = repo.counts(session, source_id)
    assert (counts['claimed'], counts['pending']) == (60, 40)


def test_settle_retry_and_dead_rows(pg, source) -> None:
    project, source_id, job = source
    enqueue(pg, project, source_id, ['ok', 'flaky', 'gone'])
    engine = pg.engine('app')
    with project_session(engine, project) as session:
        claimed = {c.file_key for c in repo.claim(session, source_id, job, 10)}
        assert claimed == {'ok', 'flaky', 'gone'}
        repo.mark_done(session, source_id, job, [('ok', 'img-1')])
        repo.mark_failed(
            session,
            source_id,
            job,
            [
                repo.Failure('flaky', repo.error_text('503', 'batch_error')),
                repo.Failure('gone', 'gone', terminal=True),
            ],
        )
        counts = repo.counts(session, source_id)
        assert counts == {'pending': 0, 'claimed': 0, 'done': 1, 'failed': 1, 'dead': 1}
        errors = {e['key']: e for e in repo.errors(session, source_id)}
        assert errors['gone']['state'] == 'dead'
        assert (errors['flaky']['state'], errors['flaky']['error']) == ('failed', '503')

    # a failed row with attempts left is claimed again; a dead one only after retry_dead
    with project_session(engine, project) as session:
        assert [c.file_key for c in repo.claim(session, source_id, job, 10)] == ['flaky']
        for _ in range(3):
            repo.mark_failed(session, source_id, job, [repo.Failure('flaky', 'again')])
        assert repo.counts(session, source_id)['dead'] == 2
        assert repo.retry_dead(session, source_id) == 2
        counts = repo.counts(session, source_id)
        assert (counts['pending'], counts['dead']) == (2, 0)
        page, total = repo.failures(session, source_id, job, offset=0, limit=10)
        assert (page, total) == ([], 0)


def test_a_dead_runs_claims_go_back_to_pending(pg, source) -> None:
    project, source_id, job = source
    enqueue(pg, project, source_id, ['a', 'b'])
    with project_session(pg.engine('app'), project) as session:
        repo.claim(session, source_id, job, 10)
        assert repo.release_claims(session, source_id) == 2
        assert repo.counts(session, source_id)['pending'] == 2


def test_state_keys_are_set_independently(pg, source) -> None:
    project, source_id, _ = source
    with project_session(pg.engine('app'), project) as session:
        repo.state_set(session, source_id, 'scan', {'pages': 2})
        repo.state_set(session, source_id, 'watermark', 12.5)
        repo.state_update(session, source_id, {'failures': 1, 'last_poll_error': None})
        repo.state_add(session, source_id, {'events.received': 3})
        repo.state_add(session, source_id, {'events.received': 2})
        row = repo.get(session, source_id)
    assert row is not None
    assert row.scan_state == {
        'scan': {'pages': 2},
        'watermark': 12.5,
        'failures': 1,
        'last_poll_error': None,
        'events.received': 5,
    }


def test_a_name_is_unique_per_project_and_a_delete_takes_the_files(pg, source) -> None:
    project, source_id, _ = source
    enqueue(pg, project, source_id, ['a'])
    with project_session(pg.engine('app'), project) as session:
        existing = repo.get(session, source_id)
        assert existing is not None
        with pytest.raises(repo.SourceNameTakenError):
            repo.insert(session, project, name=existing.name, kind='folder', config={}, schedule={})
        assert repo.delete_source(session, source_id) is True
        assert repo.counts(session, source_id)['pending'] == 0


def _attempts(session, source_id, key):
    return session.execute(
        text(
            'SELECT attempts, retry_at FROM ingest_source_files WHERE source_id = :s AND file_key = :k'
        ),
        {'s': source_id, 'k': key},
    ).one()


def test_a_transient_failure_keeps_the_attempts_and_waits_for_retry_at(pg, source) -> None:
    project, source_id, job = source
    enqueue(pg, project, source_id, ['a', 'b'])
    with project_session(pg.engine('app'), project) as session:
        repo.claim(session, source_id, job, 10)
        repo.mark_failed(
            session,
            source_id,
            job,
            [repo.Failure('a', 'engine down', retry_in_s=30.0), repo.Failure('b', 'bad')],
        )
        attempts_a, retry_a = _attempts(session, source_id, 'a')
        attempts_b, retry_b = _attempts(session, source_id, 'b')
        assert (attempts_a, attempts_b) == (0, 1)  # a transient failure is not an attempt
        assert retry_b is None  # any other kind is claimable at once, as before
        assert retry_a is not None
        # the claim takes b and skips a until its retry_at has passed
        assert [c.file_key for c in repo.claim(session, source_id, job, 10)] == ['b']
        rows, due = repo.waiting(session, source_id) or (0, 0.0)
        assert rows == 1
        assert 0 < due <= 30
        session.execute(
            text(
                "UPDATE ingest_source_files SET retry_at = now() - interval '1 second' "
                "WHERE source_id = :s AND file_key = 'a'"
            ),
            {'s': source_id},
        )
        assert repo.waiting(session, source_id) == (1, 0.0)
        claimed = repo.claim(session, source_id, job, 10)
        assert [(c.file_key, c.attempts) for c in claimed] == [('a', 0)]
        assert _attempts(session, source_id, 'a')[1] is None  # the claim clears it
        assert repo.waiting(session, source_id) is None


def test_transient_failures_never_make_a_row_dead_and_a_terminal_one_still_does(pg, source) -> None:
    project, source_id, job = source
    enqueue(pg, project, source_id, ['flaky', 'gone'])
    with project_session(pg.engine('app'), project) as session:
        for _ in range(repo.MAX_ATTEMPTS + 2):
            assert 'flaky' in {c.file_key for c in repo.claim(session, source_id, job, 10)}
            repo.mark_failed(session, source_id, job, [repo.Failure('flaky', 'x', retry_in_s=0.0)])
        repo.mark_failed(session, source_id, job, [repo.Failure('gone', 'x', terminal=True)])
        counts = repo.counts(session, source_id)
        assert (counts['failed'], counts['dead']) == (1, 1)
        assert _attempts(session, source_id, 'flaky')[0] == 0
        assert _attempts(session, source_id, 'gone')[0] == repo.MAX_ATTEMPTS
        # a dead row never waits; a requeue clears a pause
        assert repo.waiting(session, source_id) is not None
        repo.requeue_all(session, source_id)
        assert repo.waiting(session, source_id) is None
        assert _attempts(session, source_id, 'flaky')[1] is None
