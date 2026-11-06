"""publish writes the row in the caller's transaction and wakes only after the commit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _events_support import count_events, count_global_events, seed_project

from src.config.project_context import ProjectNotBound
from src.db.repos.events import EventPayloadTooLargeError, UnknownProjectError, insert_event
from src.db.session import sync_project_tx, sync_session_factory
from src.services.events import publish as publish_module
from src.services.events.publish import GLOBAL, publish


pytestmark = pytest.mark.db


def test_a_rolled_back_transaction_writes_no_row_and_sends_no_wakeup(pg_env, broker) -> None:
    project = seed_project(pg_env, 'rolled-back')
    with sync_session_factory()() as session:
        try:
            with sync_project_tx(session, project):
                publish('job.state', {'state': 'running'}, project=project, session=session)
                raise RuntimeError('the caller fails after publishing')
        except RuntimeError:
            pass
    assert count_events(pg_env, project) == 0
    assert broker.published == []


def test_a_committed_transaction_writes_the_row_and_wakes_after_the_commit(pg_env, broker) -> None:
    project = seed_project(pg_env, 'committed')
    seen_at_wakeup: list[int] = []
    broker.on_publish = lambda _channel: seen_at_wakeup.append(count_events(pg_env, project))
    with sync_session_factory()() as session, sync_project_tx(session, project):
        publish('job.state', {'state': 'running'}, project=project, session=session)
        assert broker.published == [], 'no wake-up while the transaction is open'
    assert count_events(pg_env, project) == 1
    assert broker.published == [f'op:events:{project}']
    assert seen_at_wakeup == [1], 'the row was already committed when the wake-up went out'


def test_a_publish_without_a_session_commits_on_its_own(pg_env, broker) -> None:
    project = seed_project(pg_env, 'own-tx')
    publish('config.changed', {'topic': 'config'}, project='own-tx')
    assert count_events(pg_env, project) == 1
    assert broker.published == [f'op:events:{project}']


def test_a_global_publish_uses_the_global_channel(pg_env, broker) -> None:
    before = count_global_events(pg_env)
    publish('project.created', {'topic': 'project'}, project=GLOBAL)
    assert count_global_events(pg_env) == before + 1
    assert broker.published == ['op:events:global']


def test_an_unbound_publish_without_a_project_fails_closed(pg_env, broker, monkeypatch) -> None:
    def unbound():
        raise ProjectNotBound

    monkeypatch.setattr(publish_module, 'current_project', unbound)
    with pytest.raises(ProjectNotBound):
        publish('crop.created', {})


def test_a_failed_wakeup_is_ignored(pg_env, broker) -> None:
    project = seed_project(pg_env, 'valkey-down')
    broker.down = True
    publish('job.state', {}, project=project)
    assert count_events(pg_env, project) == 1


def test_a_payload_over_the_cap_is_refused_before_the_database(pg_env, broker) -> None:
    project = seed_project(pg_env, 'too-big')
    with pytest.raises(EventPayloadTooLargeError):
        publish('job.state', {'blob': 'x' * 7000}, project=project)
    assert count_events(pg_env, project) == 0


def test_an_unknown_slug_is_refused(pg_env) -> None:
    with (
        sync_session_factory()() as session,
        sync_project_tx(session, seed_project(pg_env, 'known')),
    ):
        with pytest.raises(UnknownProjectError):
            insert_event(session, 'nope', 'job.state', {})
        session.rollback()


def test_insert_event_accepts_a_slug_or_an_id(pg_env) -> None:
    project = seed_project(pg_env, 'either')
    with sync_session_factory()() as session, sync_project_tx(session, project):
        by_slug = insert_event(session, 'either', 'a', {})
        by_id = insert_event(session, project, 'b', {})
    assert by_slug.project_id == by_id.project_id == project
    assert count_events(pg_env, project) == 2


def test_a_publish_for_the_bound_project_uses_its_id_not_a_slug_lookup(
    pg_env, broker, monkeypatch
) -> None:
    project = seed_project(pg_env, 'bound-by-id')
    # The bound record carries the project's id; its slug names no row, so a slug lookup would fail.
    bound = SimpleNamespace(record=SimpleNamespace(slug='no-such-slug', id=project))
    monkeypatch.setattr(publish_module, 'current_project', lambda: bound)
    publish('crop.created', {})
    assert count_events(pg_env, project) == 1
