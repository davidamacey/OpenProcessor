"""Activations: compare-and-set, the event log, rollback and the axis/kind check."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from _config_fixtures import make_project

from src.db.repos import config as repo
from src.db.session import async_session_factory, deployment_tx, project_tx


pytestmark = pytest.mark.db


@pytest.fixture
def project(pg) -> uuid.UUID:
    return make_project(pg, f'act-{uuid.uuid4().hex[:12]}')


async def _doc(project_id, name, kind='region_profile', expected=None):
    async with async_session_factory()() as session, project_tx(session, project_id):
        return await repo.save_config(
            session,
            project_id=project_id,
            kind=kind,
            name=name,
            body={'n': name},
            expected_revision=expected,
        )


async def _activate(project_id, axis, doc, expected, *, revision=None):
    async with async_session_factory()() as session, project_tx(session, project_id):
        return await repo.activate(
            session,
            project_id=project_id,
            axis=axis,
            config_id=None if doc is None else doc.id,
            revision=None if doc is None else (revision or doc.revision),
            expected_active=expected,
        )


@pytest.mark.asyncio
async def test_a_stale_expected_active_is_a_conflict(pg_env, project) -> None:
    a = await _doc(project, 'prof-a')
    b = await _doc(project, 'prof-b')
    await _activate(project, 'detection_profile', a, None)
    with pytest.raises(repo.ActiveConflictError) as exc:
        await _activate(project, 'detection_profile', b, {'name': 'other', 'revision': 1})
    assert exc.value.current == {'name': 'prof-a', 'revision': 1}
    ref = await _activate(project, 'detection_profile', b, {'name': 'prof-a', 'revision': 1})
    assert ref.name == 'prof-b'


@pytest.mark.asyncio
async def test_two_first_activations_one_wins(pg_env, project) -> None:
    a = await _doc(project, 'first-a', kind='prompt_pack')
    b = await _doc(project, 'first-b', kind='prompt_pack')
    results = await asyncio.gather(
        _activate(project, 'prompt_pack', a, None),
        _activate(project, 'prompt_pack', b, None),
        return_exceptions=True,
    )
    assert [type(r).__name__ for r in sorted(results, key=lambda r: type(r).__name__)] == [
        'ActiveConflictError',
        'ActiveRef',
    ]


@pytest.mark.asyncio
async def test_a_region_profile_on_the_vlm_axis_is_refused_by_the_database(pg_env, project) -> None:
    doc = await _doc(project, 'wrong-axis')
    async with async_session_factory()() as session, deployment_tx(session, 'axis test'):
        with pytest.raises(repo.AxisKindError):
            await repo.activate(
                session,
                project_id=project,
                axis='vlm',
                config_id=doc.id,
                revision=1,
                expected_active=None,
            )


@pytest.mark.asyncio
async def test_rollback_restores_the_previous_revision_and_logs_it(pg_env, project) -> None:
    doc = await _doc(project, 'roll')
    v2 = await _doc(project, 'roll', expected=1)
    await _activate(project, 'detection_profile', doc, None, revision=1)
    await _activate(project, 'detection_profile', v2, {'name': 'roll', 'revision': 1})
    async with async_session_factory()() as session, project_tx(session, project):
        ref = await repo.rollback(
            session,
            project_id=project,
            axis='detection_profile',
            expected_active={'name': 'roll', 'revision': 2},
        )
        events = await repo.list_activation_events(session, project, 'detection_profile', limit=2)
    assert (ref.name, ref.revision) == ('roll', 1)
    assert events[0].reason == 'rollback'
    assert (events[0].from_revision, events[0].to_revision) == (2, 1)


@pytest.mark.asyncio
async def test_off_is_distinct_from_never_activated_and_rolls_back(pg_env, project) -> None:
    doc = await _doc(project, 'off-prof')
    async with async_session_factory()() as session, project_tx(session, project):
        assert await repo.get_activation(session, project, 'open_vocab') is None
    ov = await _doc(project, 'ov', kind='open_vocab_set')
    await _activate(project, 'open_vocab', ov, None)
    off = await _activate(project, 'open_vocab', None, {'name': 'ov', 'revision': 1})
    assert off.state == 'off'
    # nothing is active, so a caller with no expectation can activate again
    again = await _activate(project, 'open_vocab', ov, None)
    assert again.state == 'active'
    assert doc.name == 'off-prof'
    async with async_session_factory()() as session, project_tx(session, project):
        back = await repo.rollback(
            session,
            project_id=project,
            axis='open_vocab',
            expected_active={'name': 'ov', 'revision': 1},
        )
    assert back.state == 'off'


@pytest.mark.asyncio
async def test_rollback_without_an_earlier_state_is_refused(pg_env, project) -> None:
    ing = await _doc(project, 'ing', kind='ingest_profile')
    await _activate(project, 'ingest_profile', ing, None)
    async with async_session_factory()() as session, project_tx(session, project):
        with pytest.raises(LookupError, match='no_previous'):
            await repo.rollback(
                session,
                project_id=project,
                axis='ingest_profile',
                expected_active={'name': 'ing', 'revision': 1},
            )


@pytest.mark.asyncio
async def test_an_activation_bumps_the_epoch_and_pins_the_revision(pg_env, project) -> None:
    doc = await _doc(project, 'pinned', kind='prompt_pack')
    async with async_session_factory()() as session, project_tx(session, project):
        before = await repo.get_epoch(session, project)
    await _activate(project, 'prompt_pack', doc, None)
    async with async_session_factory()() as session, project_tx(session, project):
        after = await repo.get_epoch(session, project)
        ref = await repo.get_activation(session, project, 'prompt_pack')
    await _doc(project, 'pinned', kind='prompt_pack', expected=1)
    assert after == before + 1
    assert ref is not None
    assert (ref.name, ref.revision) == ('pinned', 1)


@pytest.mark.asyncio
async def test_load_activation_events_makes_the_last_event_the_activation(pg_env, pg) -> None:
    import datetime as dt

    project = make_project(pg, f'load-{uuid.uuid4().hex[:12]}')
    a = await _doc(project, 'la')
    b = await _doc(project, 'lb')
    t = dt.datetime(2026, 3, 1, tzinfo=dt.UTC)
    events = [
        repo.ActivationEventLoad('detection_profile', a.id, 1, t),
        repo.ActivationEventLoad(
            'detection_profile', b.id, 1, t, from_config_id=a.id, from_revision=1
        ),
    ]
    async with async_session_factory()() as session, deployment_tx(session, 'load test'):
        await repo.load_activation_events(session, project, events)
    async with async_session_factory()() as session, project_tx(session, project):
        ref = await repo.get_activation(session, project, 'detection_profile')
        logged = await repo.list_activation_events(session, project, 'detection_profile')
    assert ref is not None
    assert (ref.name, ref.activated_at) == ('lb', t)
    assert [e.reason for e in logged] == ['upgrade', 'upgrade']


@pytest.mark.asyncio
async def test_probes_and_runtime_reports_round_trip(pg_env, project) -> None:
    async with async_session_factory()() as session, deployment_tx(session, 'probe test'):
        ep = await repo.save_config(
            session,
            project_id=None,
            kind='vlm_endpoint',
            name='probed',
            body={'url': 'http://x'},
            expected_revision=None,
        )
        await repo.record_probe(session, config_id=ep.id, revision=1, ok=True, result={'ms': 5})
        await repo.record_probe(session, config_id=ep.id, revision=1, ok=False, result={'ms': 9})
        probes = await repo.list_probes(session)
    mine = [p for p in probes if p.config_id == ep.id]
    assert [(p.ok, p.result) for p in mine] == [(False, {'ms': 9})]
    async with async_session_factory()() as session, project_tx(session, project):
        await repo.upsert_runtime_report(
            session, project_id=project, process='worker', host='h1', config_epoch=4
        )
        await repo.upsert_runtime_report(
            session, project_id=project, process='worker', host='h1', config_epoch=6
        )
        reports = await repo.list_runtime_reports(
            session, project_id=project, process='worker', max_age_s=300
        )
    assert [(r.host, r.config_epoch) for r in reports] == [('h1', 6)]
