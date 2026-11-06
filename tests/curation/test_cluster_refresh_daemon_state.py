"""Issues #235 and #280: the cluster-refresh daemon.

#235: the growth state (items at the last retrain, the job in flight, the previous
poll's count) lives in the project's ``project_state`` row, so a restart neither retrains
a project that was just retrained nor starts a retrain in the middle of an ingest.

#280: an audit-gate 409 from ``auto_promote`` means nothing was promoted; the retrain
is still queued."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from projects.test_project_counts import _record
from sqlalchemy.exc import OperationalError

from curation.query_fakes import SettingsFakeOpenSearch
from scripts.curation import cluster_refresh_daemon as d
from src.config.project_context import bind_project
from src.services.curation.clustering import refresh_state_store
from src.services.curation.clustering.refresh_schedule import (
    Action,
    Observation,
    Policy,
    ProjectSchedule,
    ScheduleBook,
    step,
)
from src.services.curation.clustering.refresh_state_store import RefreshState
from src.services.curation.policy_doc_store import PolicyConflictError


if TYPE_CHECKING:
    from collections.abc import Callable

POLICY = Policy(threshold=200, max_deferral_s=1800, retry_backoff_s=600)


class _Fake(SettingsFakeOpenSearch):
    """Item and unfinished-region counts the daemon polls, plus the settings document."""

    def __init__(self) -> None:
        super().__init__({})
        self.items = 0
        self.unfinished = 0
        self.recent_ingest = 0
        self.probes: list[Any] = []

    async def count(self, *, body: Any = None, **_: Any) -> dict[str, int]:
        return {'count': self.unfinished if body else self.items}

    async def search(self, *, body: Any = None, **_: Any) -> dict[str, Any]:
        """The "anything ingested lately" probe: it must ask for no documents and stop at
        the first match, since only zero versus nonzero matters."""
        self.probes.append(body)
        return {'hits': {'total': {'value': min(self.recent_ingest, 1)}, 'hits': []}}


class _Api:
    """The daemon's HTTP surface: records calls, answers per-route."""

    def __init__(self, promote_status: int = 200) -> None:
        self.calls: list[str] = []
        self.promote_status = promote_status

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f'{request.method} {path.rsplit("/", 2)[-2]}/{path.rsplit("/", 1)[-1]}')
        if path.endswith('/assist/loops'):
            return httpx.Response(200, json={'loops': []})
        if path.endswith('/clusters/auto_promote'):
            return httpx.Response(self.promote_status, json={'promoted': 0, 'detail': 'gate'})
        if path.endswith('/auto_label/start'):
            return httpx.Response(200, json={'job_id': 'job-1'})
        if '/auto_label/status/' in path:  # the job this daemon started
            return httpx.Response(200, json={'status': 'completed'})
        if path.endswith('/auto_label/status'):
            return httpx.Response(200, json={'status': 'idle'})
        return httpx.Response(404)

    @property
    def started(self) -> bool:
        return any(c.endswith('auto_label/start') for c in self.calls)


def _poll(
    api: _Api, fake: _Fake, sched: ProjectSchedule, now: float, *, record: Any = None
) -> None:
    async def go() -> None:
        async with api.client() as client:
            await d._iteration(
                client,
                api='http://api',
                api_prefix='/curation',
                record=record or _record('alpha'),
                opensearch=fake,
                sched=sched,
                policy=POLICY,
                auto_label=True,
                settle_s=240,
                now=now,
            )

    asyncio.run(go())


# ---- #280 -----------------------------------------------------------------


def test_an_audit_gate_409_still_queues_the_retrain() -> None:
    api, fake, sched = _Api(promote_status=409), _Fake(), ScheduleBook().get('alpha')
    fake.items = 1000

    _poll(api, fake, sched, 0.0)

    assert api.started
    assert sched.job_id == 'job-1'


def test_a_promote_server_error_still_stops_the_retrain() -> None:
    api, fake, sched = _Api(promote_status=500), _Fake(), ScheduleBook().get('alpha')
    fake.items = 1000

    _poll(api, fake, sched, 0.0)

    assert not api.started
    assert sched.job_id is None


# ---- #235 -----------------------------------------------------------------


def test_the_state_round_trips_through_project_state() -> None:
    async def go() -> RefreshState:
        saved = await refresh_state_store.save_refresh_state(
            RefreshState(last_trained_count=1200, job_id='j', prev_count=1300)
        )
        assert saved.revision == 1
        return await refresh_state_store.load_refresh_state()

    stored = asyncio.run(go())
    assert (stored.last_trained_count, stored.job_id, stored.prev_count) == (1200, 'j', 1300)


def test_saving_stale_state_conflicts() -> None:
    async def go() -> None:
        save = refresh_state_store.save_refresh_state
        first = await save(RefreshState(last_trained_count=1))
        await save(first.model_copy(update={'last_trained_count': 2}))
        await save(first.model_copy(update={'last_trained_count': 3}))

    with pytest.raises(PolicyConflictError):
        asyncio.run(go())


def test_a_restart_does_not_retrain_a_project_that_was_just_retrained() -> None:
    api, fake = _Api(), _Fake()
    fake.items = 1000
    _poll(api, fake, ScheduleBook().get('alpha'), 0.0)
    assert api.started
    api.calls.clear()

    fake.items = 1050  # +50 since the retrain: below the threshold
    restarted = ScheduleBook().get('alpha')  # the daemon process restarts
    _poll(api, fake, restarted, 10_000.0)

    assert not api.started
    assert restarted.last_trained_count == 1000


def test_a_restart_does_not_start_a_retrain_in_the_middle_of_an_ingest() -> None:
    api, fake = _Api(), _Fake()
    fake.items = 1000
    first = ScheduleBook().get('alpha')
    _poll(api, fake, first, 0.0)  # retrain at 1000
    api.calls.clear()
    fake.items = 1300  # ingest keeps adding items; the poll before the restart saw 1300
    first_job = first.job_id
    assert first_job == 'job-1'
    fake.unfinished = 0

    done = api
    _poll(done, fake, first, 100.0)  # records prev_count=1300, crossed but not quiet yet
    done.calls.clear()

    fake.items = 1600  # still growing when the daemon restarts
    restarted = ScheduleBook().get('alpha')
    _poll(done, fake, restarted, 200.0)

    assert not done.started  # count changed since the persisted previous poll: not quiet


def test_without_the_persisted_state_a_restart_would_start_early() -> None:
    """The schedule alone, to show what the persisted ``prev_count`` buys: a fresh
    schedule has no previous count and calls a growing ingest quiet."""
    fresh = ProjectSchedule(last_trained_count=1000)
    obs = Observation(
        count=1600,
        unfinished=0,
        tracked_status=None,
        busy=False,
        loop_live=False,
        recent_ingest=False,
    )
    assert step(fresh, obs, 0.0, POLICY) is Action.START
    restored = ProjectSchedule(last_trained_count=1000, prev_count=1300)
    assert step(restored, obs, 0.0, POLICY) is Action.DEFER


def test_a_failed_state_load_skips_the_project_for_this_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def down() -> RefreshState:
        raise RuntimeError('database is down')

    monkeypatch.setattr(d, 'load_refresh_state', down)
    api, fake = _Api(), _Fake()
    fake.items = 1000
    sched = ScheduleBook().get('alpha')

    _poll(api, fake, sched, 0.0)

    assert not api.started


def test_a_failed_save_is_retried_on_the_next_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, api = _Fake(), _Api()
    fake.items = 1000
    sched = ScheduleBook().get('alpha')
    real_save: Callable[..., Any] = d.save_refresh_state
    calls = {'n': 0}

    async def flaky(state: RefreshState) -> RefreshState:
        calls['n'] += 1
        if calls['n'] <= 2:
            raise OperationalError('save', {}, OSError('database is down'))
        return await real_save(state)

    monkeypatch.setattr(d, 'save_refresh_state', flaky)

    _poll(api, fake, sched, 0.0)  # retrain dispatched, the save fails
    _poll(api, fake, sched, 10.0)  # the next poll saves it

    with bind_project(_record('alpha'), read_only=True):
        stored = asyncio.run(refresh_state_store.load_refresh_state())
    assert stored.last_trained_count == 1000
    assert stored.trained_before is None  # the job completed by the second poll


# ---- #443 -----------------------------------------------------------------


def test_a_slow_ingest_batch_with_a_flat_count_does_not_start_the_retrain() -> None:
    """Items did not change between two polls and nothing is pending, yet images were
    ingested inside the settle window: ingest is still running."""
    api, fake, sched = _Api(), _Fake(), ScheduleBook().get('alpha')
    fake.items = 1000
    fake.recent_ingest = 12

    _poll(api, fake, sched, 0.0)
    _poll(api, fake, sched, 120.0)

    assert not api.started

    fake.recent_ingest = 0
    _poll(api, fake, sched, 240.0)

    assert api.started


def test_the_recent_ingest_probe_reads_no_documents_and_stops_at_one_match() -> None:
    api, fake, sched = _Api(), _Fake(), ScheduleBook().get('alpha')
    fake.items = 1000
    fake.recent_ingest = 5000

    _poll(api, fake, sched, 0.0)

    (probe,) = fake.probes
    assert probe['size'] == 0
    assert probe['terminate_after'] == 1
    assert probe['query']['range']['indexed_at']['gte'].startswith('now-')
