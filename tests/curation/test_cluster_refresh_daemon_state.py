"""Issues #235 and #280: the cluster-refresh daemon.

#235: the growth state (items at the last retrain, the job in flight, the previous
poll's count) lives in the project's settings document, so a restart neither retrains
a project that was just retrained nor starts a retrain in the middle of an ingest.

#280: an audit-gate 409 from ``auto_promote`` means nothing was promoted; the retrain
is still queued."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from projects.test_project_counts import _record

from curation.query_fakes import SettingsFakeOpenSearch
from scripts.curation import cluster_refresh_daemon as d
from src.config.project_context import bind_project
from src.services.curation.clustering.refresh_schedule import (
    Action,
    Observation,
    Policy,
    ProjectSchedule,
    ScheduleBook,
    step,
)
from src.services.curation.clustering.refresh_state_store import (
    RefreshState,
    load_refresh_state,
    save_refresh_state,
)
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

    async def count(self, *, body: Any = None, **_: Any) -> dict[str, int]:
        return {'count': self.unfinished if body else self.items}


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


def test_the_state_round_trips_through_the_settings_document() -> None:
    fake = _Fake()

    async def go() -> RefreshState:
        saved = await save_refresh_state(
            fake, RefreshState(last_trained_count=1200, job_id='j', prev_count=1300)
        )
        assert saved.revision == 1
        return await load_refresh_state(fake)

    stored = asyncio.run(go())
    assert (stored.last_trained_count, stored.job_id, stored.prev_count) == (1200, 'j', 1300)


def test_saving_stale_state_conflicts() -> None:
    fake = _Fake()

    async def go() -> None:
        first = await save_refresh_state(fake, RefreshState(last_trained_count=1))
        await save_refresh_state(fake, first.model_copy(update={'last_trained_count': 2}))
        await save_refresh_state(fake, first.model_copy(update={'last_trained_count': 3}))

    with pytest.raises(PolicyConflictError):
        asyncio.run(go())


def test_an_unreadable_settings_document_fails_the_load() -> None:
    class _Down(_Fake):
        async def get(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError('opensearch is down')

    with pytest.raises(RuntimeError, match='down'):
        asyncio.run(load_refresh_state(_Down()))


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
    obs = Observation(count=1600, unfinished=0, tracked_status=None, busy=False)
    assert step(fresh, obs, 0.0, POLICY) is Action.START
    restored = ProjectSchedule(last_trained_count=1000, prev_count=1300)
    assert step(restored, obs, 0.0, POLICY) is Action.DEFER


def test_a_failed_state_load_skips_the_project_for_this_poll() -> None:
    class _Down(_Fake):
        async def get(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError('opensearch is down')

    api, fake = _Api(), _Down()
    fake.items = 1000
    sched = ScheduleBook().get('alpha')

    _poll(api, fake, sched, 0.0)

    assert not api.started


def test_a_failed_save_is_retried_on_the_next_poll() -> None:
    fake, api = _Fake(), _Api()
    fake.items = 1000
    sched = ScheduleBook().get('alpha')
    real_index: Callable[..., Any] = fake.index
    calls = {'n': 0}

    async def flaky(**kw: Any) -> None:
        calls['n'] += 1
        if calls['n'] <= 2:
            raise OpenSearchConnectionError('N/A', 'opensearch is down', OSError('down'))
        await real_index(**kw)

    fake.index = flaky  # type: ignore[method-assign]

    _poll(api, fake, sched, 0.0)  # retrain dispatched, the save fails
    _poll(api, fake, sched, 10.0)  # the next poll saves it

    with bind_project(_record('alpha'), read_only=True):
        stored = asyncio.run(load_refresh_state(fake))
    assert stored.last_trained_count == 1000
    assert stored.trained_before is None  # the job completed by the second poll
