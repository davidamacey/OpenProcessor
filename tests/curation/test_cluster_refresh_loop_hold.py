"""While an auto-label loop runs, nothing retrains the clusters under it: the refresh daemon
neither retrains nor auto-promotes, the auto-label worker's idle retrain stays off, and the
retraining start routes answer 409 ``assist_loop_running``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from projects.test_project_counts import _record

from curation.query_fakes import SettingsFakeOpenSearch
from curation.test_auto_label_selection import client, job_dir, packs  # noqa: F401 - fixtures
from scripts.curation import cluster_refresh_daemon as d
from src.services.curation.assist_loop import job
from src.services.curation.clustering.refresh_schedule import (
    Action,
    Observation,
    Policy,
    ProjectSchedule,
    ScheduleBook,
    step,
)


if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

POLICY = Policy(threshold=200, max_deferral_s=600.0, retry_backoff_s=300.0)
START = '/curation/projects/default/pipeline/auto_label/start'


@pytest.fixture
def http(request: pytest.FixtureRequest) -> TestClient:
    return request.getfixturevalue('client')  # type: ignore[no-any-return]


@pytest.fixture(autouse=True)
def _loops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_ASSIST_LOOP_JOBS_DIR', str(tmp_path / 'loops'))


def _live_loop() -> str:
    return job.create_loop(state={'status': 'queued'}, request={}).directory.name


# ---- the schedule ---------------------------------------------------------


def _obs(count: int, *, loop_live: bool = False) -> Observation:
    return Observation(
        count=count, unfinished=0, tracked_status=None, busy=False, loop_live=loop_live
    )


def test_a_live_loop_holds_the_retrain_and_keeps_the_request() -> None:
    s = ProjectSchedule(last_trained_count=1000, prev_count=1250)

    assert step(s, _obs(1250, loop_live=True), 0.0, POLICY) is Action.NONE
    assert s.pending_since == 0.0, 'the request is kept, not dropped'
    assert step(s, _obs(1250, loop_live=True), 900.0, POLICY) is Action.NONE, (
        'even past the deferral bound'
    )
    assert step(s, _obs(1250), 960.0, POLICY) is Action.START


# ---- the daemon -----------------------------------------------------------


class _Fake(SettingsFakeOpenSearch):
    def __init__(self) -> None:
        super().__init__({})

    async def count(self, *, body: Any = None, **_: Any) -> dict[str, int]:
        return {'count': 0 if body else 1000}


class _Api:
    def __init__(self, *, loops: Any) -> None:
        self.calls: list[str] = []
        self.loops = loops

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f'{request.method} {path.removeprefix("/curation/projects/alpha")}')
        if path.endswith('/assist/loops'):
            assert request.url.params['active'] == 'true'
            if isinstance(self.loops, int):
                return httpx.Response(self.loops)
            return httpx.Response(200, json={'loops': self.loops})
        if path.endswith('/auto_label/status'):
            return httpx.Response(200, json={'status': 'idle'})
        if path.endswith('/auto_label/start'):
            return httpx.Response(200, json={'job_id': 'job-1'})
        return httpx.Response(200, json={'promoted': 0})

    def poll(self, sched: ProjectSchedule, *, auto_label: bool = True) -> None:
        async def go() -> None:
            transport = httpx.MockTransport(self._handle)
            async with httpx.AsyncClient(transport=transport) as http:
                await d._iteration(
                    http,
                    api='http://api',
                    api_prefix='/curation',
                    record=_record('alpha'),
                    opensearch=_Fake(),
                    sched=sched,
                    policy=POLICY,
                    auto_label=auto_label,
                    now=0.0,
                )

        asyncio.run(go())

    @property
    def retrained(self) -> bool:
        return any(c.endswith(('auto_label/start', 'auto_promote')) for c in self.calls)


@pytest.mark.parametrize('auto_label', [True, False])
def test_the_daemon_neither_promotes_nor_retrains_while_a_loop_is_live(auto_label: bool) -> None:
    api, sched = (
        _Api(loops=[{'loop_id': 'alp_1', 'status': 'running'}]),
        ScheduleBook().get('alpha'),
    )

    api.poll(sched, auto_label=auto_label)

    assert not api.retrained
    assert sched.job_id is None
    assert sched.pending_since == 0.0


def test_the_daemon_retrains_when_no_loop_is_live() -> None:
    api, sched = _Api(loops=[]), ScheduleBook().get('alpha')

    api.poll(sched)

    assert any(c.endswith('auto_label/start') for c in api.calls)


def test_the_daemon_fails_closed_when_it_cannot_tell_whether_a_loop_is_live() -> None:
    api, sched = _Api(loops=500), ScheduleBook().get('alpha')

    api.poll(sched)

    assert not api.retrained


# ---- the auto-label worker's idle retrain --------------------------------


def test_the_workers_idle_retrain_stays_off_while_a_loop_is_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.curation import auto_label_worker as w
    from src.services.curation.clustering import retrain_policy

    async def _due(_client: Any) -> dict[str, Any]:
        return {'should': True}

    monkeypatch.setattr(retrain_policy, 'should_retrain_centroids', _due)
    fake = SettingsFakeOpenSearch({})
    assert asyncio.run(w._maybe_auto_retrain(fake)) is not None
    _live_loop()

    assert asyncio.run(w._maybe_auto_retrain(fake)) is None


# ---- the start routes -----------------------------------------------------


@pytest.mark.usefixtures('packs', 'job_dir')
def test_a_retraining_start_is_refused_with_409_while_a_loop_is_live(http: TestClient) -> None:
    loop_id = _live_loop()

    refused = http.post(START)

    assert refused.status_code == 409, refused.text
    detail = refused.json()['detail']
    assert detail['error'] == 'assist_loop_running'
    assert detail['loop_id'] == loop_id


@pytest.mark.usefixtures('packs', 'job_dir')
def test_a_start_that_does_not_retrain_is_not_held(http: TestClient) -> None:
    _live_loop()

    allowed = http.post(START, params={'train_clusters': 'false'})

    assert allowed.status_code == 200, allowed.text


@pytest.mark.usefixtures('packs', 'job_dir')
def test_the_synchronous_chain_is_held_too(http: TestClient) -> None:
    _live_loop()

    refused = http.post('/curation/projects/default/pipeline/auto_label')

    assert refused.status_code == 409, refused.text
    assert refused.json()['detail']['error'] == 'assist_loop_running'


@pytest.mark.usefixtures('packs', 'job_dir')
def test_a_start_goes_through_once_the_loop_has_ended(http: TestClient) -> None:
    loop = job.create_loop(state={'status': 'queued'}, request={})
    loop.update(status='completed')

    assert http.post(START).status_code == 200
