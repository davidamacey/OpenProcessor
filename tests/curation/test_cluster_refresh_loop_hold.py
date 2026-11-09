"""While an auto-label loop runs, nothing retrains the clusters under it: the periodic refresh
keeps its request and starts nothing, and the retraining start routes answer 409
``assist_loop_running``. (The tick's own hold is run on a database in
``tests/tasks/test_cluster_refresh_starts_cluster_run.py``.)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.test_auto_label_selection import client, job_dir, packs  # noqa: F401 - fixtures
from src.services.clustering.refresh_schedule import (
    Action,
    Observation,
    Policy,
    ProjectSchedule,
    step,
)
from src.services.curation.assist_loop import service


if TYPE_CHECKING:
    from fastapi.testclient import TestClient

# The real hold, taken at import: the offline tier's autouse fixture replaces it with a no-op
# in every non-db test, and a test of the hold itself puts it back.
REAL_HOLD = service.require_no_live_loop

POLICY = Policy(threshold=200, max_deferral_s=600.0, retry_backoff_s=300.0)
START = '/curation/projects/default/pipeline/auto_label/start'


@pytest.fixture
def http(request: pytest.FixtureRequest) -> TestClient:
    return request.getfixturevalue('client')  # type: ignore[no-any-return]


LOOP_ID = '01a1b198-3400-7fd7-a7c3-1acc88024d4a'


@pytest.fixture
def live_loop(monkeypatch: pytest.MonkeyPatch) -> str:
    """The project has a live loop (the lookup the hold reads, answered)."""

    from src.routers.curation import pipeline_public

    async def live(_session: Any) -> str:
        return LOOP_ID

    monkeypatch.setattr(service, '_live_loop_id', live)
    monkeypatch.setattr(service, 'require_no_live_loop', REAL_HOLD)
    monkeypatch.setattr(pipeline_public, 'require_no_live_loop', REAL_HOLD)
    return LOOP_ID


@pytest.fixture(autouse=True)
def _no_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    async def none(_session: Any) -> None:
        return None

    monkeypatch.setattr(service, '_live_loop_id', none)


# ---- the schedule ---------------------------------------------------------


def _obs(count: int, *, loop_live: bool = False) -> Observation:
    return Observation(
        count=count,
        built=1000,
        unfinished=0,
        tracked_status=None,
        busy=False,
        loop_live=loop_live,
        recent_ingest=False,
    )


def test_a_live_loop_holds_the_retrain_and_keeps_the_request() -> None:
    s = ProjectSchedule(prev_count=1250)

    assert step(s, _obs(1250, loop_live=True), 0.0, POLICY) is Action.NONE
    assert s.pending_since == 0.0, 'the request is kept, not dropped'
    assert step(s, _obs(1250, loop_live=True), 900.0, POLICY) is Action.NONE, (
        'even past the deferral bound'
    )
    assert step(s, _obs(1250), 960.0, POLICY) is Action.START


# ---- the start routes -----------------------------------------------------


@pytest.mark.usefixtures('packs', 'job_dir')
def test_a_retraining_start_is_refused_with_409_while_a_loop_is_live(
    http: TestClient, live_loop: str
) -> None:
    loop_id = live_loop

    refused = http.post(START)

    assert refused.status_code == 409, refused.text
    detail = refused.json()['detail']
    assert detail['error'] == 'assist_loop_running'
    assert detail['loop_id'] == loop_id


@pytest.mark.usefixtures('packs', 'job_dir')
@pytest.mark.usefixtures('live_loop')
def test_a_start_that_does_not_retrain_is_not_held(http: TestClient) -> None:

    allowed = http.post(START, params={'train_clusters': 'false'})

    assert allowed.status_code == 202, allowed.text


@pytest.mark.usefixtures('packs', 'job_dir')
@pytest.mark.usefixtures('live_loop')
def test_the_synchronous_chain_is_held_too(http: TestClient) -> None:

    refused = http.post('/curation/projects/default/pipeline/auto_label')

    assert refused.status_code == 409, refused.text
    assert refused.json()['detail']['error'] == 'assist_loop_running'


@pytest.mark.usefixtures('packs', 'job_dir')
def test_a_start_goes_through_once_the_loop_has_ended(http: TestClient) -> None:
    assert http.post(START).status_code == 202
