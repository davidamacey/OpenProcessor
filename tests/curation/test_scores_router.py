"""Tests for POST /curation/scores/compute + status/cancel/coverage.

The job itself (queue, busy, run) is covered by ``tests/curation/jobs_a``; these tests drive the
routes with the job seam (``scores_job.start``, ``latest_job``, ``cancel_latest``) replaced.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.core.db_session import get_db
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setenv('OP_SCORES_ENABLED', '1')
    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    monkeypatch.setattr('src.routers.curation.scores.current_project_id', lambda: 'project-id')

    fake_os = AsyncMock()
    fake_os.count = AsyncMock(return_value={'count': 0})

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os

    async def no_database():
        yield None

    app.dependency_overrides[get_db] = no_database

    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def _job_seam(monkeypatch: pytest.MonkeyPatch, latest) -> AsyncMock:
    start = AsyncMock()
    monkeypatch.setattr('src.services.curation.item_scores.job.start', start)
    monkeypatch.setattr(
        'src.services.curation.latest_job.latest_job', AsyncMock(return_value=latest)
    )
    return start


def _queued_view() -> SimpleNamespace:
    return SimpleNamespace(
        id='j1',
        status='queued',
        started_at=None,
        finished_at=None,
        error_message=None,
        params={'scorers': ['uniqueness']},
        progress={},
        result=None,
    )


def test_compute_queues_the_job_and_answers_202_with_its_state(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = _job_seam(monkeypatch, _queued_view())
    r = app_client.post(
        '/curation/projects/default/scores/compute', json={'scorers': ['uniqueness']}
    )
    assert r.status_code == 202, r.text
    assert (r.json()['job_id'], r.json()['status']) == ('j1', 'queued')
    assert start.await_args.args[1:] == ('project-id', ['uniqueness'])


def test_status_reflects_the_newest_job(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _job_seam(monkeypatch, _queued_view())
    r = app_client.get('/curation/projects/default/scores/status')
    assert r.json()['scorers'] == ['uniqueness']
    assert r.json()['status'] == 'queued'


def test_status_idle_with_no_job(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _job_seam(monkeypatch, None)
    r = app_client.get('/curation/projects/default/scores/status')
    assert r.status_code == 200
    assert r.json()['status'] == 'idle'


def test_cancel_reports_whether_a_job_was_cancelled(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _job_seam(monkeypatch, _queued_view())
    cancel = AsyncMock(return_value=True)
    monkeypatch.setattr('src.services.curation.latest_job.cancel_latest', cancel)
    r = app_client.post('/curation/projects/default/scores/cancel')
    assert r.json()['cancelled'] is True
    cancel.assert_awaited_once()


def test_unknown_scorer_400(app_client: TestClient) -> None:
    r = app_client.post(
        '/curation/projects/default/scores/compute', json={'scorers': ['not_a_real_scorer']}
    )
    assert r.status_code == 400
    assert 'not_a_real_scorer' in r.json()['detail']


def test_disabled_when_flag_off_400(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_SCORES_ENABLED', raising=False)
    r = app_client.post(
        '/curation/projects/default/scores/compute', json={'scorers': ['uniqueness']}
    )
    assert r.status_code == 400
    assert 'disabled' in r.json()['detail']


def test_coverage_reports_per_field_counts(app_client: TestClient) -> None:
    async def _count(index, body):
        query = body['query']
        if 'exists' not in query:
            return {'count': 100}  # total-crops count (match_all)
        field = query['exists']['field']
        return {'count': 7} if field == 'uniqueness_score' else {'count': 3}

    app_client.fake_os.count = AsyncMock(side_effect=_count)  # type: ignore[attr-defined]

    r = app_client.get('/curation/projects/default/scores/coverage')
    assert r.status_code == 200
    coverage = r.json()['coverage']
    assert set(coverage) == {'uniqueness', 'mistakenness'}
    assert coverage['uniqueness']['n_scored'] == 7
    assert coverage['uniqueness']['field'] == 'uniqueness_score'
    assert coverage['uniqueness']['total'] == 100
    assert coverage['mistakenness']['n_scored'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
