"""``/projects/combine*`` over HTTP: preview, start, poll, cancel/resume
errors, and the target's status transitions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import projects_combine
from src.routers.curation.projects import global_router
from src.services.projects import lifecycle

from .test_combine_execute import MAPPING, build


if TYPE_CHECKING:
    from .world import World


def _body(world: World, **extra: Any) -> dict[str, Any]:
    return {
        'target': {'slug': 'combined', 'display_name': 'Combined'},
        'sources': [{'project': 'cars-a'}, {'project': 'cars-b'}],
        'class_mapping': MAPPING,
        **extra,
    }


@pytest.fixture
def api(world: World, monkeypatch: pytest.MonkeyPatch):
    build(world)
    calls: dict[str, list[Any]] = {'create': [], 'finish': []}

    async def make_client() -> Any:
        return world.fake

    async def create_project(client: Any, **kwargs: Any):
        calls['create'].append(kwargs)
        record = world.project(kwargs['slug'], [], status='building')
        return record, []

    async def finish_building(client: Any, record: Any, *, ok: bool = True):
        calls['finish'].append((record.slug, ok))
        return record

    monkeypatch.setattr(projects_combine, 'make_curation_opensearch', make_client)
    monkeypatch.setattr(lifecycle, 'create_project', create_project)
    monkeypatch.setattr(lifecycle, 'finish_building', finish_building)
    app = FastAPI()
    app.include_router(global_router, prefix='/curation')
    with TestClient(app) as client:
        client.calls = calls  # type: ignore[attr-defined]
        yield client


def _wait(client: TestClient, job_id: str, until: set[str]) -> dict[str, Any]:
    deadline = time.time() + 20
    while time.time() < deadline:
        job = client.get(f'/curation/projects/combine/{job_id}').json()
        if job['status'] in until:
            return job
        time.sleep(0.05)
    raise AssertionError(job)


def test_preview_then_start_runs_the_job_and_activates_the_target(
    api: TestClient, world: World
) -> None:
    preview = api.post('/curation/projects/combine/preview', json=_body(world))
    assert preview.status_code == 200
    body = preview.json()
    assert body['ok'] is True
    assert body['dedup']['identical_images'] == 1

    started = api.post(
        '/curation/projects/combine',
        json=_body(world, expected_preview_sha=body['preview_sha']),
    )
    assert started.status_code == 202
    job_id = started.json()['job_id']
    assert started.json()['target'] == 'combined'
    job = _wait(api, job_id, {'completed', 'failed'})
    assert job['status'] == 'completed', job
    assert job['done'] == job['total'] == 6  # every source image, the duplicate included
    assert job['report']['images_copied'] == 5
    assert api.calls['create'][0]['origin']['kind'] == 'combine'  # type: ignore[attr-defined]
    assert api.calls['create'][0]['activate'] is False  # type: ignore[attr-defined]
    assert api.calls['finish'] == [('combined', True)]  # type: ignore[attr-defined]
    assert len(world.images('combined')) == 5


def test_a_stale_preview_sha_is_409(api: TestClient, world: World) -> None:
    r = api.post('/curation/projects/combine', json=_body(world, expected_preview_sha='deadbeef'))
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'preview_stale'
    assert api.calls['create'] == []  # type: ignore[attr-defined]


def test_an_invalid_request_is_422_with_a_report(api: TestClient, world: World) -> None:
    body = _body(world, expected_preview_sha='x')
    body['class_mapping'] = {}
    r = api.post('/curation/projects/combine', json=body)
    assert r.status_code == 422
    detail = r.json()['detail']
    assert detail['error'] == 'combine_invalid'
    assert {e['code'] for e in detail['report']['errors']} == {'unmapped_class'}


def test_a_to_or_drop_mapping_body_is_422(api: TestClient, world: World) -> None:
    body = _body(world)
    body['class_mapping'] = {'cars-a': [{'dataset_class': 'car', 'to': 'vehicle'}]}
    assert api.post('/curation/projects/combine/preview', json=body).status_code == 422
    body['class_mapping'] = {'cars-a': [{'dataset_class': 'car', 'drop': True}]}
    assert api.post('/curation/projects/combine/preview', json=body).status_code == 422


def test_unknown_jobs_and_wrong_state_actions(api: TestClient, world: World) -> None:
    assert api.get('/curation/projects/combine/cmb_20261001T000000_00000000').status_code == 404
    assert api.get('/curation/projects/combine/not-a-job-id').status_code == 404
    assert api.post('/curation/projects/combine/not-a-job-id/cancel').status_code == 404
    started = api.post(
        '/curation/projects/combine',
        json=_body(
            world,
            expected_preview_sha=api.post(
                '/curation/projects/combine/preview', json=_body(world)
            ).json()['preview_sha'],
        ),
    )
    job_id = started.json()['job_id']
    _wait(api, job_id, {'completed'})
    cancel = api.post(f'/curation/projects/combine/{job_id}/cancel')
    assert cancel.status_code == 409
    assert cancel.json()['detail']['error'] == 'combine_not_resumable'
    resume = api.post(f'/curation/projects/combine/{job_id}/resume')
    assert resume.status_code == 409


def test_combine_is_not_a_project_slug_route(api: TestClient) -> None:
    # `combine` is reserved, so these never fall through to /projects/{project}.
    assert api.get('/curation/projects/combine/preview').status_code in (404, 405)
