"""Status, runs, log tail, cancel, profiles/presets and artifact serving (routes over fake seams;
the database-backed behaviour is tested in ``tests/training``)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from src.services.training.job_models import TrainJobStatus


if TYPE_CHECKING:
    from pathlib import Path
    from types import SimpleNamespace

    from fastapi.testclient import TestClient


# =============================================================================
# /status
# =============================================================================


def test_status_endpoint_returns_null_when_empty(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/train/status')
    assert r.status_code == 200
    assert r.json() is None


def test_status_by_id_404(app_client: TestClient) -> None:
    r = app_client.get(f'/curation/projects/default/train/status/{uuid.uuid4()}')
    assert r.status_code == 404


def test_status_by_id_returns_status(app_client: TestClient, train_fake: SimpleNamespace) -> None:
    job_id = str(uuid.uuid4())
    train_fake.statuses[job_id] = TrainJobStatus(job_id=job_id, state='running')
    r = app_client.get(f'/curation/projects/default/train/status/{job_id}')
    assert r.status_code == 200
    body = r.json()
    assert body['job_id'] == job_id
    assert body['state'] == 'running'


def test_status_by_id_invalid_id(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/train/status/..%2Fescape')
    # Path-param percent decoding is up to FastAPI; either 400 or 404 is fine.
    assert r.status_code in (400, 404)


# =============================================================================
# /runs
# =============================================================================


def test_runs_list_empty(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/train/runs')
    assert r.status_code == 200
    body = r.json()
    assert body['items'] == []
    assert body['total'] == 0


def test_runs_list_pages_and_totals(app_client: TestClient, train_fake: SimpleNamespace) -> None:
    ids = [str(uuid.uuid4()) for _ in range(3)]
    for job_id in ids:
        train_fake.statuses[job_id] = TrainJobStatus(job_id=job_id, state='queued')
    body = app_client.get('/curation/projects/default/train/runs?limit=2&offset=1').json()
    assert [i['job_id'] for i in body['items']] == ids[1:3]
    assert body['total'] == 3


# =============================================================================
# /log/tail
# =============================================================================


def test_log_tail_returns_lines(app_client: TestClient, train_fake: SimpleNamespace) -> None:
    job_id = str(uuid.uuid4())
    train_fake.logs[job_id] = ['a', 'b', 'c', 'd']
    r = app_client.get(f'/curation/projects/default/train/log/tail/{job_id}?lines=2')
    assert r.status_code == 200
    body = r.json()
    assert body['job_id'] == job_id
    assert body['lines'] == ['c', 'd']


# =============================================================================
# /cancel
# =============================================================================


def test_cancel_cancels_the_run(app_client: TestClient, train_fake: SimpleNamespace) -> None:
    job_id = str(uuid.uuid4())
    r = app_client.post(f'/curation/projects/default/train/cancel/{job_id}')
    assert r.status_code == 200, r.text
    assert train_fake.cancelled == [job_id]
    assert r.json()['cancelled'] is True


def test_cancel_of_an_unknown_run_is_404(
    app_client: TestClient, train_fake: SimpleNamespace
) -> None:
    train_fake.cancel_found = False
    assert (
        app_client.post(f'/curation/projects/default/train/cancel/{uuid.uuid4()}').status_code
        == 404
    )


def test_cancel_campaign(app_client: TestClient, train_fake: SimpleNamespace) -> None:
    train_fake.campaign_cancelled = 2
    r = app_client.post('/curation/projects/default/train/cancel_campaign/camp')
    assert r.status_code == 200, r.text
    assert r.json() == {'cancelled': 2, 'campaign_id': 'camp'}


# =============================================================================
# /profiles, /presets
# =============================================================================


def test_profiles_endpoint(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/train/profiles')
    assert r.status_code == 200
    body = r.json()
    assert 'profiles' in body
    names = {p['name'] for p in body['profiles']}
    assert {'probe', 'nano', 'small', 'medium', 'large', 'xlarge'}.issubset(names)
    medium = next(p for p in body['profiles'] if p['name'] == 'medium')
    assert medium['defaults']['optimizer'] == 'MuSGD'


def test_presets_endpoint(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/train/presets')
    assert r.status_code == 200
    body = r.json()
    assert 'class_subset_presets' in body
    names = {p['name'] for p in body['class_subset_presets']}
    # Neutral by default: no region profile configured in this test app,
    # so only the generic 'all' preset is served.
    assert names == {'all'}


# =============================================================================
# /artifacts/{job_id}/{name}
# =============================================================================


def test_get_run_artifact_serves_whitelisted_file(
    app_client: TestClient, train_fake: SimpleNamespace, tmp_path: Path
) -> None:
    job_id = str(uuid.uuid4())
    artifact = tmp_path / 'confusion_matrix.png'
    artifact.write_bytes(b'\x89PNG-fake-bytes')
    train_fake.artifacts[(job_id, 'confusion_matrix.png')] = artifact

    r = app_client.get(f'/curation/projects/default/train/artifacts/{job_id}/confusion_matrix.png')

    assert r.status_code == 200, r.text
    assert r.content == b'\x89PNG-fake-bytes'
    assert r.headers['content-type'] == 'image/png'


def test_get_run_artifact_404_for_unknown_job_or_missing_file(app_client: TestClient) -> None:
    r = app_client.get(
        f'/curation/projects/default/train/artifacts/{uuid.uuid4()}/confusion_matrix.png'
    )
    assert r.status_code == 404


def test_get_run_artifact_400_for_invalid_job_id(app_client: TestClient) -> None:
    r = app_client.get(
        '/curation/projects/default/train/artifacts/..%2Fescape/confusion_matrix.png'
    )
    # Percent-decoding of the path param is up to FastAPI/Starlette; either
    # a 400 (our validation) or 404 (routing never matched) is acceptable,
    # matching the existing /status/{job_id} traversal test's tolerance.
    assert r.status_code in (400, 404)
    assert (
        app_client.get(
            '/curation/projects/default/train/artifacts/not-a-uuid/confusion_matrix.png'
        ).status_code
        == 400
    )
