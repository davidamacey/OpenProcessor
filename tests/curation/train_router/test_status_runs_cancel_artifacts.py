"""Status, runs, log tail, cancel, profiles/presets and artifact serving."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from fastapi.testclient import TestClient


# =============================================================================
# /status
# =============================================================================


def test_status_endpoint_returns_null_when_empty(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/train/status')
    assert r.status_code == 200
    assert r.json() is None


def test_status_by_id_404(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/train/status/nope')
    assert r.status_code == 404


def test_status_by_id_returns_status(app_client: TestClient, tmp_path: Any) -> None:
    import json
    from datetime import UTC, datetime

    (tmp_path / 'projects' / 'default' / 'real.status.json').write_text(
        json.dumps(
            {
                'job_id': 'real',
                'state': 'running',
                'heartbeat_at': datetime.now(UTC).isoformat(),
            }
        )
    )
    r = app_client.get('/curation/projects/default/train/status/real')
    assert r.status_code == 200
    body = r.json()
    assert body['job_id'] == 'real'
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


# =============================================================================
# /log/tail
# =============================================================================


def test_log_tail_returns_lines(app_client: TestClient, tmp_path: Any) -> None:
    (tmp_path / 'projects' / 'default' / 'logj.run.log').write_text('a\nb\nc\nd\n')
    r = app_client.get('/curation/projects/default/train/log/tail/logj?lines=2')
    assert r.status_code == 200
    body = r.json()
    assert body['job_id'] == 'logj'
    assert body['lines'] == ['c', 'd']


# =============================================================================
# /cancel
# =============================================================================


def test_cancel_writes_sentinel(app_client: TestClient, tmp_path: Any) -> None:
    r = app_client.post('/curation/projects/default/train/cancel/some_job')
    assert r.status_code == 200, r.text
    assert (tmp_path / 'projects' / 'default' / 'some_job.cancel').exists()
    assert r.json()['cancelled'] is True


def test_cancel_campaign(app_client: TestClient, tmp_path: Any) -> None:
    import json
    from datetime import UTC, datetime

    # Two jobs, both belonging to one campaign, one running, one queued.
    for jid in ('camp_run00', 'camp_run01'):
        (tmp_path / 'projects' / 'default' / f'{jid}.job.json').write_text(
            json.dumps({'job_id': jid, 'campaign_id': 'camp', 'dataset_export_dir': '/x'})
        )
        (tmp_path / 'projects' / 'default' / f'{jid}.status.json').write_text(
            json.dumps(
                {
                    'job_id': jid,
                    'campaign_id': 'camp',
                    'state': 'running',
                    'heartbeat_at': datetime.now(UTC).isoformat(),
                }
            )
        )
    r = app_client.post('/curation/projects/default/train/cancel_campaign/camp')
    assert r.status_code == 200, r.text
    assert r.json()['cancelled'] == 2
    assert (tmp_path / 'projects' / 'default' / 'camp_run00.cancel').exists()
    assert (tmp_path / 'projects' / 'default' / 'camp_run01.cancel').exists()


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


def test_get_run_artifact_serves_whitelisted_file(app_client: TestClient, tmp_path: Any) -> None:
    run_dir = tmp_path / 'runs' / 'artjob'
    run_dir.mkdir(parents=True)
    (run_dir / 'confusion_matrix.png').write_bytes(b'\x89PNG-fake-bytes')
    (tmp_path / 'projects' / 'default' / 'artjob.status.json').write_text(
        json.dumps(
            {
                'job_id': 'artjob',
                'state': 'finished',
                'eval': {'confusion_matrix_path': str(run_dir / 'confusion_matrix.png')},
            }
        )
    )

    r = app_client.get('/curation/projects/default/train/artifacts/artjob/confusion_matrix.png')

    assert r.status_code == 200, r.text
    assert r.content == b'\x89PNG-fake-bytes'
    assert r.headers['content-type'] == 'image/png'


def test_get_run_artifact_404_for_non_whitelisted_name(
    app_client: TestClient, tmp_path: Any
) -> None:
    run_dir = tmp_path / 'runs' / 'artjob2'
    run_dir.mkdir(parents=True)
    (run_dir / 'best.pt').write_bytes(b'weights')
    (tmp_path / 'projects' / 'default' / 'artjob2.status.json').write_text(
        json.dumps(
            {'job_id': 'artjob2', 'state': 'finished', 'checkpoint_path': str(run_dir / 'best.pt')}
        )
    )

    r = app_client.get('/curation/projects/default/train/artifacts/artjob2/best.pt')

    assert r.status_code == 404


def test_get_run_artifact_404_for_unknown_job(app_client: TestClient) -> None:
    r = app_client.get(
        '/curation/projects/default/train/artifacts/does_not_exist/confusion_matrix.png'
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


def test_get_run_artifact_404_when_file_never_written(
    app_client: TestClient, tmp_path: Any
) -> None:
    run_dir = tmp_path / 'runs' / 'artjob3'
    run_dir.mkdir(parents=True)
    (tmp_path / 'projects' / 'default' / 'artjob3.status.json').write_text(
        json.dumps(
            {
                'job_id': 'artjob3',
                'state': 'running',
                'eval': {'confusion_matrix_path': str(run_dir / 'confusion_matrix.png')},
            }
        )
    )

    r = app_client.get('/curation/projects/default/train/artifacts/artjob3/confusion_matrix.png')

    assert r.status_code == 404


def test_status_eval_never_carries_a_filesystem_path(app_client: TestClient, tmp_path: Any) -> None:
    """No consumer of the wire should ever see a server path -- only the
    artifact-route URL (or null)."""
    run_dir = tmp_path / 'runs' / 'pathcheck'
    run_dir.mkdir(parents=True)
    (tmp_path / 'projects' / 'default' / 'pathcheck.status.json').write_text(
        json.dumps(
            {
                'job_id': 'pathcheck',
                'state': 'finished',
                'eval': {
                    'map50': 0.5,
                    'split': 'val',
                    'confusion_matrix_path': str(run_dir / 'confusion_matrix.png'),
                },
            }
        )
    )

    r = app_client.get('/curation/projects/default/train/status/pathcheck')

    assert r.status_code == 200, r.text
    body = r.json()
    assert 'confusion_matrix_path' not in body['eval']
    assert str(tmp_path) not in json.dumps(body)
    assert (
        body['eval']['confusion_matrix_url']
        == '/curation/projects/default/train/artifacts/pathcheck/confusion_matrix.png'
    )
