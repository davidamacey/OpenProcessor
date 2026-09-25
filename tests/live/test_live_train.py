"""Live cohort-(d) scenarios: the training control surface.

The API's entire protocol with a trainer is a shared-volume file protocol
(``src/services/training/jobs.py``). These tests drive it against the fake
trainer in ``docker/test/fake_trainer.sh`` and assert on the actual files
that appear in the shared jobs directory.

Two safety properties are asserted explicitly here, because they are the
reason this harness can run on a machine full of other people's
containers:

* ``POST /train/preflight`` must be side-effect free — no job file at all.
* ``POST /train/start`` must not stop or start any container. The GPU
  arbiter is a no-op in this repo and the harness never mounts the docker
  socket; a ``docker ps`` snapshot taken either side of the call proves it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from .conftest import JOBS_DIR, docker_ps_snapshot, wait_for_state, wait_until


pytestmark = pytest.mark.live

# Container-side path of the dataset the API hands the trainer.
CONTAINER_EXPORT_CURRENT = '/verify-data/exports/current'


def _job_files(job_id: str) -> dict[str, Any]:
    return {
        'job': JOBS_DIR / f'{job_id}.job.json',
        'status': JOBS_DIR / f'{job_id}.status.json',
        'cancel': JOBS_DIR / f'{job_id}.cancel',
        'log': JOBS_DIR / f'{job_id}.run.log',
        'manifest': JOBS_DIR / f'{job_id}.manifest.json',
    }


@pytest.fixture(scope='module')
def run_state() -> dict[str, Any]:
    """Carries the submitted job id between the ordered scenarios below."""
    return {}


def _spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        'dataset_export_dir': CONTAINER_EXPORT_CURRENT,
        'model_size': 'n',
        'profile': 'probe',
        'cuda_visible_devices': '0',
    }
    spec.update(overrides)
    return spec


def test_preflight_reports_checks_and_writes_no_job_file(api_client: Any) -> None:
    before = sorted(p.name for p in JOBS_DIR.glob('*.job.json'))

    resp = api_client.post('/train/preflight', json=_spec())
    assert resp.status_code == 200, resp.text
    report = resp.json()
    assert isinstance(report['blocked'], bool)
    assert report['checks'], report
    assert {c['severity'] for c in report['checks']} <= {'ok', 'warn', 'block', 'unknown'}

    after = sorted(p.name for p in JOBS_DIR.glob('*.job.json'))
    assert after == before, 'preflight must be side-effect free'


def test_start_writes_a_job_file_and_stops_no_container(
    api_client: Any, run_state: dict[str, Any]
) -> None:
    containers_before = docker_ps_snapshot()

    resp = api_client.post('/train/start', params={'force': True}, json=_spec())
    assert resp.status_code == 201, resp.text
    job_id = resp.json()['job_id']
    assert resp.json()['preflight']['checks']

    containers_after = docker_ps_snapshot()
    assert containers_after == containers_before, (
        'POST /train/start changed the set of running containers — the GPU '
        'arbiter must be a no-op and the docker socket is not mounted'
    )

    paths = _job_files(job_id)
    job_file = wait_until(lambda: paths['job'].is_file() and paths['job'], timeout=30)
    assert job_file, f'no {job_id}.job.json appeared in {JOBS_DIR}'
    payload = json.loads(paths['job'].read_text())
    assert payload['job_id'] == job_id
    assert payload['dataset_export_dir'] == CONTAINER_EXPORT_CURRENT
    assert payload['model_size'] == 'n'
    # Lineage the API pins at submit time so a later class rename cannot
    # silently relabel a promoted model.
    assert payload['registry_sha']
    assert payload['registry_snapshot_path']

    run_state['job_id'] = job_id


def test_fake_trainer_drives_the_run_to_finished(
    api_client: Any, run_state: dict[str, Any]
) -> None:
    job_id = run_state.get('job_id')
    assert job_id, 'the start scenario must run first'
    paths = _job_files(job_id)

    state = wait_for_state(
        lambda: api_client.get(f'/train/status/{job_id}').json(),
        lambda s: bool(s) and s.get('state') == 'finished',
        timeout=120,
    )
    assert state['state'] == 'finished', state
    assert state['job_id'] == job_id
    assert state['total_epochs'] >= 1
    assert state['eval']['map50'] == pytest.approx(0.5)
    # best_metric is back-filled from eval when the trainer omits it.
    assert state['best_metric']['map50'] == pytest.approx(0.5)

    assert paths['status'].is_file()
    assert paths['manifest'].is_file()

    latest = api_client.get('/train/status')
    assert latest.status_code == 200, latest.text
    assert latest.json()['job_id'] == job_id

    runs = api_client.get('/train/runs', params={'limit': 10})
    assert runs.status_code == 200, runs.text
    assert job_id in {item['job_id'] for item in runs.json()['items']}

    log = api_client.get(f'/train/log/tail/{job_id}', params={'lines': 20})
    assert log.status_code == 200, log.text
    assert any(job_id in line for line in log.json()['lines']), log.json()


def test_manifest_is_served_for_a_finished_run(api_client: Any, run_state: dict[str, Any]) -> None:
    job_id = run_state.get('job_id')
    assert job_id
    resp = api_client.get(f'/train/manifest/{job_id}')
    assert resp.status_code == 200, resp.text
    assert resp.json()['job_id'] == job_id


def test_cancel_drops_the_sentinel_and_the_trainer_honours_it(api_client: Any) -> None:
    started = api_client.post('/train/start', params={'force': True}, json=_spec(model_size='s'))
    assert started.status_code == 201, started.text
    job_id = started.json()['job_id']
    paths = _job_files(job_id)

    resp = api_client.post(f'/train/cancel/{job_id}')
    assert resp.status_code == 200, resp.text
    assert resp.json() == {'cancelled': True, 'job_id': job_id}
    assert paths['cancel'].is_file(), 'the cancel sentinel was not written'

    state = wait_for_state(
        lambda: api_client.get(f'/train/status/{job_id}').json(),
        lambda s: bool(s) and s.get('state') == 'cancelled',
        timeout=60,
    )
    assert state['state'] == 'cancelled', state


def test_unknown_job_id_is_a_404(api_client: Any) -> None:
    resp = api_client.get('/train/status/2026-01-01T00-00-00_nosuchrun')
    assert resp.status_code == 404, resp.text


def test_campaign_submit_then_cancel_campaign(api_client: Any) -> None:
    campaign_spec = {
        'dataset_export_dir': CONTAINER_EXPORT_CURRENT,
        'cuda_visible_devices': '0',
        'runs': [
            {'profile': 'probe', 'model_size': 'n'},
            {'profile': 'probe', 'model_size': 's'},
        ],
    }
    resp = api_client.post('/train/start_campaign', params={'force': True}, json=campaign_spec)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    campaign_id = body['campaign_id']
    assert len(body['job_ids']) == 2

    for job_id in body['job_ids']:
        assert _job_files(job_id)['job'].is_file(), job_id
        payload = json.loads(_job_files(job_id)['job'].read_text())
        assert payload['campaign_id'] == campaign_id

    resp = api_client.post(f'/train/cancel_campaign/{campaign_id}')
    assert resp.status_code == 200, resp.text
    assert resp.json()['cancelled'] >= 1, resp.text
    # A run the fake trainer already drove to a terminal state is skipped by
    # design, so at least one — not necessarily every — sentinel appears.
    assert any(_job_files(job_id)['cancel'].is_file() for job_id in body['job_ids'])


def test_profiles_and_presets_are_served(api_client: Any) -> None:
    profiles = api_client.get('/train/profiles')
    assert profiles.status_code == 200, profiles.text
    assert profiles.json()['profiles'], profiles.text

    presets = api_client.get('/train/presets')
    assert presets.status_code == 200, presets.text
