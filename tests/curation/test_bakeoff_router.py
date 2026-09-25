"""Smoke tests for src/routers/curation/bakeoff.py.

The reference implementation's bake-off router has no dedicated test
file on the reference branch, so this is a new,
minimal smoke suite rather than a port — it exercises the read-only,
filesystem-driven endpoints (no OpenSearch dependency) with tmp_path
standing in for the shared jobs/output directories.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]

# CFG-6: this harness used to default to owner-private absolute paths --
# one of which named the location of a licensed proprietary image corpus
# (a "/mnt/<host-specific-mount>/..." style path) and must never appear
# in this repo as a literal string again. A generic "any absolute path"
# scan is too broad (it also matches route strings like '/bakeoff/run',
# shebangs, etc.) -- narrow this to host-mount-shaped absolute paths
# (/mnt/..., /home/..., /Users/...), which is exactly the shape the
# original regression had and nothing legitimate in this harness needs.
# Every text file in the harness tree is scanned (not a hand-picked list: a
# hand-picked list once missed a backend docstring carrying such a path).
_HARNESS_SUFFIXES = {'.py', '.json', '.txt', '.md'}


def _bakeoff_harness_files() -> list[str]:
    files = ['src/routers/curation/bakeoff.py']
    for root in ('scripts/curation/bakeoff', 'examples/bakeoff'):
        files += [
            p.relative_to(REPO_ROOT).as_posix()
            for p in sorted((REPO_ROOT / root).rglob('*'))
            if p.is_file() and p.suffix in _HARNESS_SUFFIXES
        ]
    return files


_HOST_MOUNT_PATH_RE = re.compile(r'/(?:mnt|home|Users)/[A-Za-z0-9_./\-]+')


def test_bakeoff_harness_has_no_owner_private_absolute_path_defaults() -> None:
    offenders: list[str] = []
    files = _bakeoff_harness_files()
    assert 'examples/bakeoff/license_plate/backends/lpdnet.py' in files
    for rel in files:
        text = (REPO_ROOT / rel).read_text()
        offenders.extend(f'{rel}: {match.group(0)}' for match in _HOST_MOUNT_PATH_RE.finditer(text))
    assert not offenders, (
        'bake-off harness file(s) contain a hardcoded host-mount-shaped '
        f'absolute path default: {offenders}'
    )


@pytest.fixture
def app_client() -> TestClient:
    from src.routers.curation import router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    return TestClient(app)


def test_bakeoff_router_is_registered() -> None:
    from src.main import app

    assert any(r.path == '/curation/bakeoff/runs' for r in app.routes)
    assert any(r.path == '/curation/bakeoff/eval_datasets' for r in app.routes)


def test_bakeoff_runs_empty_when_out_dir_missing(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'OUT_DIR', tmp_path / 'does-not-exist')
    r = app_client.get('/curation/bakeoff/runs')
    assert r.status_code == 200
    assert r.json() == {'runs': []}


def test_bakeoff_runs_lists_finished_jobs_newest_first(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    out_dir = tmp_path / 'out'
    for job_id, started_at in (
        ('job-older', '2026-01-01T00:00:00Z'),
        ('job-newer', '2026-02-01T00:00:00Z'),
    ):
        job_dir = out_dir / job_id
        job_dir.mkdir(parents=True)
        (job_dir / 'status.json').write_text(
            json.dumps({'state': 'finished', 'models': [], 'started_at': started_at})
        )
    monkeypatch.setattr(bakeoff, 'OUT_DIR', out_dir)

    r = app_client.get('/curation/bakeoff/runs')
    assert r.status_code == 200
    job_ids = [row['job_id'] for row in r.json()['runs']]
    assert job_ids == ['job-newer', 'job-older']


def test_bakeoff_status_404_when_missing(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'OUT_DIR', tmp_path / 'out')
    r = app_client.get('/curation/bakeoff/status/no-such-job')
    assert r.status_code == 404


def test_bakeoff_results_404_when_missing(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'OUT_DIR', tmp_path / 'out')
    r = app_client.get('/curation/bakeoff/results/no-such-job')
    assert r.status_code == 404


def test_bakeoff_run_requires_models_or_quantize(app_client: TestClient) -> None:
    r = app_client.post('/curation/bakeoff/run', json={'dataset': '/some/path'})
    assert r.status_code == 400


def test_bakeoff_run_requires_dataset(app_client: TestClient) -> None:
    r = app_client.post(
        '/curation/bakeoff/run',
        json={'models': [{'backend': 'ultralytics', 'name': 'm1'}]},
    )
    assert r.status_code == 400


def test_bakeoff_eval_datasets_empty_when_roots_missing(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'EVAL_DATASET_ROOTS', [('curated', tmp_path / 'nope')])
    r = app_client.get('/curation/bakeoff/eval_datasets')
    assert r.status_code == 200
    assert r.json() == {'datasets': [], 'count': 0}


def test_bakeoff_baseline_models_empty_when_registry_missing(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'BASELINES_PATH', tmp_path / 'no-such-baselines.json')
    r = app_client.get('/curation/bakeoff/baseline_models')
    assert r.status_code == 200
    assert r.json() == {'baselines': [], 'count': 0}


@pytest.fixture
def no_gpu_claim(monkeypatch) -> None:
    """Stub the GPU arbiter so POST /bakeoff/run touches no lock/containers."""
    import src.services.training.gpu_arbiter as arbiter

    class _Action:
        action = 'noop'

    async def _stop() -> _Action:
        return _Action()

    monkeypatch.setattr(arbiter, 'set_training_lock', lambda *_a, **_k: None)
    monkeypatch.setattr(arbiter, 'stop_gpu_services', _stop)


@pytest.mark.usefixtures('no_gpu_claim')
def test_bakeoff_run_writes_profile_into_job_spec(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'JOBS_DIR', tmp_path / 'jobs')
    monkeypatch.setattr(bakeoff, 'OUT_DIR', tmp_path / 'out')
    prof = tmp_path / 'mine.json'  # a .json path passes through to the evaluator
    r = app_client.post(
        '/curation/bakeoff/run',
        json={
            'dataset': '/data/ds',
            'job_id': 'jp1',
            'profile': str(prof),
            'models': [{'backend': 'ultralytics', 'name': 'm1', 'profile': 'generic'}],
        },
    )
    assert r.status_code == 200, r.text
    spec = json.loads((tmp_path / 'jobs' / 'jp1.job.json').read_text())
    assert spec['profile'] == str(prof)
    assert spec['models'][0]['profile'] == 'generic'


@pytest.mark.usefixtures('no_gpu_claim')
def test_bakeoff_run_without_profile_omits_it(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'JOBS_DIR', tmp_path / 'jobs')
    monkeypatch.setattr(bakeoff, 'OUT_DIR', tmp_path / 'out')
    r = app_client.post(
        '/curation/bakeoff/run',
        json={
            'dataset': '/d',
            'job_id': 'jp2',
            'models': [{'backend': 'ultralytics', 'name': 'm'}],
        },
    )
    assert r.status_code == 200
    spec = json.loads((tmp_path / 'jobs' / 'jp2.job.json').read_text())
    assert 'profile' not in spec
    assert 'profile' not in spec['models'][0]


@pytest.mark.parametrize('where', ['request', 'model'])
def test_bakeoff_run_rejects_unknown_profile(
    app_client: TestClient, monkeypatch, tmp_path: Path, where: str
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'JOBS_DIR', tmp_path / 'jobs')
    model = {'backend': 'ultralytics', 'name': 'm'}
    body: dict = {'dataset': '/d', 'models': [model]}
    if where == 'request':
        body['profile'] = 'no_such_profile'
    else:
        model['profile'] = 'no_such_profile'
    r = app_client.post('/curation/bakeoff/run', json=body)
    assert r.status_code == 400
    assert 'unknown bake-off profile' in r.json()['detail']
    assert not (tmp_path / 'jobs').exists()


@pytest.fixture
def clean_profile_env(monkeypatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith('OP_BAKEOFF_PROFILE'):
            monkeypatch.delenv(key)


@pytest.mark.usefixtures('clean_profile_env')
def test_bakeoff_profiles_lists_generic_only_by_default(app_client: TestClient) -> None:
    r = app_client.get('/curation/bakeoff/profiles')
    assert r.status_code == 200
    by_name = {p['name']: p for p in r.json()['profiles']}
    assert set(by_name) == {'generic'}  # example profiles are opt-in, never listed
    assert by_name['generic']['kind'] == 'registered'
    assert by_name['generic']['context_class_ids'] == []
    assert by_name['generic']['class_filter'] == []
    assert r.json()['default_profile'] == 'generic'
    assert by_name['generic']['default'] is True


@pytest.mark.usefixtures('clean_profile_env')
def test_bakeoff_profiles_default_from_json_path_and_field_overrides(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    f = tmp_path / 'widgets.json'
    f.write_text(json.dumps({'name': 'widgets', 'class_filter': ['widget']}))
    monkeypatch.setenv('OP_BAKEOFF_PROFILE', str(f))
    body = app_client.get('/curation/bakeoff/profiles').json()
    [row] = [p for p in body['profiles'] if p['default']]
    assert (row['name'], row['kind'], row['class_filter']) == (
        'widgets',
        'configured',
        ['widget'],
    )
    assert body['count'] == len(body['profiles'])

    monkeypatch.delenv('OP_BAKEOFF_PROFILE')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_CONTEXT_CLASS_IDS', '4')
    body = app_client.get('/curation/bakeoff/profiles').json()
    [row] = [p for p in body['profiles'] if p['default']]
    assert (row['name'], row['kind'], row['context_class_ids']) == ('generic', 'registered', [4])


@pytest.mark.usefixtures('clean_profile_env')
def test_bakeoff_profiles_bad_default_is_reported(app_client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv('OP_BAKEOFF_PROFILE', 'no_such_profile')
    body = app_client.get('/curation/bakeoff/profiles').json()
    assert body['default_profile'] is None
    assert 'unknown bake-off profile' in body['default_error']
    assert not any(p['default'] for p in body['profiles'])


def test_bakeoff_run_rejects_coreml_leg(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'JOBS_DIR', tmp_path / 'jobs')
    r = app_client.post(
        '/curation/bakeoff/run',
        json={'dataset': '/d', 'quantize': {'checkpoint': '/c.pt', 'coreml': True}},
    )
    assert r.status_code == 400
    assert 'CoreML export is not available' in r.json()['detail']
    assert not (tmp_path / 'jobs').exists()


def test_default_baseline_registry_is_empty(app_client: TestClient) -> None:
    body = app_client.get('/curation/bakeoff/baseline_models').json()
    assert body == {'baselines': [], 'count': 0}


def test_baseline_models_per_profile(app_client: TestClient) -> None:
    # Example profiles are not resolvable by name any more (opt-in, by path).
    r = app_client.get('/curation/bakeoff/baseline_models', params={'profile': 'license_plate'})
    assert r.status_code == 400
    r = app_client.get('/curation/bakeoff/baseline_models', params={'profile': 'generic'})
    assert r.status_code == 200
    assert 'lpr_nanov11_640' not in {b['name'] for b in r.json()['baselines']}
    assert (
        app_client.get('/curation/bakeoff/baseline_models', params={'profile': 'nope'}).status_code
        == 400
    )
    assert (
        app_client.get(
            '/curation/bakeoff/baseline_models', params={'profile': '../../etc/x.json'}
        ).status_code
        == 400
    )


def test_bakeoff_router_docstring_section_exists() -> None:
    from src.routers.curation import bakeoff

    doc = bakeoff.__doc__ or ''
    m = re.search(r'curation_design_rationale\.md`` §(\d+)', doc)
    assert m, 'router docstring should cite a specific rationale section'
    rationale = (REPO_ROOT / 'docs/design/curation_design_rationale.md').read_text()
    heading = re.search(rf'^## {m.group(1)}\. (.+)$', rationale, re.M)
    assert heading, f'rationale doc has no section {m.group(1)}'
    assert 'bake-off' in heading.group(1).lower()
