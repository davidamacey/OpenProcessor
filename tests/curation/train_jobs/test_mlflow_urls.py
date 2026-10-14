"""``mlflow_run_url`` is rebuilt from a public base at serving time, never the trainer's internal
container hostname.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.services.training import job_files, job_wire, jobs as train_jobs


class _FakeCurationConfig:
    def __init__(
        self,
        mlflow_public_url: str | None = None,
        api_prefix: str = '/curation',
        train_jobs_dir: Path | None = None,
    ) -> None:
        self.mlflow_public_url = mlflow_public_url
        self.mlflow_port = 0  # no request-derived URL in these tests
        self.api_prefix = api_prefix
        # _resolve_jobs_dir() reads this; these tests only care about
        # mlflow_public_url/api_prefix, so default to the same
        # OP_TRAIN_JOBS_DIR/projects/default the ``jobs_dir`` fixture
        # resolves to (default is an ordinary project -- P1: no
        # special-casing).
        # _resolve_jobs_dir() reads this (projects_plan.md §5.3); these
        # tests only care about mlflow_public_url/api_prefix. Default
        # nests /projects/default under OP_TRAIN_JOBS_DIR (P1R §6.1/D-A:
        # project_jobs_dir() always nests, default included).
        self.train_jobs_dir = train_jobs_dir or (
            Path(os.environ.get('OP_TRAIN_JOBS_DIR', '/jobs')) / 'projects' / 'default'
        )


def _patch_cfg(monkeypatch: pytest.MonkeyPatch, cfg: _FakeCurationConfig) -> None:
    # train_jobs reads the jobs dir through its own import; the MLflow URL
    # goes through resource_links.service_url, which imports from src.config.
    monkeypatch.setattr(train_jobs, 'get_curation_config', lambda: cfg)
    monkeypatch.setattr(job_files, 'get_curation_config', lambda: cfg)
    monkeypatch.setattr('src.config.get_curation_config', lambda: cfg)


def test_public_mlflow_url_builds_from_configured_base(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url='https://mlflow.example.com'))
    url = job_wire._public_mlflow_url(run_id='abc123', experiment_id='7')
    assert url == 'https://mlflow.example.com/#/experiments/7/runs/abc123'


def test_public_mlflow_url_none_when_base_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url=None))
    assert job_wire._public_mlflow_url(run_id='abc123', experiment_id='7') is None


def test_public_mlflow_url_none_when_run_or_experiment_id_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url='https://mlflow.example.com'))
    assert job_wire._public_mlflow_url(run_id=None, experiment_id='7') is None
    assert job_wire._public_mlflow_url(run_id='abc123', experiment_id=None) is None


def test_public_mlflow_url_never_leaks_the_internal_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The internal tracking URI (a container hostname) must never be the
    fallback -- unset public base means null, full stop."""
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url=None))
    url = job_wire._public_mlflow_url(run_id='abc123', experiment_id='7')
    assert url is None
    assert url != 'http://curation-mlflow:5000/#/experiments/7/runs/abc123'


@pytest.mark.asyncio
async def test_read_status_rewrites_mlflow_url_and_confusion_matrix(
    jobs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url='https://mlflow.example.com'))
    (jobs_dir / 'realmlf.status.json').write_text(
        json.dumps(
            {
                'job_id': 'realmlf',
                'state': 'finished',
                'mlflow_run_id': 'run-42',
                'mlflow_experiment_id': '3',
                'mlflow_run_url': 'http://curation-mlflow:5000/#/experiments/3/runs/run-42',
                'eval': {
                    'map50': 0.62,
                    'split': 'test',
                    'confusion_matrix_path': '/var/lib/openprocessor/training_runs/realmlf/'
                    'confusion_matrix.png',
                },
            }
        )
    )

    status = await train_jobs.read_status('realmlf')

    assert status is not None
    assert status.mlflow_run_url == 'https://mlflow.example.com/#/experiments/3/runs/run-42'
    assert status.eval is not None
    assert 'confusion_matrix_path' not in status.eval  # the fs path never reaches the wire
    assert (
        status.eval['confusion_matrix_url']
        == '/curation/projects/default/train/artifacts/realmlf/confusion_matrix.png'
    )
    assert status.eval['map50'] == 0.62  # untouched


@pytest.mark.asyncio
async def test_read_status_serves_null_mlflow_url_when_public_base_unset(
    jobs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url=None))
    (jobs_dir / 'nourl.status.json').write_text(
        json.dumps(
            {
                'job_id': 'nourl',
                'state': 'finished',
                'mlflow_run_id': 'run-1',
                'mlflow_experiment_id': '1',
                'mlflow_run_url': 'http://curation-mlflow:5000/#/experiments/1/runs/run-1',
            }
        )
    )

    status = await train_jobs.read_status('nourl')

    assert status is not None
    assert status.mlflow_run_url is None
    assert status.mlflow_run_id == 'run-1'  # kept even though the URL is null


@pytest.mark.asyncio
async def test_read_status_confusion_matrix_url_null_when_absent(
    jobs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url=None))
    (jobs_dir / 'noeval.status.json').write_text(
        json.dumps(
            {'job_id': 'noeval', 'state': 'finished', 'eval': {'map50': 0.5, 'split': 'val'}}
        )
    )

    status = await train_jobs.read_status('noeval')

    assert status is not None
    assert status.eval is not None
    assert status.eval['confusion_matrix_url'] is None


@pytest.mark.asyncio
async def test_read_manifest_rewrites_eval_and_mlflow_url(
    jobs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url='https://mlflow.example.com'))
    (jobs_dir / 'manifjob.manifest.json').write_text(
        json.dumps(
            {
                'job_id': 'manifjob',
                'results': {
                    'final_state': 'finished',
                    'mlflow_run_id': 'run-9',
                    'mlflow_experiment_id': '2',
                    'mlflow_run_url': 'http://curation-mlflow:5000/#/experiments/2/runs/run-9',
                    'eval': {
                        'map50': 0.7,
                        'split': 'test',
                        'confusion_matrix_path': '/var/lib/openprocessor/training_runs/'
                        'manifjob/confusion_matrix.png',
                    },
                },
            }
        )
    )

    manifest = await train_jobs.read_manifest('manifjob')

    assert manifest is not None
    results = manifest['results']
    assert results['mlflow_run_url'] == 'https://mlflow.example.com/#/experiments/2/runs/run-9'
    assert 'confusion_matrix_path' not in results['eval']
    assert (
        results['eval']['confusion_matrix_url']
        == '/curation/projects/default/train/artifacts/manifjob/confusion_matrix.png'
    )
