"""``mlflow_run_url`` is rebuilt from a public base at serving time, never the trainer's internal
container hostname.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import pytest

from src.db.repos.training_runs import RunRow
from src.services.training import job_manifest, job_wire
from src.services.training.run_wire import to_status


class _FakeCurationConfig:
    def __init__(self, mlflow_public_url: str | None = None, api_prefix: str = '/curation') -> None:
        self.mlflow_public_url = mlflow_public_url
        self.mlflow_port = 0  # no request-derived URL in these tests
        self.api_prefix = api_prefix


def _patch_cfg(monkeypatch: pytest.MonkeyPatch, cfg: _FakeCurationConfig) -> None:
    # The MLflow URL goes through resource_links.service_url, which imports from src.config.
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


def _row(manifest: dict[str, Any] | None = None, **progress_counters: Any) -> RunRow:
    return RunRow(
        job_id=uuid.UUID('00000000-0000-0000-0000-000000000001'),
        project_id=uuid.uuid4(),
        spec={},
        class_names=[],
        family='yolo26',
        base_model='yolo26n',
        artifact_prefix='projects/default/training/x',
        manifest=manifest,
        mlflow_run_id=None,
        gpu_uuids=[],
        status='completed',
        params={},
        progress={'counters': progress_counters},
        error_code=None,
        error_message=None,
        created_at=dt.datetime.now(dt.UTC),
        started_at=None,
        finished_at=dt.datetime.now(dt.UTC),
        heartbeat_at=None,
    )


def test_the_status_rewrites_mlflow_url_and_confusion_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url='https://mlflow.example.com'))
    row = _row(
        mlflow_run_id='run-42',
        mlflow_experiment_id='3',
        eval={
            'map50': 0.62,
            'split': 'test',
            'confusion_matrix_path': '/var/lib/openprocessor/training_runs/x/confusion_matrix.png',
        },
    )

    status = to_status(row)

    assert status.mlflow_run_url == 'https://mlflow.example.com/#/experiments/3/runs/run-42'
    assert status.eval is not None
    assert 'confusion_matrix_path' not in status.eval  # the fs path never reaches the wire
    assert (
        status.eval['confusion_matrix_url']
        == f'/curation/projects/default/train/artifacts/{row.job_id}/confusion_matrix.png'
    )
    assert status.eval['map50'] == 0.62  # untouched


def test_the_status_serves_null_mlflow_url_when_public_base_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url=None))

    status = to_status(_row(mlflow_run_id='run-1', mlflow_experiment_id='1'))

    assert status.mlflow_run_url is None
    assert status.mlflow_run_id == 'run-1'  # kept even though the URL is null


def test_the_confusion_matrix_url_is_null_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url=None))

    status = to_status(_row(eval={'map50': 0.5, 'split': 'val'}))

    assert status.eval is not None
    assert status.eval['confusion_matrix_url'] is None


@pytest.mark.asyncio
async def test_read_manifest_rewrites_eval_and_mlflow_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cfg(monkeypatch, _FakeCurationConfig(mlflow_public_url='https://mlflow.example.com'))
    manifest = {
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
    row = _row(manifest)

    async def run(_job_id: str) -> RunRow:
        return row

    monkeypatch.setattr(job_manifest, '_run', run)

    served = await job_manifest.read_manifest(str(row.job_id))

    assert served is not None
    results = served['results']
    assert results['mlflow_run_url'] == 'https://mlflow.example.com/#/experiments/2/runs/run-9'
    assert 'confusion_matrix_path' not in results['eval']
    assert (
        results['eval']['confusion_matrix_url']
        == f'/curation/projects/default/train/artifacts/{row.job_id}/confusion_matrix.png'
    )
