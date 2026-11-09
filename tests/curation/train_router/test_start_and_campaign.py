"""POST /start and /start_campaign: queueing a run, refusals and the GPU-arbiter preflight."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch


if TYPE_CHECKING:
    from pathlib import Path
    from types import SimpleNamespace

    import pytest
    from fastapi.testclient import TestClient
from curation.train_router._helpers import _set_arbiter_config


# =============================================================================
# /start
# =============================================================================


def test_start_queues_a_run(
    app_client: TestClient, train_fake: SimpleNamespace, project_export_root: Path
) -> None:
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
    }
    with patch(
        'src.routers.curation.train.start.run_preflight',
        new=AsyncMock(
            return_value=__import__(
                'src.services.training.preflight_checks', fromlist=['PreflightReport']
            ).PreflightReport(blocked=False, checks=[], summary='ok')
        ),
    ):
        r = app_client.post('/curation/projects/default/train/start', json=body)
    assert r.status_code == 201, r.text
    out = r.json()
    assert out['job_id']
    [(queued, _kwargs)] = train_fake.started
    assert queued.profile == 'medium'
    assert queued.dataset_export_dir == str(project_export_root / 'x')


def test_start_rejects_optimizer_auto(app_client: TestClient, project_export_root: Path) -> None:
    """The router relies on the preflight check, which the live impl runs."""
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
        'hyperparameters': {'optimizer': 'auto'},
    }
    r = app_client.post('/curation/projects/default/train/start', json=body)
    assert r.status_code == 422, r.text
    detail = r.json()['detail']
    assert 'preflight' in detail
    assert detail['preflight']['blocked'] is True


def test_start_with_force_bypasses_block(
    app_client: TestClient, train_fake: SimpleNamespace, project_export_root: Path
) -> None:
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
        'hyperparameters': {'optimizer': 'auto'},
    }
    r = app_client.post('/curation/projects/default/train/start?force=true', json=body)
    # Even with force, optimizer=auto block path emits 422 only when not forced.
    # Here force=true allows past the blocked report. The route still queues the run.
    assert r.status_code == 201, r.text
    assert len(train_fake.started) == 1


def test_start_returns_409_when_active_run_exists(
    app_client: TestClient, train_fake: SimpleNamespace, project_export_root: Path
) -> None:
    """With a run in progress, /start answers 409 and queues nothing."""
    from src.services.training.job_models import TrainJobStatus

    train_fake.active = TrainJobStatus(job_id='live', state='running')
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
    }
    r = app_client.post('/curation/projects/default/train/start', json=body)
    assert r.status_code == 409, r.text
    detail = r.json()['detail']
    assert 'progress' in detail['message'].lower() or 'preflight' in detail
    assert train_fake.started == []


def test_start_queues_the_run_and_leaves_the_gpu_claim_to_the_trainer(
    app_client: TestClient,
    train_fake: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    project_export_root: Path,
) -> None:
    """The API holds no Docker socket: /start never stops a service. The trainer reserves its
    cards and waits for the arbiter tick (``tests/trainer``), so a configured service that
    shares the card does not change the answer here."""
    from src.config import GpuArbiterConfig

    _set_arbiter_config(
        monkeypatch,
        GpuArbiterConfig(
            allowed_gpu_ids=frozenset({0}),
            containers=('vlm-inference-container',),
            container_gpus=(('vlm-inference-container', frozenset({0})),),
        ),
    )
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
        'cuda_visible_devices': '0',
    }
    with patch(
        'src.routers.curation.train.start.run_preflight',
        new=AsyncMock(
            return_value=__import__(
                'src.services.training.preflight_checks', fromlist=['PreflightReport']
            ).PreflightReport(blocked=False, checks=[], summary='ok')
        ),
    ):
        r = app_client.post('/curation/projects/default/train/start', json=body)

    assert r.status_code == 201, r.text
    assert len(train_fake.started) == 1


def test_preflight_blocks_when_a_stop_is_required_and_no_arbiter_worker_is_alive(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch, project_export_root: Path
) -> None:
    """The gpu_arbiter preflight check goes 'block' when the claim would need to stop a service
    and ``gpu-arbiter-worker`` has no fresh liveness row, before the user submits."""
    from src.config import GpuArbiterConfig

    _set_arbiter_config(
        monkeypatch,
        GpuArbiterConfig(
            allowed_gpu_ids=frozenset({0}),
            containers=('vlm-inference-container',),
            container_gpus=(('vlm-inference-container', frozenset({0})),),
        ),
    )
    monkeypatch.setattr(
        'src.services.training.preflight_run.arbiter_consumer_live',
        AsyncMock(return_value=False),
    )

    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
        'cuda_visible_devices': '0',
    }
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    checks = {c['name']: c for c in r.json()['checks']}
    assert checks['gpu_arbiter']['severity'] == 'block'
    assert 'vlm-inference-container' in checks['gpu_arbiter']['message']
    assert r.json()['blocked'] is True


def test_preflight_names_the_services_it_will_stop_when_the_arbiter_worker_is_alive(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch, project_export_root: Path
) -> None:
    from src.config import GpuArbiterConfig

    _set_arbiter_config(
        monkeypatch,
        GpuArbiterConfig(
            allowed_gpu_ids=frozenset({0}),
            containers=('vlm-inference-container',),
            container_gpus=(('vlm-inference-container', frozenset({0})),),
        ),
    )
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
        'cuda_visible_devices': '0',
    }
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    checks = {c['name']: c for c in r.json()['checks']}
    assert checks['gpu_arbiter']['severity'] == 'ok'
    assert checks['gpu_arbiter']['message'] == 'will stop: vlm-inference-container'


# =============================================================================
# /start_campaign
# =============================================================================


def test_start_campaign_queues_n_runs(
    app_client: TestClient, train_fake: SimpleNamespace, project_export_root: Path
) -> None:
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'runs': [
            {'profile': 'nano', 'model_size': 'n'},
            {'profile': 'medium', 'model_size': 'm'},
        ],
    }
    r = app_client.post('/curation/projects/default/train/start_campaign?force=true', json=body)
    assert r.status_code == 201, r.text
    out = r.json()
    assert len(out['job_ids']) == 2
    assert out['campaign_id'] == 'camp-1'
    assert len(train_fake.campaigns[0].runs) == 2


def test_start_campaign_rejects_empty_runs(
    app_client: TestClient, project_export_root: Path
) -> None:
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'runs': [],
    }
    r = app_client.post('/curation/projects/default/train/start_campaign', json=body)
    assert r.status_code == 422  # Pydantic validation
