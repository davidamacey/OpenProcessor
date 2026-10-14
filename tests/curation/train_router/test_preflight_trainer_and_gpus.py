"""Preflight trainer reachability, GPU scoping against trainer capabilities, and GET /train/gpus."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from curation.train_router._helpers import _set_arbiter_config


if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from fastapi.testclient import TestClient


# =============================================================================
# /preflight
# =============================================================================


def test_preflight_smoke(app_client: TestClient, project_export_root: Path) -> None:
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'model_size': 'm',
        'profile': 'medium',
    }
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert 'blocked' in out
    assert 'checks' in out
    # We want every named check to appear so the UI can render the table.
    names = {c['name'] for c in out['checks']}
    expected = {
        'optimizer_not_auto',
        'free_disk',
        'trainer_reachable',
        'active_run',
        'class_balance',
        'test_holdout',
        'empty_labels',
        'region_pairing',
    }
    assert expected.issubset(names), f'missing checks: {expected - names}'
    from src.services.curation.dataset_thresholds import dataset_thresholds

    assert out['thresholds'] == dataset_thresholds()

    # P2-8: empty_labels/region_pairing used to be hardcoded 'ok' unconditionally
    # -- never actually scanned. str(project_export_root / 'x') doesn't exist on disk, so a
    # real implementation MUST report 'unknown' here, never 'ok' (a lie: nothing
    # was checked).
    empty_labels_check = next(c for c in out['checks'] if c['name'] == 'empty_labels')
    region_pairing_check = next(c for c in out['checks'] if c['name'] == 'region_pairing')
    assert empty_labels_check['severity'] == 'unknown', empty_labels_check
    assert region_pairing_check['severity'] == 'unknown', region_pairing_check


def test_preflight_blocks_when_trainer_unreachable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch, project_export_root: Path
) -> None:
    """P1-8: before the fix, preflight had no trainer-reachable check at
    all, so submitting a job when the trainer container was never started queued
    it forever with no error. Simulate "container not running"."""
    monkeypatch.setattr(
        'src.routers.curation_train.preflight.probe_trainer_reachable',
        AsyncMock(return_value=('block', "'trainer' container does not exist")),
    )
    body = {'dataset_export_dir': str(project_export_root / 'x'), 'profile': 'medium'}
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    trainer_check = next(c for c in out['checks'] if c['name'] == 'trainer_reachable')
    assert trainer_check['severity'] == 'block'
    assert out['blocked'] is True


def test_preflight_ok_when_trainer_reachable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch, project_export_root: Path
) -> None:
    monkeypatch.setattr(
        'src.routers.curation_train.preflight.probe_trainer_reachable',
        AsyncMock(return_value=('ok', "'trainer' is running")),
    )
    body = {'dataset_export_dir': str(project_export_root / 'x'), 'profile': 'medium'}
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    trainer_check = next(c for c in out['checks'] if c['name'] == 'trainer_reachable')
    assert trainer_check['severity'] == 'ok'


# =============================================================================
# trainer_gpus preflight check (GPU scoping)
# =============================================================================


def _write_trainer_capabilities(jobs_dir: Any, gpu_order: list[int]) -> None:
    (jobs_dir / '.trainer_capabilities.json').write_text(
        json.dumps({'gpu_order': gpu_order, 'visible_count': len(gpu_order) or None})
    )


def test_preflight_blocks_gpu_the_trainer_is_not_attached_to(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    """A capabilities file naming [2] and a request for host GPU 0 must
    block -- the original TR-2 failure mode (API accepts a GPU the trainer
    isn't attached to, job hangs in queued/starting forever)."""
    _write_trainer_capabilities(tmp_path, [2])
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
        'cuda_visible_devices': '0',
    }
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    check = next(c for c in out['checks'] if c['name'] == 'trainer_gpus')
    assert check['severity'] == 'block'
    assert '[2]' in check['message'] or '2' in str(check['detail']['trainer_gpu_order'])
    assert out['blocked'] is True


def test_preflight_ok_for_gpu_the_trainer_is_attached_to(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    _write_trainer_capabilities(tmp_path, [0, 2])
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
        'cuda_visible_devices': '2',
    }
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    check = next(c for c in out['checks'] if c['name'] == 'trainer_gpus')
    assert check['severity'] == 'ok'


def test_preflight_warns_when_no_capabilities_file(
    app_client: TestClient, project_export_root: Path
) -> None:
    """No file (older trainer image, or not started yet) is a warning, not
    a block -- an absent file must never be read as 'attached to nothing'."""
    body = {'dataset_export_dir': str(project_export_root / 'x'), 'profile': 'medium'}
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    check = next(c for c in out['checks'] if c['name'] == 'trainer_gpus')
    assert check['severity'] == 'warn'


def test_preflight_ok_when_capabilities_file_reports_unrestricted(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    """An empty gpu_order means the trainer sees every GPU at its host
    index (OP_TRAIN_GPU_ORDER unset) -- no restriction, any id passes."""
    _write_trainer_capabilities(tmp_path, [])
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
        'cuda_visible_devices': '5',
    }
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    check = next(c for c in out['checks'] if c['name'] == 'trainer_gpus')
    assert check['severity'] == 'ok'


def test_train_gpus_intersects_with_trainer_capabilities(
    app_client: TestClient, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/train/gpus must not offer a GPU the trainer isn't attached to, even
    though OP_GPU_ALLOWED_IDS never restricted it (a policy allowlist is a
    different fact than physical GPU attachment)."""
    from src.config import GpuArbiterConfig

    _set_arbiter_config(monkeypatch, GpuArbiterConfig())
    _write_trainer_capabilities(tmp_path, [2])
    r = app_client.get('/curation/projects/default/train/gpus')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['allowed_ids'] == [2]
    values = {o['value'] for o in body['options']}
    assert values == {'2'}


def test_start_returns_422_when_trainer_unreachable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch, project_export_root: Path
) -> None:
    """/start must refuse (not silently queue) when the trainer is down."""
    monkeypatch.setattr(
        'src.routers.curation_train.preflight.probe_trainer_reachable',
        AsyncMock(return_value=('block', "'trainer' container does not exist")),
    )
    body = {'dataset_export_dir': str(project_export_root / 'x'), 'profile': 'medium'}
    r = app_client.post('/curation/projects/default/train/start', json=body)
    assert r.status_code == 422, r.text
    detail = r.json()['detail']
    trainer_check = next(
        c for c in detail['preflight']['checks'] if c['name'] == 'trainer_reachable'
    )
    assert trainer_check['severity'] == 'block'


def test_preflight_blocks_optimizer_auto(app_client: TestClient, project_export_root: Path) -> None:
    body = {
        'dataset_export_dir': str(project_export_root / 'x'),
        'profile': 'medium',
        'hyperparameters': {'optimizer': 'auto'},
    }
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200
    out = r.json()
    optimizer_check = next(c for c in out['checks'] if c['name'] == 'optimizer_not_auto')
    assert optimizer_check['severity'] == 'block'
    assert out['blocked'] is True


# =============================================================================
# GET /train/gpus
# =============================================================================


def test_train_gpus_single_allowed_with_label_and_scoped_container(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import GpuArbiterConfig

    _set_arbiter_config(
        monkeypatch,
        GpuArbiterConfig(
            allowed_gpu_ids=frozenset({2}),
            containers=('vlm-inference-container',),
            container_gpus=(('vlm-inference-container', frozenset({2})),),
            gpu_labels={0: 'RTX A6000', 1: 'RTX 3080 Ti', 2: 'RTX A6000'},
        ),
    )
    r = app_client.get('/curation/projects/default/train/gpus')
    assert r.status_code == 200
    body = r.json()
    assert body['allowed_ids'] == [2]
    assert body['unrestricted'] is False
    assert len(body['options']) == 1
    opt = body['options'][0]
    assert opt['value'] == '2'
    assert opt['gpu_ids'] == [2]
    assert opt['label'] == 'RTX A6000 (GPU 2)'
    assert opt['stops_containers'] == ['vlm-inference-container']
    assert 'Stops vlm-inference-container' in opt['advisory']
    assert opt['default'] is True


def test_train_gpus_multi_id_allowlist(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import GpuArbiterConfig

    _set_arbiter_config(
        monkeypatch,
        GpuArbiterConfig(
            allowed_gpu_ids=frozenset({0, 2}),
            gpu_labels={0: 'RTX A6000', 2: 'RTX A6000'},
        ),
    )
    r = app_client.get('/curation/projects/default/train/gpus')
    assert r.status_code == 200
    body = r.json()
    assert body['allowed_ids'] == [0, 2]
    assert body['unrestricted'] is False
    values = {o['value']: o for o in body['options']}
    assert set(values) == {'0', '2', '0,2'}
    single0 = values['0']
    assert single0['gpu_ids'] == [0]
    assert single0['label'] == 'RTX A6000 (GPU 0)'
    assert single0['stops_containers'] == []
    assert 'Shares GPU' in single0['advisory']
    multi = values['0,2']
    assert multi['gpu_ids'] == [0, 2]
    assert multi['label'] == '2× RTX A6000 (GPUs 0,2)'
    # Default is the value matching default_train_gpu_value() -- smallest
    # allowed id when OP_TRAIN_DEFAULT_GPUS is unset.
    defaults = [o for o in body['options'] if o['default']]
    assert len(defaults) == 1
    assert defaults[0]['value'] == '0'


def test_train_gpus_unrestricted(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import GpuArbiterConfig

    _set_arbiter_config(monkeypatch, GpuArbiterConfig())
    r = app_client.get('/curation/projects/default/train/gpus')
    assert r.status_code == 200
    body = r.json()
    assert body['allowed_ids'] == []
    assert body['unrestricted'] is True
    assert len(body['options']) == 1
    assert body['options'][0]['value'] == '0'
    assert body['options'][0]['default'] is True


def test_train_gpus_no_labels_falls_back_to_gpu_id(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import GpuArbiterConfig

    _set_arbiter_config(monkeypatch, GpuArbiterConfig(allowed_gpu_ids=frozenset({3})))
    r = app_client.get('/curation/projects/default/train/gpus')
    body = r.json()
    assert body['options'][0]['label'] == 'GPU 3'


def test_train_gpus_mixed_labels_multi_option_lists_ids_only(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the multi-GPU ids don't all share a label, the multi option is
    just 'GPUs a,b', not a misleading '{n}x {label}'."""
    from src.config import GpuArbiterConfig

    _set_arbiter_config(
        monkeypatch,
        GpuArbiterConfig(
            allowed_gpu_ids=frozenset({0, 1}),
            gpu_labels={0: 'RTX A6000', 1: 'RTX 3080 Ti'},
        ),
    )
    r = app_client.get('/curation/projects/default/train/gpus')
    body = r.json()
    values = {o['value']: o for o in body['options']}
    assert values['0,1']['label'] == 'GPUs 0,1'
