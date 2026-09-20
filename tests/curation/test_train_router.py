"""Tests for the curation_train router (Phase 1b).

We mount the router on a minimal FastAPI app, override the OpenSearch
dep with an AsyncMock, and patch ``train_jobs.*`` so no real ``/jobs/``
volume is required. Each test exercises a single endpoint.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_opensearch() -> AsyncMock:
    """An AsyncMock with the methods the preflight uses."""
    fake = AsyncMock()
    fake.search = AsyncMock(
        return_value={
            'hits': {'hits': [], 'total': {'value': 0}},
            'aggregations': {'by_class': {'buckets': []}},
        }
    )
    # _count_pending_ingest (curation_train.py) does
    # `int(resp.get('count', 0))` on the awaited result — a *sync* dict
    # method call. Left unstubbed, AsyncMock auto-generates `.count` (and
    # then `.get` on its return value) as further AsyncMocks, so `resp`
    # ends up being an AsyncMock instead of a dict and `.get(...)` itself
    # returns an unawaited coroutine (TypeError: int() argument ... not
    # 'coroutine'). Stub it with a real dict so `.get` behaves like OpenSearch's
    # actual `count` API response.
    fake.count = AsyncMock(return_value={'count': 0})
    return fake


@pytest.fixture
def app_client(
    fake_opensearch: AsyncMock,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """Build a minimal FastAPI app with just the train router."""
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))

    # Stub the registry so preflight class-resolution doesn't blow up.
    class _Reg:
        def load(self) -> Any:
            class _Snap:
                classes: list[Any] = []

            return _Snap()

        def get(self, _cid: int) -> Any:
            return None

    monkeypatch.setattr(
        'src.routers.curation_train.get_class_registry',
        lambda: _Reg(),
    )

    from src.routers.curation._common import _raw_opensearch_dep
    from src.routers.curation_train import router as kb_train_router

    app = FastAPI()
    app.include_router(kb_train_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch

    with TestClient(app) as client:
        yield client


# =============================================================================
# /preflight
# =============================================================================


def test_preflight_smoke(app_client: TestClient) -> None:
    body = {
        'dataset_export_dir': '/data/exports/x',
        'model_size': 'm',
        'profile': 'medium',
    }
    r = app_client.post('/curation/train/preflight', json=body)
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

    # P2-8: empty_labels/region_pairing used to be hardcoded 'ok' unconditionally
    # -- never actually scanned. '/data/exports/x' doesn't exist on disk, so a
    # real implementation MUST report 'unknown' here, never 'ok' (a lie: nothing
    # was checked).
    empty_labels_check = next(c for c in out['checks'] if c['name'] == 'empty_labels')
    region_pairing_check = next(c for c in out['checks'] if c['name'] == 'region_pairing')
    assert empty_labels_check['severity'] == 'unknown', empty_labels_check
    assert region_pairing_check['severity'] == 'unknown', region_pairing_check


def test_preflight_blocks_when_trainer_unreachable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-8: before the fix, preflight had no trainer-reachable check at
    all, so submitting a job when the trainer container was never started queued
    it forever with no error. Simulate "container not running"."""
    monkeypatch.setattr(
        'src.routers.curation_train.probe_trainer_reachable',
        AsyncMock(return_value=(False, "'trainer' container does not exist")),
    )
    body = {'dataset_export_dir': '/data/exports/x', 'profile': 'medium'}
    r = app_client.post('/curation/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    trainer_check = next(c for c in out['checks'] if c['name'] == 'trainer_reachable')
    assert trainer_check['severity'] == 'block'
    assert out['blocked'] is True


def test_preflight_ok_when_trainer_reachable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'src.routers.curation_train.probe_trainer_reachable',
        AsyncMock(return_value=(True, "'trainer' is running")),
    )
    body = {'dataset_export_dir': '/data/exports/x', 'profile': 'medium'}
    r = app_client.post('/curation/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    trainer_check = next(c for c in out['checks'] if c['name'] == 'trainer_reachable')
    assert trainer_check['severity'] == 'ok'


def test_start_returns_422_when_trainer_unreachable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/start must refuse (not silently queue) when the trainer is down."""
    monkeypatch.setattr(
        'src.routers.curation_train.probe_trainer_reachable',
        AsyncMock(return_value=(False, "'trainer' container does not exist")),
    )
    body = {'dataset_export_dir': '/data/exports/x', 'profile': 'medium'}
    r = app_client.post('/curation/train/start', json=body)
    assert r.status_code == 422, r.text
    detail = r.json()['detail']
    trainer_check = next(
        c for c in detail['preflight']['checks'] if c['name'] == 'trainer_reachable'
    )
    assert trainer_check['severity'] == 'block'


def test_preflight_blocks_optimizer_auto(app_client: TestClient) -> None:
    body = {
        'dataset_export_dir': '/data/exports/x',
        'profile': 'medium',
        'hyperparameters': {'optimizer': 'auto'},
    }
    r = app_client.post('/curation/train/preflight', json=body)
    assert r.status_code == 200
    out = r.json()
    optimizer_check = next(c for c in out['checks'] if c['name'] == 'optimizer_not_auto')
    assert optimizer_check['severity'] == 'block'
    assert out['blocked'] is True


# =============================================================================
# free-disk preflight check (P2-8)
# =============================================================================


def test_free_gb_returns_none_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Before the fix, an OSError from shutil.disk_usage() returned
    float('inf') -- 'infinite free space' -- instead of a signal the check
    couldn't run. It must return None so the caller reports 'unknown'."""
    from src.routers.curation_train import _free_gb

    def _raise(_path: str) -> None:
        raise OSError('no such path')

    monkeypatch.setattr('src.routers.curation_train.shutil.disk_usage', _raise)
    assert _free_gb('/does/not/matter') is None


def test_free_gb_normal_case_returns_sane_number(tmp_path: Any) -> None:
    from src.routers.curation_train import _free_gb

    free = _free_gb(str(tmp_path))
    assert free is not None
    assert free > 0


def test_training_volume_mount_sane_false_when_same_device_as_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact bug this check exists to catch: the real training volume
    isn't mounted, so the 'data' path silently resolves to the container's
    own root filesystem."""
    from pathlib import Path

    from src.routers.curation_train import _training_volume_mount_sane

    class _Stat:
        st_dev = 42

    real_stat = Path.stat

    def _fake_stat(self: Path, *, follow_symlinks: bool = True) -> object:
        if str(self) in ('/mnt/nvm/train_staging', '/'):
            return _Stat()
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, 'stat', _fake_stat)
    assert _training_volume_mount_sane('/mnt/nvm/train_staging') is False


def test_training_volume_mount_sane_true_for_distinct_device(tmp_path: Any) -> None:
    from src.routers.curation_train import _training_volume_mount_sane

    # tmp_path and '/' are the same device in most CI sandboxes, so this
    # only asserts the function runs without raising and returns a bool --
    # the real "different device" case is covered by the monkeypatched
    # test above, which is what actually exercises the guard's logic.
    result = _training_volume_mount_sane(str(tmp_path))
    assert isinstance(result, bool)


def test_preflight_blocks_when_mount_not_sane(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'src.routers.curation_train._training_volume_mount_sane',
        lambda _path: False,
    )
    body = {'dataset_export_dir': '/data/exports/x', 'profile': 'medium'}
    r = app_client.post('/curation/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    disk_check = next(c for c in out['checks'] if c['name'] == 'free_disk')
    assert disk_check['severity'] == 'block'
    assert 'mount' in disk_check['message'].lower()
    assert out['blocked'] is True


def test_preflight_reports_unknown_not_ok_when_disk_unreadable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'src.routers.curation_train._training_volume_mount_sane',
        lambda _path: True,
    )
    monkeypatch.setattr('src.routers.curation_train._free_gb', lambda _path: None)
    body = {'dataset_export_dir': '/data/exports/x', 'profile': 'medium'}
    r = app_client.post('/curation/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    disk_check = next(c for c in out['checks'] if c['name'] == 'free_disk')
    assert disk_check['severity'] == 'unknown'


# =============================================================================
# include_classes resolvable against the export (P2-8)
# =============================================================================


def test_unresolvable_include_classes_flags_unknown_ids(tmp_path: Any) -> None:
    import json

    from src.routers.curation_train import _unresolvable_include_classes

    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    (export_dir / 'class_registry.json').write_text(json.dumps({'export_id_map': {'1': 0, '2': 1}}))

    assert _unresolvable_include_classes(str(export_dir), [1, 2]) == []
    assert _unresolvable_include_classes(str(export_dir), [1, 999]) == [999]


def test_unresolvable_include_classes_no_export_id_map_flags_all(tmp_path: Any) -> None:
    """A pre-Phase-5 export (no export_id_map) can't resolve any id — every
    requested class is unresolvable, not a silent pass."""
    import json

    from src.routers.curation_train import _unresolvable_include_classes

    export_dir = tmp_path / 'export_old'
    export_dir.mkdir()
    (export_dir / 'class_registry.json').write_text(json.dumps({'classes': []}))

    assert _unresolvable_include_classes(str(export_dir), [1, 2]) == [1, 2]


def test_preflight_blocks_unresolvable_include_classes(
    app_client: TestClient, tmp_path: Any
) -> None:
    import json

    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    (export_dir / 'class_registry.json').write_text(json.dumps({'export_id_map': {'1': 0}}))

    body = {
        'dataset_export_dir': str(export_dir),
        'profile': 'medium',
        'include_classes': [1, 999],
    }
    r = app_client.post('/curation/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    check = next(c for c in out['checks'] if c['name'] == 'include_classes_resolvable')
    assert check['severity'] == 'block'
    assert 999 in check['detail']['unresolvable_class_ids']
    assert out['blocked'] is True


def test_preflight_skips_include_classes_check_for_lpr(
    app_client: TestClient, tmp_path: Any
) -> None:
    """LPR jobs have no include_classes concept — the check must not even
    appear, per the existing dataset_kind branch."""
    import json

    export_dir = tmp_path / 'export_lpr'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset_kind': 'lpr_single_class',
                'positive_images': 1000,
                'split_counts': {'test': 100},
            }
        )
    )

    body = {
        'dataset_export_dir': str(export_dir),
        'profile': 'medium',
        'include_classes': [999],  # would be unresolvable for a vehicle export
    }
    r = app_client.post('/curation/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    names = {c['name'] for c in out['checks']}
    assert 'include_classes_resolvable' not in names


# =============================================================================
# empty_labels / region_pairing real scan (P2-8)
# =============================================================================


def test_preflight_empty_labels_blocks_when_whole_export_is_empty(
    app_client: TestClient, tmp_path: Any
) -> None:
    import json

    export_dir = tmp_path / 'export'
    (export_dir / 'labels' / 'train').mkdir(parents=True)
    (export_dir / 'labels' / 'train' / 'a.txt').write_text('')
    (export_dir / 'class_registry.json').write_text(
        json.dumps({'classes': [{'class_id': 1, 'class_name': 'sedan'}], 'export_id_map': {'1': 0}})
    )
    (export_dir / 'manifest.json').write_text(json.dumps({'dataset_kind': 'vehicle'}))

    body = {'dataset_export_dir': str(export_dir), 'profile': 'medium'}
    r = app_client.post('/curation/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    check = next(c for c in out['checks'] if c['name'] == 'empty_labels')
    assert check['severity'] == 'block'
    assert out['blocked'] is True


def test_preflight_lpr_export_skips_scan_and_reports_not_applicable(
    app_client: TestClient, tmp_path: Any
) -> None:
    import json

    export_dir = tmp_path / 'export_lpr'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset_kind': 'lpr_single_class',
                'positive_images': 900,
                'total_images': 1000,
                'split_counts': {'test': 100},
            }
        )
    )

    body = {'dataset_export_dir': str(export_dir), 'profile': 'medium'}
    r = app_client.post('/curation/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    empty_check = next(c for c in out['checks'] if c['name'] == 'empty_labels')
    plate_check = next(c for c in out['checks'] if c['name'] == 'region_pairing')
    assert empty_check['severity'] == 'ok'
    assert 'background' in empty_check['message'].lower()
    assert plate_check['severity'] == 'ok'
    assert 'not applicable' in plate_check['message'].lower()


# =============================================================================
# /start
# =============================================================================


def test_start_writes_job(app_client: TestClient, tmp_path: Any) -> None:
    body = {
        'dataset_export_dir': '/data/exports/x',
        'profile': 'medium',
    }
    with patch(
        'src.routers.curation_train._run_preflight',
        new=AsyncMock(
            return_value=__import__(
                'src.routers.curation_train', fromlist=['PreflightReport']
            ).PreflightReport(blocked=False, checks=[], summary='ok')
        ),
    ):
        r = app_client.post('/curation/train/start', json=body)
    assert r.status_code == 201, r.text
    out = r.json()
    assert out['job_id']
    # File was written
    assert any(tmp_path.glob(f'{out["job_id"]}.job.json'))


def test_start_rejects_optimizer_auto(app_client: TestClient) -> None:
    """The router relies on the preflight check, which the live impl runs."""
    body = {
        'dataset_export_dir': '/data/exports/x',
        'profile': 'medium',
        'hyperparameters': {'optimizer': 'auto'},
    }
    r = app_client.post('/curation/train/start', json=body)
    assert r.status_code == 422, r.text
    detail = r.json()['detail']
    assert 'preflight' in detail
    assert detail['preflight']['blocked'] is True


def test_start_with_force_bypasses_block(app_client: TestClient, tmp_path: Any) -> None:
    body = {
        'dataset_export_dir': '/data/exports/x',
        'profile': 'medium',
        'hyperparameters': {'optimizer': 'auto'},
    }
    r = app_client.post('/curation/train/start?force=true', json=body)
    # Even with force, optimizer=auto block path emits 422 only when not forced.
    # Here force=true allows past the blocked report. The route still writes.
    assert r.status_code == 201, r.text
    job_id = r.json()['job_id']
    assert any(tmp_path.glob(f'{job_id}.job.json'))


def test_start_returns_409_when_active_run_exists(
    app_client: TestClient,
    tmp_path: Any,
) -> None:
    """Drop a status.json with state=running; /start should 409."""
    import json
    from datetime import UTC, datetime

    (tmp_path / 'live.status.json').write_text(
        json.dumps(
            {
                'job_id': 'live',
                'state': 'running',
                'heartbeat_at': datetime.now(UTC).isoformat(),
            }
        )
    )
    body = {
        'dataset_export_dir': '/data/exports/x',
        'profile': 'medium',
    }
    r = app_client.post('/curation/train/start', json=body)
    assert r.status_code == 409, r.text
    detail = r.json()['detail']
    assert 'progress' in detail['message'].lower() or 'preflight' in detail


# =============================================================================
# /start_campaign
# =============================================================================


def test_start_campaign_writes_n_jobs(app_client: TestClient, tmp_path: Any) -> None:
    body = {
        'dataset_export_dir': '/data/exports/x',
        'runs': [
            {'profile': 'nano', 'model_size': 'n'},
            {'profile': 'medium', 'model_size': 'm'},
        ],
    }
    r = app_client.post('/curation/train/start_campaign?force=true', json=body)
    assert r.status_code == 201, r.text
    out = r.json()
    assert len(out['job_ids']) == 2
    for jid in out['job_ids']:
        assert any(tmp_path.glob(f'{jid}.job.json'))


def test_start_campaign_rejects_empty_runs(app_client: TestClient) -> None:
    body = {
        'dataset_export_dir': '/data/exports/x',
        'runs': [],
    }
    r = app_client.post('/curation/train/start_campaign', json=body)
    assert r.status_code == 422  # Pydantic validation


# =============================================================================
# /status
# =============================================================================


def test_status_endpoint_returns_null_when_empty(app_client: TestClient) -> None:
    r = app_client.get('/curation/train/status')
    assert r.status_code == 200
    assert r.json() is None


def test_status_by_id_404(app_client: TestClient) -> None:
    r = app_client.get('/curation/train/status/nope')
    assert r.status_code == 404


def test_status_by_id_returns_status(app_client: TestClient, tmp_path: Any) -> None:
    import json
    from datetime import UTC, datetime

    (tmp_path / 'real.status.json').write_text(
        json.dumps(
            {
                'job_id': 'real',
                'state': 'running',
                'heartbeat_at': datetime.now(UTC).isoformat(),
            }
        )
    )
    r = app_client.get('/curation/train/status/real')
    assert r.status_code == 200
    body = r.json()
    assert body['job_id'] == 'real'
    assert body['state'] == 'running'


def test_status_by_id_invalid_id(app_client: TestClient) -> None:
    r = app_client.get('/curation/train/status/..%2Fescape')
    # Path-param percent decoding is up to FastAPI; either 400 or 404 is fine.
    assert r.status_code in (400, 404)


# =============================================================================
# /runs
# =============================================================================


def test_runs_list_empty(app_client: TestClient) -> None:
    r = app_client.get('/curation/train/runs')
    assert r.status_code == 200
    body = r.json()
    assert body['items'] == []
    assert body['total'] == 0


# =============================================================================
# /log/tail
# =============================================================================


def test_log_tail_returns_lines(app_client: TestClient, tmp_path: Any) -> None:
    (tmp_path / 'logj.run.log').write_text('a\nb\nc\nd\n')
    r = app_client.get('/curation/train/log/tail/logj?lines=2')
    assert r.status_code == 200
    body = r.json()
    assert body['job_id'] == 'logj'
    assert body['lines'] == ['c', 'd']


# =============================================================================
# /cancel
# =============================================================================


def test_cancel_writes_sentinel(app_client: TestClient, tmp_path: Any) -> None:
    r = app_client.post('/curation/train/cancel/some_job')
    assert r.status_code == 200, r.text
    assert (tmp_path / 'some_job.cancel').exists()
    assert r.json()['cancelled'] is True


def test_cancel_campaign(app_client: TestClient, tmp_path: Any) -> None:
    import json
    from datetime import UTC, datetime

    # Two jobs, both belonging to one campaign, one running, one queued.
    for jid in ('camp_run00', 'camp_run01'):
        (tmp_path / f'{jid}.job.json').write_text(
            json.dumps({'job_id': jid, 'campaign_id': 'camp', 'dataset_export_dir': '/x'})
        )
        (tmp_path / f'{jid}.status.json').write_text(
            json.dumps(
                {
                    'job_id': jid,
                    'campaign_id': 'camp',
                    'state': 'running',
                    'heartbeat_at': datetime.now(UTC).isoformat(),
                }
            )
        )
    r = app_client.post('/curation/train/cancel_campaign/camp')
    assert r.status_code == 200, r.text
    assert r.json()['cancelled'] == 2
    assert (tmp_path / 'camp_run00.cancel').exists()
    assert (tmp_path / 'camp_run01.cancel').exists()


# =============================================================================
# /profiles, /presets
# =============================================================================


def test_profiles_endpoint(app_client: TestClient) -> None:
    r = app_client.get('/curation/train/profiles')
    assert r.status_code == 200
    body = r.json()
    assert 'profiles' in body
    names = {p['name'] for p in body['profiles']}
    assert {'probe', 'nano', 'small', 'medium', 'large', 'xlarge'}.issubset(names)
    medium = next(p for p in body['profiles'] if p['name'] == 'medium')
    assert medium['defaults']['optimizer'] == 'MuSGD'


def test_presets_endpoint(app_client: TestClient) -> None:
    r = app_client.get('/curation/train/presets')
    assert r.status_code == 200
    body = r.json()
    assert 'class_subset_presets' in body
    names = {p['name'] for p in body['class_subset_presets']}
    assert {
        'all_vehicles',
        'plates_only',
        'vehicles_and_plates',
        'cars_only',
        'bikes_only',
    }.issubset(names)


# =============================================================================
# Promote gate (design §15.2)
# =============================================================================


def test_promote_gate_passes_when_metrics_meet_thresholds() -> None:
    from src.routers.curation_train import _evaluate_promote_gate

    eval_block = {
        'map50': 0.91,
        'map50_95': 0.74,
        'per_class': [
            {'class_id': 12, 'name': 'pickup', 'precision': 0.94, 'support': 1240},
            {'class_id': 81, 'name': 'motorcycle', 'precision': 0.83, 'support': 412},
        ],
    }
    assert _evaluate_promote_gate(eval_block) == []


def test_promote_gate_blocks_low_map50() -> None:
    from src.routers.curation_train import _evaluate_promote_gate

    failures = _evaluate_promote_gate({'map50': 0.50, 'per_class': []})
    assert any('mAP50' in m for m in failures)


def test_promote_gate_blocks_low_class_precision() -> None:
    from src.routers.curation_train import _evaluate_promote_gate

    failures = _evaluate_promote_gate(
        {
            'map50': 0.80,
            'per_class': [
                {'class_id': 12, 'name': 'pickup', 'precision': 0.30, 'support': 100},
            ],
        }
    )
    assert any('precision' in m for m in failures)


def test_promote_gate_blocks_low_support() -> None:
    from src.routers.curation_train import _evaluate_promote_gate

    failures = _evaluate_promote_gate(
        {
            'map50': 0.80,
            'per_class': [
                {'class_id': 12, 'name': 'pickup', 'precision': 0.95, 'support': 2},
            ],
        }
    )
    assert any('support' in m for m in failures)


def test_promote_gate_blocks_missing_eval_block() -> None:
    from src.routers.curation_train import _evaluate_promote_gate

    failures = _evaluate_promote_gate(None)
    assert failures == ['no eval block in status.json — trainer never ran val()']


def test_promote_endpoint_returns_422_when_gate_fails(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: gate failure surfaces as 422 with structured detail."""
    from src.services.training.jobs import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id='gate-fail-job',
        state='finished',
        checkpoint_path='/jobs/gate-fail-job/best.pt',
        eval={'map50': 0.50, 'per_class': []},  # below 0.65 floor
    )

    async def _fake_read_status(job_id: str) -> TrainJobStatus | None:
        return fake_status if job_id == 'gate-fail-job' else None

    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        _fake_read_status,
    )

    r = app_client.post(
        '/curation/train/promote/gate-fail-job',
        json={'triton_name': 'yolo26m_fail'},
    )
    assert r.status_code == 422, r.text
    detail = r.json()['detail']
    assert detail['message'] == 'promote gate failed'
    assert any('mAP50' in f for f in detail['failures'])
    assert detail['thresholds']['map50_min'] == 0.65


def test_manifest_endpoint_returns_404_when_absent(app_client: TestClient) -> None:
    r = app_client.get('/curation/train/manifest/no-such-job')
    assert r.status_code == 404


def test_manifest_endpoint_returns_payload_when_present(
    app_client: TestClient, tmp_path: Any
) -> None:
    """The labeler reads the manifest verbatim; we return whatever JSON is on disk."""
    job_id = '20260509-test-job'
    payload = {
        'kind': 'train',
        'job_id': job_id,
        'lineage': {'export_dir': '/data/exports/x', 'include_classes': [12, 81]},
        'results': {'eval': {'map50': 0.91}},
        'promoted_to': None,
    }
    import json

    (tmp_path / f'{job_id}.manifest.json').write_text(json.dumps(payload))

    r = app_client.get(f'/curation/train/manifest/{job_id}')
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['kind'] == 'train'
    assert out['lineage']['include_classes'] == [12, 81]


def test_promote_endpoint_force_bypasses_gate(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force=true skips the gate and proceeds to the (mocked) handoff."""
    from pathlib import Path

    from src.services.training.jobs import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id='force-job',
        state='finished',
        checkpoint_path='/jobs/force-job/best.pt',
        eval={'map50': 0.10, 'per_class': []},
    )

    async def _fake_read_status(job_id: str) -> TrainJobStatus | None:
        return fake_status if job_id == 'force-job' else None

    async def _fake_promote(**_kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        return PromoteResult(
            job_id='force-job',
            triton_name='yolo26m_forced',
            onnx_path=str(Path('/triton-models/yolo26m_forced/1/model.onnx')),
            config_path=str(Path('/triton-models/yolo26m_forced/config.pbtxt')),
            labels_path=str(Path('/triton-models/yolo26m_forced/labels.txt')),
            triton_loaded=True,
        )

    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        _fake_read_status,
    )
    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton',
        _fake_promote,
    )

    r = app_client.post(
        '/curation/train/promote/force-job',
        json={'triton_name': 'yolo26m_forced', 'force': True},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['triton_name'] == 'yolo26m_forced'
    assert out['triton_loaded'] is True


# =============================================================================
# Phase 9 — P1-9 gate strictness (non-numeric metrics must FAIL, not skip)
# =============================================================================


def test_gate_blocks_null_precision() -> None:
    """Before the fix: a null precision short-circuits `isinstance(...) and`
    to False, so the row is silently treated as passing."""
    from src.routers.curation_train import _evaluate_promote_gate

    failures = _evaluate_promote_gate(
        {
            'map50': 0.80,
            'per_class': [
                {'class_id': 12, 'name': 'pickup', 'precision': None, 'support': 100},
            ],
        }
    )
    assert any('pickup' in m and 'precision' in m for m in failures), failures


def test_gate_blocks_string_support() -> None:
    """Before the fix: a string support ("n/a") short-circuits the same way."""
    from src.routers.curation_train import _evaluate_promote_gate

    failures = _evaluate_promote_gate(
        {
            'map50': 0.80,
            'per_class': [
                {'class_id': 12, 'name': 'pickup', 'precision': 0.95, 'support': 'n/a'},
            ],
        }
    )
    assert any('pickup' in m and 'support' in m for m in failures), failures


# =============================================================================
# Phase 9 — P1-9 force=true must return the gate report + persist force_used
# =============================================================================


def test_force_promote_returns_gate_report(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the fix: the computed gate report is discarded on force=true —
    the 200 response carries no evidence the gate ever ran or what it found."""
    from pathlib import Path

    from src.services.training.jobs import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id='force-report-job',
        state='finished',
        checkpoint_path='/jobs/force-report-job/best.pt',
        eval={'map50': 0.10, 'per_class': []},  # well below the 0.65 floor
    )

    async def _fake_read_status(job_id: str) -> TrainJobStatus | None:
        return fake_status if job_id == 'force-report-job' else None

    async def _fake_promote(**_kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        return PromoteResult(
            job_id='force-report-job',
            triton_name='yolo26m_forced_report',
            onnx_path=str(Path('/triton-models/yolo26m_forced_report/1/model.onnx')),
            config_path=str(Path('/triton-models/yolo26m_forced_report/config.pbtxt')),
            labels_path=str(Path('/triton-models/yolo26m_forced_report/labels.txt')),
            triton_loaded=True,
        )

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)
    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton', _fake_promote
    )

    r = app_client.post(
        '/curation/train/promote/force-report-job',
        json={'triton_name': 'yolo26m_forced_report', 'force': True},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['force_used'] is True
    assert out['gate_report'] is not None
    assert any('mAP50' in f for f in out['gate_report']['failures'])
    assert out['gate_report']['thresholds']['map50_min'] == 0.65


def test_force_promote_records_force_used_in_manifest(
    app_client: TestClient, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the fix: nothing about force=true reaches the manifest at all."""
    import json
    from pathlib import Path

    from src.services.training.jobs import TrainJobStatus

    job_id = 'force-manifest-job'
    (tmp_path / f'{job_id}.manifest.json').write_text(
        json.dumps({'kind': 'train', 'job_id': job_id, 'promoted_to': None})
    )

    fake_status = TrainJobStatus(
        job_id=job_id,
        state='finished',
        checkpoint_path=f'/jobs/{job_id}/best.pt',
        eval={'map50': 0.10, 'per_class': []},
    )

    async def _fake_read_status(jid: str) -> TrainJobStatus | None:
        return fake_status if jid == job_id else None

    async def _fake_promote(**_kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        return PromoteResult(
            job_id=job_id,
            triton_name='yolo26m_forced_manifest',
            onnx_path=str(Path('/triton-models/yolo26m_forced_manifest/1/model.onnx')),
            config_path=str(Path('/triton-models/yolo26m_forced_manifest/config.pbtxt')),
            labels_path=str(Path('/triton-models/yolo26m_forced_manifest/labels.txt')),
            triton_loaded=True,
        )

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)
    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton', _fake_promote
    )

    r = app_client.post(
        f'/curation/train/promote/{job_id}',
        json={'triton_name': 'yolo26m_forced_manifest', 'force': True},
    )
    assert r.status_code == 200, r.text
    assert r.json()['lineage_stamped'] is True

    manifest = json.loads((tmp_path / f'{job_id}.manifest.json').read_text())
    promoted_to = manifest['promoted_to']
    assert promoted_to['force_used'] is True
    assert promoted_to['gate_report'] is not None
    assert any('mAP50' in f for f in promoted_to['gate_report']['failures'])


# =============================================================================
# Phase 9 — P1-10 a stamp_manifest_promotion failure must not be swallowed
# =============================================================================


def test_stamp_failure_is_not_swallowed(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the fix: `except Exception: logger.warning(...)` around the
    stamp call means a real write failure is invisible — the caller gets a
    normal 200 with no signal that lineage was never recorded.

    ``configure_logging()`` only runs at FastAPI startup (src/main.py), not
    in pytest context, so structlog's default wrapper never reaches
    stdlib logging — ``caplog`` would silently capture nothing here. Use
    structlog's own test capture instead (same pattern as
    ``tests/integration/test_request_id_propagation.py``).
    """
    from pathlib import Path

    from structlog.testing import capture_logs

    from src.services.training.jobs import TrainJobStatus

    job_id = 'stamp-fail-job'

    fake_status = TrainJobStatus(
        job_id=job_id,
        state='finished',
        checkpoint_path=f'/jobs/{job_id}/best.pt',
        eval={
            'map50': 0.91,
            'per_class': [
                {'class_id': 12, 'name': 'pickup', 'precision': 0.94, 'support': 1240},
            ],
        },
    )

    async def _fake_read_status(jid: str) -> TrainJobStatus | None:
        return fake_status if jid == job_id else None

    async def _fake_promote(**_kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        return PromoteResult(
            job_id=job_id,
            triton_name='yolo26m_stamp_fail',
            onnx_path=str(Path('/triton-models/yolo26m_stamp_fail/1/model.onnx')),
            config_path=str(Path('/triton-models/yolo26m_stamp_fail/config.pbtxt')),
            labels_path=str(Path('/triton-models/yolo26m_stamp_fail/labels.txt')),
            triton_loaded=True,
        )

    async def _fake_stamp(*_args: Any, **_kwargs: Any) -> bool:
        msg = 'disk exploded'
        raise OSError(msg)

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)
    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton', _fake_promote
    )
    monkeypatch.setattr('src.services.training.jobs.stamp_manifest_promotion', _fake_stamp)

    with capture_logs() as cap:
        r = app_client.post(
            f'/curation/train/promote/{job_id}',
            json={'triton_name': 'yolo26m_stamp_fail'},
        )

    # Promote itself succeeded (the model is already live in Triton by this
    # point) — we don't turn that into a 500, but the response must say
    # lineage was NOT recorded, and the failure must be logged loudly (at
    # error, not the previous silent warning-and-forget).
    assert r.status_code == 200, r.text
    assert r.json()['lineage_stamped'] is False
    stamp_failure_logs = [e for e in cap if e.get('event') == 'manifest_stamp_failed']
    assert stamp_failure_logs, cap
    assert stamp_failure_logs[0]['log_level'] == 'error'
    assert 'disk exploded' in stamp_failure_logs[0]['error']
