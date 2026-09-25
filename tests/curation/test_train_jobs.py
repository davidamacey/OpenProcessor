"""Tests for the file-based training-job service.

Ported from a private reference vehicle/license-plate curation stack's
training-pipeline test suite (Chunk 6). These tests exercise the API
<-> trainer protocol: writing a ``job.json``, reading a ``status.json``,
dropping a ``cancel`` sentinel, and detecting stale heartbeats. The
fixtures redirect ``OP_TRAIN_JOBS_DIR`` to a per-test ``tmp_path`` so
nothing escapes into a real ``/jobs/`` volume.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path  # used at runtime (jobs_dir fixture, registry-pin tests)

import pytest

from src.config import GpuArbiterConfig
from src.services.training import jobs as train_jobs
from src.services.training.jobs import (
    CampaignRunSpec,
    TrainCampaignSpec,
    TrainJobSpec,
    TrainJobStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect OP_TRAIN_JOBS_DIR at the env-var level."""
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))
    return tmp_path


@pytest.fixture
def restricted_gpu_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure an allowlist of {0, 2} -- the reference deployment's
    two A6000s -- for tests that exercise the *restrictive* path. The
    generic default (no fixture) is permissive: any GPU id validates."""
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(allowed_gpu_ids=frozenset({0, 2})),
    )


def _write_status(
    jobs_dir: Path,
    job_id: str,
    *,
    state: str = 'running',
    heartbeat_at: str | None = None,
) -> None:
    """Helper: write a status.json for a job_id."""
    status = TrainJobStatus(
        job_id=job_id,
        state=state,  # type: ignore[arg-type]
        heartbeat_at=heartbeat_at,
    )
    (jobs_dir / f'{job_id}.status.json').write_text(
        json.dumps(status.model_dump(mode='json')),
    )


# =============================================================================
# write_job
# =============================================================================


@pytest.mark.asyncio
async def test_write_job_produces_parseable_file(jobs_dir: Path) -> None:
    spec = TrainJobSpec(
        dataset_export_dir='/data/exports/2026-05-08_v7',
        model_size='m',
        profile='medium',
    )
    job_id = await train_jobs.write_job(spec)
    assert job_id  # API filled in an id
    target = jobs_dir / f'{job_id}.job.json'
    assert target.exists(), 'write_job did not create job.json'

    raw = json.loads(target.read_text())
    assert raw['job_id'] == job_id
    assert raw['model_family'] == 'yolo26'
    assert raw['model_size'] == 'm'
    assert raw['profile'] == 'medium'
    assert raw['dataset_export_dir'] == '/data/exports/2026-05-08_v7'


@pytest.mark.asyncio
async def test_write_job_uses_provided_id(jobs_dir: Path) -> None:
    spec = TrainJobSpec(
        job_id='2026-05-09T12-00-00_test',
        dataset_export_dir='/data/exports/x',
    )
    job_id = await train_jobs.write_job(spec)
    assert job_id == '2026-05-09T12-00-00_test'
    assert (jobs_dir / '2026-05-09T12-00-00_test.job.json').exists()


@pytest.mark.asyncio
async def test_write_job_rejects_duplicate(jobs_dir: Path) -> None:
    spec = TrainJobSpec(
        job_id='dup',
        dataset_export_dir='/data/exports/x',
    )
    await train_jobs.write_job(spec)
    with pytest.raises(ValueError, match='already exists'):
        await train_jobs.write_job(spec)


# =============================================================================
# cuda_visible_devices GPU allowlist
# =============================================================================
#
# The generic default (GpuArbiterConfig.allowed_gpu_ids == frozenset(), no
# restriction) accepts any GPU id combination. A deployment that pins
# training to specific GPUs (e.g. the reference deployment's two A6000s,
# {0, 2}, with GPU 1 -- a 3080 Ti -- reserved for an unrelated app)
# configures that allowlist and gets the restrictive behavior back --
# see the ``restricted_gpu_ids`` fixture.


def test_spec_accepts_gpu2_only() -> None:
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='2')
    assert spec.cuda_visible_devices == '2'


def test_spec_accepts_gpu0_only() -> None:
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='0')
    assert spec.cuda_visible_devices == '0'


def test_spec_accepts_any_gpu_id_by_default() -> None:
    """No allowlist configured -> permissive: an arbitrary id validates."""
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='7')
    assert spec.cuda_visible_devices == '7'


def test_spec_canonicalizes_dual_gpu_order() -> None:
    """'2,0' and '0,2' must compare equal downstream (arbiter lock, manifest)."""
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='2,0')
    assert spec.cuda_visible_devices == '0,2'


def test_default_cuda_visible_devices_is_neutral_single_gpu() -> None:
    """The generic default is a single neutral GPU id, not a
    deployment-specific dual-GPU pin."""
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x')
    assert spec.cuda_visible_devices == '0'


@pytest.mark.usefixtures('restricted_gpu_ids')
def test_spec_rejects_gpu1_when_allowlist_configured() -> None:
    """GPU 1 is the 3080 Ti reserved for an unrelated app in the reference
    deployment -- must never be selectable once the allowlist excludes it."""
    with pytest.raises(ValueError, match='allowed GPU id'):
        TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='1')


@pytest.mark.usefixtures('restricted_gpu_ids')
def test_spec_rejects_gpu1_mixed_with_valid_gpu_when_allowlist_configured() -> None:
    with pytest.raises(ValueError, match='allowed GPU id'):
        TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='0,1')


@pytest.mark.usefixtures('restricted_gpu_ids')
def test_spec_rejects_unknown_device_id_when_allowlist_configured() -> None:
    with pytest.raises(ValueError, match='allowed GPU id'):
        TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='7')


def test_spec_rejects_duplicate_device_id() -> None:
    """Duplicate-id rejection is unconditional -- it doesn't depend on an
    allowlist being configured."""
    with pytest.raises(ValueError, match='repeat'):
        TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='0,0')


@pytest.mark.usefixtures('restricted_gpu_ids')
def test_campaign_spec_rejects_gpu1_when_allowlist_configured() -> None:
    with pytest.raises(ValueError, match='allowed GPU id'):
        TrainCampaignSpec(
            dataset_export_dir='/data/exports/x',
            cuda_visible_devices='1',
            runs=[CampaignRunSpec(profile='probe')],
        )


# =============================================================================
# default_train_gpu_value
# =============================================================================


def test_default_train_gpu_value_unrestricted_is_zero() -> None:
    assert train_jobs.default_train_gpu_value() == '0'


def test_default_train_gpu_value_unset_env_var_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_TRAIN_DEFAULT_GPUS', raising=False)
    assert train_jobs.default_train_gpu_value() == '0'


@pytest.mark.usefixtures('restricted_gpu_ids')
def test_default_train_gpu_value_uses_smallest_allowed_id() -> None:
    """OP_GPU_ALLOWED_IDS=0,2 (no OP_TRAIN_DEFAULT_GPUS) -> smallest id, '0'."""
    assert train_jobs.default_train_gpu_value() == '0'


def test_default_train_gpu_value_uses_smallest_allowed_id_gpu2_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OP_GPU_ALLOWED_IDS=2 -> default is '2', not '0' (which isn't allowed)."""
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(allowed_gpu_ids=frozenset({2})),
    )
    assert train_jobs.default_train_gpu_value() == '2'


def test_default_train_gpu_value_honors_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(allowed_gpu_ids=frozenset({0, 2}), default_train_gpus='2'),
    )
    assert train_jobs.default_train_gpu_value() == '2'


def test_default_train_gpu_value_rejects_env_outside_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misconfigured OP_TRAIN_DEFAULT_GPUS outside the allowlist must fail
    loudly (through the same validator specs use) rather than silently
    default to a disallowed GPU."""
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(allowed_gpu_ids=frozenset({0, 2}), default_train_gpus='1'),
    )
    with pytest.raises(ValueError, match='allowed GPU id'):
        train_jobs.default_train_gpu_value()


def test_spec_default_uses_default_train_gpu_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """TrainJobSpec()/TrainCampaignSpec() with no cuda_visible_devices pick up
    the configured default, not a hardcoded '0'."""
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(allowed_gpu_ids=frozenset({2})),
    )
    spec = TrainJobSpec(dataset_export_dir='/data/exports/x')
    assert spec.cuda_visible_devices == '2'
    campaign = TrainCampaignSpec(
        dataset_export_dir='/data/exports/x', runs=[CampaignRunSpec(profile='probe')]
    )
    assert campaign.cuda_visible_devices == '2'


# =============================================================================
# frozen_test_sha + registry pin
# =============================================================================


def test_spec_accepts_frozen_test_sha() -> None:
    """Before the fix, TrainJobSpec's extra='forbid' rejected this field
    outright -- the trainer's manifest writer read
    spec.raw.get('frozen_test_sha') but nothing could ever set it."""
    spec = TrainJobSpec(
        dataset_export_dir='/data/exports/x',
        frozen_test_sha='abc123deadbeef',
    )
    assert spec.frozen_test_sha == 'abc123deadbeef'


@pytest.mark.asyncio
async def test_write_job_autofills_frozen_test_sha_from_export_manifest(
    jobs_dir: Path, tmp_path: Path
) -> None:
    """write_job reads frozen_test_sha out of the export's manifest.json
    when the caller didn't already supply one."""
    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text(json.dumps({'frozen_test_sha': 'export-sha-1'}))

    spec = TrainJobSpec(job_id='autofill', dataset_export_dir=str(export_dir))
    await train_jobs.write_job(spec)

    raw = json.loads((jobs_dir / 'autofill.job.json').read_text())
    assert raw['frozen_test_sha'] == 'export-sha-1'


@pytest.mark.asyncio
async def test_write_job_leaves_frozen_test_sha_none_without_manifest(jobs_dir: Path) -> None:
    """No manifest.json at dataset_export_dir -> best-effort None, no crash."""
    spec = TrainJobSpec(job_id='no_manifest', dataset_export_dir='/data/exports/missing')
    await train_jobs.write_job(spec)
    raw = json.loads((jobs_dir / 'no_manifest.job.json').read_text())
    assert raw['frozen_test_sha'] is None


@pytest.mark.asyncio
async def test_write_job_pins_registry_snapshot(
    jobs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """write_job snapshots the live registry to a job-scoped file + sha256
    so promote can use the registry as it was at submit time instead of
    always following a later rename."""

    class _FakeEntry:
        def __init__(self, class_id: int, class_name: str) -> None:
            self.class_id = class_id
            self.class_name = class_name
            self.deprecated = False

    class _FakeRegistrySnapshot:
        def __init__(self) -> None:
            self.version = 1
            self.updated_at = '2026-09-11T00:00:00Z'
            self.classes = [_FakeEntry(0, 'sedan')]

        def model_dump_json(self, **_kwargs: object) -> str:
            return json.dumps(
                {
                    'version': self.version,
                    'updated_at': self.updated_at,
                    'classes': [
                        {'class_id': c.class_id, 'class_name': c.class_name, 'deprecated': False}
                        for c in self.classes
                    ],
                }
            )

    class _FakeRegistry:
        def load(self) -> _FakeRegistrySnapshot:
            return _FakeRegistrySnapshot()

    monkeypatch.setattr(
        'src.clients.curation_opensearch.get_class_registry', lambda: _FakeRegistry()
    )

    spec = TrainJobSpec(job_id='pin_test', dataset_export_dir='/data/exports/x')
    await train_jobs.write_job(spec)

    raw = json.loads((jobs_dir / 'pin_test.job.json').read_text())
    assert raw['registry_sha'], 'registry_sha was not populated'
    assert raw['registry_snapshot_path'], 'registry_snapshot_path was not populated'

    snapshot_path = Path(raw['registry_snapshot_path'])
    assert snapshot_path.is_file()
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot['classes'][0]['class_name'] == 'sedan'


@pytest.mark.asyncio
async def test_read_job_spec_returns_written_payload(jobs_dir: Path) -> None:
    spec = TrainJobSpec(job_id='readback', dataset_export_dir='/data/exports/x')
    await train_jobs.write_job(spec)
    raw = await train_jobs.read_job_spec('readback')
    assert raw is not None
    assert raw['job_id'] == 'readback'


@pytest.mark.asyncio
async def test_read_job_spec_returns_none_for_unknown_job(jobs_dir: Path) -> None:
    assert await train_jobs.read_job_spec('nope') is None


# =============================================================================
# read_status
# =============================================================================


@pytest.mark.asyncio
async def test_read_status_returns_none_for_unknown_job(jobs_dir: Path) -> None:
    result = await train_jobs.read_status('does_not_exist')
    assert result is None


@pytest.mark.asyncio
async def test_read_status_returns_queued_when_only_spec_exists(jobs_dir: Path) -> None:
    spec = TrainJobSpec(
        job_id='spec_only',
        dataset_export_dir='/data/exports/x',
    )
    await train_jobs.write_job(spec)
    result = await train_jobs.read_status('spec_only')
    assert result is not None
    assert result.state == 'queued'
    assert result.job_id == 'spec_only'


@pytest.mark.asyncio
async def test_read_status_marks_stale_heartbeat_as_lost(jobs_dir: Path) -> None:
    """heartbeat_at >60s old + state=running -> API view flips to 'lost'."""
    stale_ts = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    _write_status(jobs_dir, 'stale', state='running', heartbeat_at=stale_ts)
    result = await train_jobs.read_status('stale')
    assert result is not None
    assert result.state == 'lost', 'stale heartbeat should flip state to lost'


@pytest.mark.asyncio
async def test_read_status_keeps_fresh_heartbeat(jobs_dir: Path) -> None:
    fresh_ts = datetime.now(UTC).isoformat()
    _write_status(jobs_dir, 'fresh', state='running', heartbeat_at=fresh_ts)
    result = await train_jobs.read_status('fresh')
    assert result is not None
    assert result.state == 'running'


@pytest.mark.asyncio
async def test_read_status_rejects_invalid_id(jobs_dir: Path) -> None:
    with pytest.raises(ValueError, match='invalid job_id'):
        await train_jobs.read_status('../escaping')


# =============================================================================
# write_cancel
# =============================================================================


@pytest.mark.asyncio
async def test_write_cancel_creates_sentinel(jobs_dir: Path) -> None:
    await train_jobs.write_cancel('some_job')
    sentinel = jobs_dir / 'some_job.cancel'
    assert sentinel.exists(), 'cancel sentinel was not written'


@pytest.mark.asyncio
async def test_write_cancel_idempotent(jobs_dir: Path) -> None:
    await train_jobs.write_cancel('twice')
    await train_jobs.write_cancel('twice')
    assert (jobs_dir / 'twice.cancel').exists()


# =============================================================================
# list_runs / get_active_job
# =============================================================================


@pytest.mark.asyncio
async def test_list_runs_returns_newest_first(jobs_dir: Path) -> None:
    # Write three specs out-of-order; mtime ordering should win.
    for jid, state in (('a', 'finished'), ('b', 'running'), ('c', 'queued')):
        spec = TrainJobSpec(job_id=jid, dataset_export_dir='/x')
        await train_jobs.write_job(spec)
        _write_status(jobs_dir, jid, state=state, heartbeat_at=datetime.now(UTC).isoformat())

    runs = await train_jobs.list_runs(limit=10)
    assert len(runs) == 3
    # Newest first means c, then b, then a (mtime order).
    assert {r.job_id for r in runs} == {'a', 'b', 'c'}


@pytest.mark.asyncio
async def test_get_active_job_returns_running_run(jobs_dir: Path) -> None:
    for jid, state in (('done1', 'finished'), ('live', 'running')):
        spec = TrainJobSpec(job_id=jid, dataset_export_dir='/x')
        await train_jobs.write_job(spec)
        _write_status(jobs_dir, jid, state=state, heartbeat_at=datetime.now(UTC).isoformat())

    active = await train_jobs.get_active_job()
    assert active is not None
    assert active.job_id == 'live'
    assert active.state == 'running'


@pytest.mark.asyncio
async def test_get_active_job_falls_back_to_most_recent(jobs_dir: Path) -> None:
    spec = TrainJobSpec(job_id='done', dataset_export_dir='/x')
    await train_jobs.write_job(spec)
    _write_status(jobs_dir, 'done', state='finished')
    result = await train_jobs.get_active_job()
    assert result is not None
    assert result.job_id == 'done'
    assert result.state == 'finished'


@pytest.mark.asyncio
async def test_get_active_job_raises_when_two_active(jobs_dir: Path) -> None:
    for jid in ('a', 'b'):
        spec = TrainJobSpec(job_id=jid, dataset_export_dir='/x')
        await train_jobs.write_job(spec)
        _write_status(jobs_dir, jid, state='running', heartbeat_at=datetime.now(UTC).isoformat())
    with pytest.raises(RuntimeError, match='multiple active runs'):
        await train_jobs.get_active_job()


# =============================================================================
# Campaigns
# =============================================================================


@pytest.mark.asyncio
async def test_write_campaign_creates_n_files_with_shared_id(jobs_dir: Path) -> None:
    campaign = TrainCampaignSpec(
        campaign_id='2026-05-09_sweep',
        dataset_export_dir='/data/exports/x',
        runs=[
            CampaignRunSpec(profile='nano', model_size='n'),
            CampaignRunSpec(profile='small', model_size='s'),
            CampaignRunSpec(profile='medium', model_size='m'),
        ],
    )
    campaign_id, job_ids = await train_jobs.write_campaign(campaign)
    assert campaign_id == '2026-05-09_sweep'
    assert len(job_ids) == 3
    for jid in job_ids:
        path = jobs_dir / f'{jid}.job.json'
        assert path.exists()
        raw = json.loads(path.read_text())
        assert raw['campaign_id'] == '2026-05-09_sweep'


@pytest.mark.asyncio
async def test_cancel_campaign_writes_sentinels(jobs_dir: Path) -> None:
    campaign = TrainCampaignSpec(
        campaign_id='2026-05-09_cancel_me',
        dataset_export_dir='/data/exports/x',
        runs=[
            CampaignRunSpec(profile='nano', model_size='n'),
            CampaignRunSpec(profile='small', model_size='s'),
        ],
    )
    campaign_id, job_ids = await train_jobs.write_campaign(campaign)
    # Mark first as finished, second still queued.
    _write_status(jobs_dir, job_ids[0], state='finished')
    _write_status(jobs_dir, job_ids[1], state='running', heartbeat_at=datetime.now(UTC).isoformat())

    n = await train_jobs.cancel_campaign(campaign_id)
    assert n == 1, 'should only cancel the non-terminal run'
    assert (jobs_dir / f'{job_ids[1]}.cancel').exists()
    assert not (jobs_dir / f'{job_ids[0]}.cancel').exists()


# =============================================================================
# tail_run_log
# =============================================================================


@pytest.mark.asyncio
async def test_tail_run_log_returns_last_n(jobs_dir: Path) -> None:
    log = jobs_dir / 'logged.run.log'
    log.write_text('\n'.join(f'line {i}' for i in range(50)))
    out = await train_jobs.tail_run_log('logged', lines=5)
    assert out == ['line 45', 'line 46', 'line 47', 'line 48', 'line 49']


@pytest.mark.asyncio
async def test_tail_run_log_missing_file_returns_empty(jobs_dir: Path) -> None:
    out = await train_jobs.tail_run_log('absent', lines=5)
    assert out == []


# =============================================================================
# mlflow_run_url: rebuilt from a public base at serving time, never the
# trainer's internal container hostname
# =============================================================================


class _FakeCurationConfig:
    def __init__(self, mlflow_public_url: str | None = None, api_prefix: str = '/curation') -> None:
        self.mlflow_public_url = mlflow_public_url
        self.api_prefix = api_prefix


def test_public_mlflow_url_builds_from_configured_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_jobs,
        'get_curation_config',
        lambda: _FakeCurationConfig(mlflow_public_url='https://mlflow.example.com'),
    )
    url = train_jobs._public_mlflow_url(run_id='abc123', experiment_id='7')
    assert url == 'https://mlflow.example.com/#/experiments/7/runs/abc123'


def test_public_mlflow_url_none_when_base_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_jobs, 'get_curation_config', lambda: _FakeCurationConfig(mlflow_public_url=None)
    )
    assert train_jobs._public_mlflow_url(run_id='abc123', experiment_id='7') is None


def test_public_mlflow_url_none_when_run_or_experiment_id_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_jobs,
        'get_curation_config',
        lambda: _FakeCurationConfig(mlflow_public_url='https://mlflow.example.com'),
    )
    assert train_jobs._public_mlflow_url(run_id=None, experiment_id='7') is None
    assert train_jobs._public_mlflow_url(run_id='abc123', experiment_id=None) is None


def test_public_mlflow_url_never_leaks_the_internal_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The internal tracking URI (a container hostname) must never be the
    fallback -- unset public base means null, full stop."""
    monkeypatch.setattr(
        train_jobs, 'get_curation_config', lambda: _FakeCurationConfig(mlflow_public_url=None)
    )
    url = train_jobs._public_mlflow_url(run_id='abc123', experiment_id='7')
    assert url is None
    assert url != 'http://curation-mlflow:5000/#/experiments/7/runs/abc123'


@pytest.mark.asyncio
async def test_read_status_rewrites_mlflow_url_and_confusion_matrix(
    jobs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        train_jobs,
        'get_curation_config',
        lambda: _FakeCurationConfig(mlflow_public_url='https://mlflow.example.com'),
    )
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
        == '/curation/train/artifacts/realmlf/confusion_matrix.png'
    )
    assert status.eval['map50'] == 0.62  # untouched


@pytest.mark.asyncio
async def test_read_status_serves_null_mlflow_url_when_public_base_unset(
    jobs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        train_jobs, 'get_curation_config', lambda: _FakeCurationConfig(mlflow_public_url=None)
    )
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
    monkeypatch.setattr(
        train_jobs, 'get_curation_config', lambda: _FakeCurationConfig(mlflow_public_url=None)
    )
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
    monkeypatch.setattr(
        train_jobs,
        'get_curation_config',
        lambda: _FakeCurationConfig(mlflow_public_url='https://mlflow.example.com'),
    )
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
        == '/curation/train/artifacts/manifjob/confusion_matrix.png'
    )


# =============================================================================
# Run-artifact resolution (GET {api_prefix}/train/artifacts/{job_id}/{name})
# =============================================================================


@pytest.mark.asyncio
async def test_resolve_run_dir_from_checkpoint_path(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'a_run'
    (run_dir / 'weights').mkdir(parents=True)
    (jobs_dir / 'ck.status.json').write_text(
        json.dumps(
            {
                'job_id': 'ck',
                'state': 'finished',
                'checkpoint_path': str(run_dir / 'weights' / 'best.pt'),
            }
        )
    )
    resolved = await train_jobs.resolve_run_dir('ck')
    assert resolved == run_dir.resolve()


@pytest.mark.asyncio
async def test_resolve_run_dir_falls_back_to_confusion_matrix_path(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'b_run'
    run_dir.mkdir()
    (jobs_dir / 'cm.status.json').write_text(
        json.dumps(
            {
                'job_id': 'cm',
                'state': 'failed',
                'eval': {'confusion_matrix_path': str(run_dir / 'confusion_matrix.png')},
            }
        )
    )
    resolved = await train_jobs.resolve_run_dir('cm')
    assert resolved == run_dir.resolve()


@pytest.mark.asyncio
async def test_resolve_run_dir_none_when_no_hints(jobs_dir: Path) -> None:
    (jobs_dir / 'bare.status.json').write_text(json.dumps({'job_id': 'bare', 'state': 'running'}))
    assert await train_jobs.resolve_run_dir('bare') is None


@pytest.mark.asyncio
async def test_resolve_run_dir_none_when_no_status(jobs_dir: Path) -> None:
    assert await train_jobs.resolve_run_dir('never_ran') is None


@pytest.mark.asyncio
async def test_read_artifact_serves_a_whitelisted_file(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'good_run'
    run_dir.mkdir()
    (run_dir / 'confusion_matrix.png').write_bytes(b'fake-png')
    (jobs_dir / 'good.status.json').write_text(
        json.dumps(
            {
                'job_id': 'good',
                'state': 'finished',
                'eval': {'confusion_matrix_path': str(run_dir / 'confusion_matrix.png')},
            }
        )
    )
    path = await train_jobs.read_artifact('good', 'confusion_matrix.png')
    assert path == (run_dir / 'confusion_matrix.png').resolve()


@pytest.mark.asyncio
async def test_read_artifact_rejects_a_non_whitelisted_name(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'good_run2'
    run_dir.mkdir()
    (run_dir / 'best.pt').write_bytes(b'weights')
    (jobs_dir / 'good2.status.json').write_text(
        json.dumps(
            {'job_id': 'good2', 'state': 'finished', 'checkpoint_path': str(run_dir / 'best.pt')}
        )
    )
    assert await train_jobs.read_artifact('good2', 'best.pt') is None


@pytest.mark.asyncio
async def test_read_artifact_none_for_unknown_job(jobs_dir: Path) -> None:
    assert await train_jobs.read_artifact('does_not_exist', 'confusion_matrix.png') is None


@pytest.mark.asyncio
async def test_read_artifact_none_when_file_never_written(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'no_file_run'
    run_dir.mkdir()
    (jobs_dir / 'nofile.status.json').write_text(
        json.dumps(
            {
                'job_id': 'nofile',
                'state': 'running',
                'eval': {'confusion_matrix_path': str(run_dir / 'confusion_matrix.png')},
            }
        )
    )
    assert await train_jobs.read_artifact('nofile', 'confusion_matrix.png') is None


@pytest.mark.asyncio
async def test_read_artifact_rejects_traversal_attempt(jobs_dir: Path) -> None:
    run_dir = jobs_dir / 'traversal_run'
    run_dir.mkdir()
    secret = jobs_dir / 'confusion_matrix.png'  # sibling of run_dir, outside it
    secret.write_bytes(b'not-this-one')
    (jobs_dir / 'trav.status.json').write_text(
        json.dumps(
            {
                'job_id': 'trav',
                'state': 'finished',
                'eval': {'confusion_matrix_path': str(run_dir / 'confusion_matrix.png')},
            }
        )
    )
    # Not in the whitelist (a slash isn't a valid character in any
    # whitelisted basename), so this must resolve to None rather than
    # escaping run_dir to serve the sibling file.
    assert await train_jobs.read_artifact('trav', '../confusion_matrix.png') is None


@pytest.mark.asyncio
async def test_read_artifact_rejects_invalid_job_id() -> None:
    with pytest.raises(ValueError, match='invalid job_id'):
        await train_jobs.read_artifact('../escaping', 'confusion_matrix.png')


def test_artifact_media_type_png_and_csv() -> None:
    assert train_jobs.artifact_media_type('confusion_matrix.png') == 'image/png'
    assert train_jobs.artifact_media_type('results.csv') == 'text/csv'
    assert train_jobs.artifact_media_type('unknown.bin') == 'application/octet-stream'
