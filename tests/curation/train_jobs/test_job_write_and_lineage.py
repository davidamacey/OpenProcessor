"""Writing and reading ``job.json``: validation, lineage stamping and the registry pin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import GpuArbiterConfig
from src.services.training import jobs as train_jobs
from src.services.training.job_models import TrainJobSpec


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


@pytest.mark.asyncio
async def test_write_job_refuses_a_missing_dataset_export_dir_clearly(jobs_dir: Path) -> None:
    """``dataset_export_dir`` is optional on the wire (defaults to the
    current export via _run_preflight), but write_job must still refuse
    clearly if it's still unset here -- e.g. a force=True /start call
    that bypassed a blocking preflight with no current export -- instead
    of an opaque TypeError from deep inside read_export_identity(None)."""
    spec = TrainJobSpec(model_size='m', profile='medium')
    assert spec.dataset_export_dir is None

    with pytest.raises(ValueError, match='dataset_export_dir'):
        await train_jobs.write_job(spec)


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
async def test_write_job_copies_lineage_from_a_full_export_manifest(
    jobs_dir: Path, tmp_path: Path
) -> None:
    """write_job copies dataset_sha / frozen_test_sha / test_label_sha /
    dataset_version_tag straight off a manifest that already has them (a
    fresh export) -- no on-disk recomputation needed."""
    export_dir = tmp_path / 'export'
    (export_dir / 'labels' / 'test').mkdir(parents=True)
    (export_dir / 'labels' / 'test' / 'a.txt').write_text('0 0.5 0.5 0.1 0.1\n')
    (export_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset_sha': 'dataset-sha-1',
                'frozen_test_sha': 'export-sha-1',
                'test_label_sha': 'label-sha-1',
                'version_tag': 'v1',
            }
        )
    )

    spec = TrainJobSpec(job_id='autofill', dataset_export_dir=str(export_dir))
    await train_jobs.write_job(spec)

    raw = json.loads((jobs_dir / 'autofill.job.json').read_text())
    assert raw['dataset_sha'] == 'dataset-sha-1'
    assert raw['frozen_test_sha'] == 'export-sha-1'
    assert raw['test_label_sha'] == 'label-sha-1'
    assert raw['dataset_version_tag'] == 'v1'


@pytest.mark.asyncio
async def test_write_job_computes_test_hashes_for_an_older_export_manifest(
    jobs_dir: Path, tmp_path: Path
) -> None:
    """An export written before this field existed has no frozen_test_sha/test_label_sha in
    its manifest -- write_job must compute both from the on-disk test
    split rather than leaving them None, so every *new* run still records
    them even against an old export. dataset_sha is never computed (it's
    the export's own claim) and stays None when the manifest lacks it."""
    from src.services.curation.export_support import frozen_test_sha_of, label_content_sha

    export_dir = tmp_path / 'export'
    (export_dir / 'labels' / 'test').mkdir(parents=True)
    (export_dir / 'labels' / 'test' / 'a.txt').write_text('0 0.5 0.5 0.1 0.1\n')
    (export_dir / 'manifest.json').write_text(json.dumps({}))

    spec = TrainJobSpec(job_id='backfill', dataset_export_dir=str(export_dir))
    await train_jobs.write_job(spec)

    raw = json.loads((jobs_dir / 'backfill.job.json').read_text())
    assert raw['dataset_sha'] is None
    assert raw['frozen_test_sha'] == frozen_test_sha_of(export_dir)
    assert raw['test_label_sha'] == label_content_sha(export_dir, None, truncate=16, split='test')
    assert raw['frozen_test_sha']
    assert raw['test_label_sha']


@pytest.mark.asyncio
async def test_write_job_leaves_frozen_test_sha_none_without_manifest(jobs_dir: Path) -> None:
    """No manifest.json at dataset_export_dir -> best-effort None, no crash."""
    spec = TrainJobSpec(job_id='no_manifest', dataset_export_dir='/data/exports/missing')
    await train_jobs.write_job(spec)
    raw = json.loads((jobs_dir / 'no_manifest.job.json').read_text())
    assert raw['dataset_sha'] is None
    assert raw['frozen_test_sha'] is None
    assert raw['test_label_sha'] is None


@pytest.mark.asyncio
async def test_write_job_stamps_api_sha_from_op_build_sha(
    jobs_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """api_sha is stamped from OP_BUILD_SHA (or a live git rev-parse in a
    dev checkout) at submit time -- see
    src.services.curation.export_support._code_sha."""
    monkeypatch.setenv('OP_BUILD_SHA', 'apisha123')
    spec = TrainJobSpec(job_id='stamp_api', dataset_export_dir=str(tmp_path / 'missing'))
    await train_jobs.write_job(spec)
    raw = json.loads((jobs_dir / 'stamp_api.job.json').read_text())
    assert raw['api_sha'] == 'apisha123'


@pytest.mark.asyncio
async def test_write_job_stamps_trainer_image_id_from_docker_client(
    jobs_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """trainer_image_id / trainer_image_revision come from the docker
    socket when GpuArbiterConfig names a trainer container and the client
    is reachable (src.services.training.lineage.stamp_code_versions)."""
    from src.services.training import lineage

    class _FakeImage:
        id = 'sha256:abc'
        labels = {'org.opencontainers.image.revision': 'trainer-rev-1'}

    class _FakeContainer:
        image = _FakeImage()

    class _FakeContainers:
        def get(self, _name: str) -> _FakeContainer:
            return _FakeContainer()

    class _FakeDockerClient:
        containers = _FakeContainers()

    monkeypatch.setattr(
        lineage,
        'get_gpu_arbiter_config',
        lambda: GpuArbiterConfig(trainer_container='curation-trainer'),
    )
    monkeypatch.setattr(lineage, '_docker_client', _FakeDockerClient)

    spec = TrainJobSpec(job_id='stamp_trainer', dataset_export_dir=str(tmp_path / 'missing'))
    await train_jobs.write_job(spec)
    raw = json.loads((jobs_dir / 'stamp_trainer.job.json').read_text())
    assert raw['trainer_image_id'] == 'sha256:abc'
    assert raw['trainer_image_revision'] == 'trainer-rev-1'


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
        'src.clients.curation_opensearch.registry.get_class_registry', _FakeRegistry
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


def test_hyperparameters_reject_top_level_spec_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match='include_classes'):
        TrainJobSpec(
            dataset_export_dir='/data/exports/x',
            hyperparameters={'include_classes': [1], 'epochs': 3},
        )
    TrainJobSpec(dataset_export_dir='/data/exports/x', hyperparameters={'epochs': 3})
