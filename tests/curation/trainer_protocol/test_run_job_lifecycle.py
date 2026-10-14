"""End-to-end ``run_job`` against a stub Ultralytics, cancel, failure and the watcher."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import job_spec
import job_status
import pytest
import trainer
import trainer_callbacks
import trainer_kwargs

from curation.trainer_protocol._helpers import _write_job
from src.services.training import jobs as train_jobs
from src.services.training.class_remap import build_class_id_to_name, resolve_class_remap


class _StubTrainer:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir


class _StubYOLO:
    """Minimal stand-in for ``ultralytics.YOLO``.

    ``train()`` writes exactly the artifacts the trainer looks for afterwards:
    ``weights/best.pt`` and ``results.csv``.
    """

    runs_root: Path
    raise_on_train: Exception | None = None

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.callbacks: dict[str, list[Any]] = {}
        self.trainer: _StubTrainer | None = None

    def add_callback(self, event: str, fn: Any) -> None:
        self.callbacks.setdefault(event, []).append(fn)

    def train(self, **kwargs: Any) -> None:
        if _StubYOLO.raise_on_train is not None:
            raise _StubYOLO.raise_on_train
        save_dir = Path(kwargs['project']) / kwargs['name']
        (save_dir / 'weights').mkdir(parents=True, exist_ok=True)
        (save_dir / 'weights' / 'best.pt').write_bytes(b'stub-checkpoint')
        (save_dir / 'results.csv').write_text(
            'epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.77,0.44\n'
        )
        self.trainer = _StubTrainer(save_dir)

    def export(self, **_kwargs: Any) -> None:
        Path(self.weights).with_suffix('.onnx').write_bytes(b'stub-onnx')

    def val(self, **_kwargs: Any) -> None:
        return None


@pytest.fixture
def stub_ultralytics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[_StubYOLO]:
    """Install a stub ``ultralytics`` module and point the runs root at tmp."""
    import types

    _StubYOLO.raise_on_train = None
    module = types.ModuleType('ultralytics')
    module.YOLO = _StubYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'ultralytics', module)
    monkeypatch.setattr(trainer_kwargs, 'RUNS_ROOT', str(tmp_path / 'runs'))
    monkeypatch.setattr(job_spec, 'TMP_ROOT', tmp_path / 'tmp')
    # A local file-backed tracking store, so the real MLflow code path runs
    # (and is therefore covered) without reaching for a network service that
    # isn't there.
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'file://{tmp_path / "mlruns"}')
    return _StubYOLO


def test_finalize_run_forces_the_end2end_head_for_yolo26_test_split_eval(
    jobs_dir: Path, export_dir: Path, tmp_path: Path
) -> None:
    """The trainer's own post-training test-split re-validation must score
    the served (NMS-free, one-to-one) head -- not whichever head
    Ultralytics' .val() defaults to for the reloaded checkpoint -- or the
    comparison metric silently doesn't match what Triton actually serves.
    ``.val()`` has no ``end2end=`` kwarg; the real toggle is the loaded
    model's own ``.end2end`` property."""
    job_id = _write_job(dataset_export_dir=str(export_dir), model_size='n', profile='probe')
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    state = job_status.StatusState(
        job_id=spec.job_id, campaign_id=spec.campaign_id, state='running'
    )

    save_dir = tmp_path / 'run'
    (save_dir / 'weights').mkdir(parents=True)
    best_pt = save_dir / 'weights' / 'best.pt'
    best_pt.write_bytes(b'stub-checkpoint')
    (save_dir / 'results.csv').write_text('epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.3\n')
    data_yaml_path = export_dir / 'data.yaml'

    class _Head:
        """A genuine dual-head build: one2one only exists once end2end
        training actually created it -- same as Ultralytics' real head."""

        def __init__(self) -> None:
            self.end2end = False

        @property
        def one2one(self) -> dict[str, Any]:
            return {'box_head': None, 'cls_head': None}

    class _InnerModel:
        def __init__(self) -> None:
            self._head = _Head()
            self.model = [self._head]  # DetectionModel.model[-1] is the head

        @property
        def end2end(self) -> bool:
            return self._head.end2end

        @end2end.setter
        def end2end(self, value: bool) -> None:
            self._head.end2end = value

    class _RecordingYOLO:
        observed_end2end: bool | None = None

        def __init__(self, weights: str) -> None:
            self.weights = weights
            self.model = _InnerModel()
            self.trainer: _StubTrainer | None = None

        def export(self, **_kwargs: Any) -> None:
            pass  # not under test here

        def val(self, **_kwargs: Any) -> str:
            _RecordingYOLO.observed_end2end = self.model.end2end
            return 'val-result-sentinel'

    model = _RecordingYOLO(str(best_pt))
    model.trainer = _StubTrainer(save_dir)

    trainer._finalize_run(spec, state, model, data_yaml_path)

    assert _RecordingYOLO.observed_end2end is True
    assert state.eval is not None
    assert state.eval['head'] == 'end2end'


def test_finalize_run_does_not_force_end2end_on_a_non_dual_head_model(
    jobs_dir: Path, export_dir: Path, tmp_path: Path
) -> None:
    """A model with no one2one branch (not end2end-capable) must be left
    alone -- forcing end2end=True on it would break inference, since it
    has no one-to-one head to switch to."""
    job_id = _write_job(dataset_export_dir=str(export_dir), model_size='n', profile='probe')
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    state = job_status.StatusState(
        job_id=spec.job_id, campaign_id=spec.campaign_id, state='running'
    )

    save_dir = tmp_path / 'run'
    (save_dir / 'weights').mkdir(parents=True)
    best_pt = save_dir / 'weights' / 'best.pt'
    best_pt.write_bytes(b'stub-checkpoint')
    (save_dir / 'results.csv').write_text('epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.3\n')
    data_yaml_path = export_dir / 'data.yaml'

    class _NonEndToEndHead:
        end2end = False  # no one2one property at all -- a plain single head

    class _InnerModel:
        def __init__(self) -> None:
            self.model = [_NonEndToEndHead()]
            self.end2end = False

    class _RecordingYOLO:
        def __init__(self, weights: str) -> None:
            self.weights = weights
            self.model = _InnerModel()
            self.trainer: _StubTrainer | None = None

        def export(self, **_kwargs: Any) -> None:
            pass

        def val(self, **_kwargs: Any) -> str:
            return 'val-result-sentinel'

    model = _RecordingYOLO(str(best_pt))
    model.trainer = _StubTrainer(save_dir)

    trainer._finalize_run(spec, state, model, data_yaml_path)

    assert model.model.end2end is False  # left untouched
    assert state.eval is not None
    assert 'head' not in state.eval


@pytest.mark.integration
def test_run_job_drives_a_whole_export_run_to_finished(
    jobs_dir: Path, export_dir: Path, stub_ultralytics: type[_StubYOLO], tmp_path: Path
) -> None:
    job_id = _write_job(
        dataset_export_dir=str(export_dir),
        model_size='n',
        profile='probe',
        hyperparameters={'epochs': 1, 'batch': 2, 'optimizer': 'MuSGD', 'seed': 7},
    )
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    trainer.run_job(spec)

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None
    assert status.state == 'finished', status.error
    assert status.checkpoint_path is not None
    assert status.checkpoint_path.endswith('weights/best.pt')
    # The stub's val() returns None (no test-split pass), so eval falls
    # back to the training-time validation numbers -- split must say 'val',
    # never silently look like a test-split result.
    assert status.eval == {
        'map50': 0.77,
        'map50_95': 0.44,
        'split': 'val',
        'val_last': {'map50': 0.77, 'map50_95': 0.44},
        'confusion_matrix_url': None,
    }
    assert status.class_remap_copy_failed is False  # type: ignore[attr-defined]
    # ONNX sibling is what promote actually copies into the Triton repo.
    assert Path(status.checkpoint_path).with_suffix('.onnx').is_file()

    manifest = asyncio.run(train_jobs.read_manifest(job_id))
    assert manifest is not None
    # The export fixture's manifest.json only sets frozen_test_sha -- no
    # dataset_sha key, and dataset_sha is never computed (it's the export's
    # own claim), so it stays None.
    assert manifest['lineage']['dataset_sha'] is None
    assert manifest['lineage']['frozen_test_sha'] == 'sha-frozen-test'
    # Whole-export runs now carry a class_remap too (the promote-labels
    # fix): the export fixture's class_registry.json maps registry ids
    # 5/9/12 -> dense ids 0/1/2, so a full-class run's manifest must
    # capture that translation, not None.
    assert manifest['lineage']['class_remap'] is not None
    assert manifest['lineage']['class_remap']['original_to_new'] == {'5': 0, '9': 1, '12': 2}
    assert manifest['lineage']['class_remap']['names'] == ['widget', 'serial_plate', 'gadget']
    assert manifest['lineage']['training_seed'] == 7
    assert manifest['lineage']['deterministic'] is True
    assert manifest['results']['final_state'] == 'finished'
    assert manifest['results']['checkpoint_sha256']
    assert manifest['promoted_to'] is None

    # A full-class run's remap resolves from the manifest, and labels.txt
    # for the promoted model must be keyed by the registry id each dense
    # id actually maps to -- not identity.
    remap = resolve_class_remap(
        job_id=job_id, checkpoint_path=Path(status.checkpoint_path), manifest=manifest
    )
    assert remap.source == 'manifest'
    full_registry = {5: 'widget', 9: 'serial_plate', 12: 'gadget'}
    assert build_class_id_to_name(remap=remap, full_registry=full_registry) == {
        0: 'widget',
        1: 'serial_plate',
        2: 'gadget',
    }
    # tmp scratch is always cleaned.
    assert not spec.tmp_root.exists()


@pytest.mark.integration
def test_normal_run_manifest_has_no_null_lineage(
    jobs_dir: Path,
    export_dir: Path,
    stub_ultralytics: type[_StubYOLO],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run submitted against a full export manifest, with a build sha and
    a reachable docker client, must not leave any lineage/code_versions
    field null -- a null here silently degrades a run's reproducibility
    envelope with no visible signal. ``trainer_image_id`` is the one field
    that legitimately stays null when the API has no docker socket; here
    the socket is faked reachable so it, too, must be non-null."""
    from src.config import GpuArbiterConfig
    from src.services.training import lineage

    (export_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset_sha': 'dataset-sha-x',
                'frozen_test_sha': 'sha-frozen-test',
                'test_label_sha': 'label-sha-x',
                'version_tag': 'v9',
            }
        )
    )
    monkeypatch.setenv('OP_BUILD_SHA', 'apisha-x')

    class _FakeImage:
        id = 'sha256:trainerimg'
        labels = {'org.opencontainers.image.revision': 'trainerrev-x'}

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

    job_id = _write_job(
        dataset_export_dir=str(export_dir),
        model_size='n',
        profile='probe',
        hyperparameters={'epochs': 1, 'batch': 2, 'optimizer': 'MuSGD', 'seed': 3},
    )
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    trainer.run_job(spec)

    manifest = asyncio.run(train_jobs.read_manifest(job_id))
    assert manifest is not None
    for key in ('dataset_sha', 'frozen_test_sha', 'test_label_sha', 'registry_sha'):
        assert manifest['lineage'][key] is not None, key
    for key in ('api_sha', 'trainer_sha', 'trainer_image_id'):
        assert manifest['code_versions'][key] is not None, key


@pytest.mark.integration
def test_run_job_propagates_class_remap_for_a_subset_run(
    jobs_dir: Path, export_dir: Path, stub_ultralytics: type[_StubYOLO]
) -> None:
    """Both promote-side sources must be populated by a finished subset run."""
    job_id = _write_job(
        dataset_export_dir=str(export_dir),
        model_size='n',
        include_classes=[12, 5],
        hyperparameters={'epochs': 1, 'optimizer': 'MuSGD'},
    )
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    trainer.run_job(spec)

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None
    assert status.state == 'finished', status.error
    assert status.class_remap_copy_failed is False  # type: ignore[attr-defined]
    checkpoint = Path(status.checkpoint_path or '')
    manifest = asyncio.run(train_jobs.read_manifest(job_id))
    assert manifest is not None

    # (a) manifest lineage -- survives tmp cleanup.
    from_manifest = resolve_class_remap(
        job_id=job_id, checkpoint_path=checkpoint, manifest=manifest
    )
    assert from_manifest.source == 'manifest'
    assert from_manifest.mapping == {12: 0, 5: 1}

    # (b) weights dir -- the second, on-disk source.
    from_weights = resolve_class_remap(job_id=job_id, checkpoint_path=checkpoint, manifest=None)
    assert from_weights.source == 'weights_dir'
    assert from_weights.mapping == {12: 0, 5: 1}


@pytest.mark.integration
def test_run_job_honors_the_cancel_sentinel(
    jobs_dir: Path, export_dir: Path, stub_ultralytics: type[_StubYOLO]
) -> None:
    job_id = _write_job(
        dataset_export_dir=str(export_dir), hyperparameters={'epochs': 5, 'optimizer': 'MuSGD'}
    )
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    asyncio.run(train_jobs.write_cancel(job_id))
    # The real cancel path raises out of the on_train_epoch_end callback.
    stub_ultralytics.raise_on_train = trainer_callbacks.CancelRequestedError('cancelled')

    trainer.run_job(spec)

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None
    assert status.state == 'cancelled'
    assert status.error == 'cancelled by user'
    assert not spec.tmp_root.exists()


@pytest.mark.integration
def test_run_job_records_a_training_failure_in_status_and_log(
    jobs_dir: Path, export_dir: Path, stub_ultralytics: type[_StubYOLO]
) -> None:
    """A non-OOM error must surface, not retry-loop and not vanish."""
    job_id = _write_job(
        dataset_export_dir=str(export_dir), hyperparameters={'epochs': 1, 'optimizer': 'MuSGD'}
    )
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    stub_ultralytics.raise_on_train = RuntimeError('dataset is corrupt')

    trainer.run_job(spec)

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None
    assert status.state == 'failed'
    assert 'dataset is corrupt' in (status.error or '')
    assert any('FAILED' in line for line in asyncio.run(train_jobs.tail_run_log(job_id, 200)))

    manifest = asyncio.run(train_jobs.read_manifest(job_id))
    assert manifest is not None
    assert manifest['results']['final_state'] == 'failed'


@pytest.mark.integration
def test_watcher_processes_the_oldest_pending_job(
    jobs_dir: Path, export_dir: Path, stub_ultralytics: type[_StubYOLO]
) -> None:
    job_id = _write_job(
        dataset_export_dir=str(export_dir), hyperparameters={'epochs': 1, 'optimizer': 'MuSGD'}
    )

    assert trainer.process_one(jobs_dir) is True
    assert trainer.process_one(jobs_dir) is False  # nothing left pending

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None
    assert status.state == 'finished', status.error
