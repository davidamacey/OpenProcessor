"""``run_training`` against a stub Ultralytics: finish, class remap, cancel and failure."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import job_spec
import job_status
import pytest
import trainer
import trainer_callbacks
import trainer_kwargs

from curation.trainer_protocol._helpers import make_spec
from src.services.training import lineage
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
        # A YOLO26 detection head: the one-to-one branch is what the export reads.
        self.model = SimpleNamespace(model=[SimpleNamespace(one2one_cv2=[])])

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
    spec = make_spec(dataset_export_dir=str(export_dir), model_size='n', profile='probe')
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
        """A genuine dual-head build: the one-to-one branch (``one2one_cv2``) only exists
        once end2end training actually created it -- same as Ultralytics' real head."""

        def __init__(self) -> None:
            self.end2end = False
            self.one2one_cv2: list[Any] = []

        @property
        def one2one(self) -> dict[str, Any]:
            return {'box_head': self.one2one_cv2, 'cls_head': None}

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
    spec = make_spec(dataset_export_dir=str(export_dir), model_size='n', profile='probe')
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
        end2end = False  # no one-to-one branch at all -- a plain single head

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


def _run(spec: Any) -> tuple[job_status.StatusState, dict[str, Any] | None]:
    """``run_training`` as the handler drives it, then the manifest it stores."""
    from dataset_prep import read_class_remap
    from job_manifest import build_manifest

    state = job_status.StatusState(job_id=spec.job_id, state='starting')
    try:
        trainer.run_training(spec, state)
        return state, build_manifest(spec, state, class_remap=read_class_remap(spec))
    finally:
        shutil.rmtree(spec.tmp_root, ignore_errors=True)


@pytest.mark.integration
def test_a_whole_export_run_reaches_finished(
    jobs_dir: Path, export_dir: Path, stub_ultralytics: type[_StubYOLO], tmp_path: Path
) -> None:
    spec = make_spec(
        dataset_export_dir=str(export_dir),
        model_size='n',
        profile='probe',
        hyperparameters={'epochs': 1, 'batch': 2, 'optimizer': 'MuSGD', 'seed': 7},
    )

    state, manifest = _run(spec)

    assert state.state == 'finished', state.error
    assert state.checkpoint_path is not None
    assert state.checkpoint_path.endswith('weights/best.pt')
    # The stub's val() returns None (no test-split pass), so eval falls
    # back to the training-time validation numbers -- split must say 'val',
    # never silently look like a test-split result.
    assert state.eval == {
        'map50': 0.77,
        'map50_95': 0.44,
        'split': 'val',
        'val_last': {'map50': 0.77, 'map50_95': 0.44},
        'head': 'end2end',  # the stub is a YOLO26 head: the served one-to-one branch was selected
    }
    assert state.class_remap_copy_failed is False
    # ONNX sibling is what promote actually copies into the Triton repo.
    assert Path(state.checkpoint_path).with_suffix('.onnx').is_file()

    assert manifest is not None
    # The export fixture's manifest.json only sets frozen_test_sha -- no
    # dataset_sha key, and dataset_sha is never computed (it's the export's
    # own claim), so it stays None.
    assert manifest['lineage']['dataset_sha'] is None
    assert manifest['lineage']['frozen_test_sha'] == 'sha-frozen-test'
    # Whole-export runs carry a class_remap too (the promote-labels fix): the export fixture's
    # class_registry.json maps registry ids 5/9/12 -> dense ids 0/1/2, so a full-class run's
    # manifest must capture that translation, not None.
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
        job_id=spec.job_id, checkpoint_path=Path(state.checkpoint_path), manifest=manifest
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
def test_a_normal_run_manifest_has_no_null_lineage(
    jobs_dir: Path,
    export_dir: Path,
    stub_ultralytics: type[_StubYOLO],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run started against a full export manifest, with a build sha on both sides, must not
    leave any lineage/code_versions field null -- a null here silently degrades a run's
    reproducibility envelope with no visible signal."""
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
    identity = lineage.read_export_identity(export_dir)
    stored = {
        'dataset_sha': identity.dataset_sha,
        'frozen_test_sha': identity.frozen_test_sha,
        'test_label_sha': identity.test_label_sha,
        'api_sha': lineage.api_sha(),
        'registry_sha': 'registry-sha-x',
    }
    spec = make_spec(
        dataset_export_dir=str(export_dir),
        model_size='n',
        profile='probe',
        hyperparameters={'epochs': 1, 'batch': 2, 'optimizer': 'MuSGD', 'seed': 3},
        **stored,
    )

    _state, manifest = _run(spec)

    assert manifest is not None
    for key in ('dataset_sha', 'frozen_test_sha', 'test_label_sha', 'registry_sha'):
        assert manifest['lineage'][key] is not None, key
    for key in ('api_sha', 'trainer_sha'):
        assert manifest['code_versions'][key] is not None, key


@pytest.mark.integration
def test_a_subset_run_records_its_class_remap_in_both_promote_sources(
    jobs_dir: Path, export_dir: Path, stub_ultralytics: type[_StubYOLO]
) -> None:
    """Both promote-side sources must be populated by a finished subset run."""
    spec = make_spec(
        dataset_export_dir=str(export_dir),
        model_size='n',
        include_classes=[12, 5],
        hyperparameters={'epochs': 1, 'optimizer': 'MuSGD'},
    )

    state, manifest = _run(spec)

    assert state.state == 'finished', state.error
    assert state.class_remap_copy_failed is False
    checkpoint = Path(state.checkpoint_path or '')
    assert manifest is not None

    # (a) manifest lineage -- survives tmp cleanup.
    from_manifest = resolve_class_remap(
        job_id=spec.job_id, checkpoint_path=checkpoint, manifest=manifest
    )
    assert from_manifest.source == 'manifest'
    assert from_manifest.mapping == {12: 0, 5: 1}

    # (b) weights dir -- the second, on-disk source.
    from_weights = resolve_class_remap(
        job_id=spec.job_id, checkpoint_path=checkpoint, manifest=None
    )
    assert from_weights.source == 'weights_dir'
    assert from_weights.mapping == {12: 0, 5: 1}


@pytest.mark.integration
def test_a_cancel_raised_out_of_the_epoch_callback_ends_the_run_cancelled(
    jobs_dir: Path, export_dir: Path, stub_ultralytics: type[_StubYOLO]
) -> None:
    spec = make_spec(
        dataset_export_dir=str(export_dir), hyperparameters={'epochs': 5, 'optimizer': 'MuSGD'}
    )
    # The real cancel path raises out of the on_train_epoch_end callback.
    stub_ultralytics.raise_on_train = trainer_callbacks.CancelRequestedError('cancelled')

    state, _manifest = _run(spec)

    assert state.state == 'cancelled'
    assert state.error == 'cancelled by user'
    assert not spec.tmp_root.exists()


@pytest.mark.integration
def test_the_epoch_callback_raises_once_a_cancel_is_requested(
    jobs_dir: Path, export_dir: Path
) -> None:
    spec = make_spec(dataset_export_dir=str(export_dir))
    state = job_status.StatusState(job_id=spec.job_id)
    _start, end_of_epoch, _fit = trainer_callbacks._make_ultralytics_callbacks(spec, state, 5)

    end_of_epoch(None)  # not cancelled: the epoch ends normally
    state.cancel_requested.set()
    with pytest.raises(trainer_callbacks.CancelRequestedError):
        end_of_epoch(None)


@pytest.mark.integration
def test_a_training_failure_propagates_and_its_manifest_says_failed(
    jobs_dir: Path, export_dir: Path, stub_ultralytics: type[_StubYOLO]
) -> None:
    """A non-OOM error must surface to the handler (which fails the job), not retry-loop and
    not vanish."""
    spec = make_spec(
        dataset_export_dir=str(export_dir), hyperparameters={'epochs': 1, 'optimizer': 'MuSGD'}
    )
    stub_ultralytics.raise_on_train = RuntimeError('dataset is corrupt')
    state = job_status.StatusState(job_id=spec.job_id, state='starting')

    with pytest.raises(RuntimeError, match='dataset is corrupt'):
        trainer.run_training(spec, state)
    state.state = 'failed'  # what the handler records for an exception
    from job_manifest import build_manifest

    manifest = build_manifest(spec, state)
    shutil.rmtree(spec.tmp_root, ignore_errors=True)

    assert manifest is not None
    assert manifest['results']['final_state'] == 'failed'
