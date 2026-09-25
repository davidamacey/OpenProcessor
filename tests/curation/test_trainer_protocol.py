"""Conformance tests for the trainer container against the API's job protocol.

``docker/trainer/`` ships as its own image, so nothing in ``src/`` imports it
and a drift between the two halves of the file protocol would otherwise go
unnoticed until a real training run failed. These tests drive **both** halves
against one tmpdir: the API writer (:mod:`src.services.training.jobs`) produces
the files, the trainer parses them, the trainer produces status/manifest files,
and the API reader parses those back.

The end-to-end ``run_job`` tests inject a stub ``ultralytics`` module, so they
exercise the trainer's real orchestration (status transitions, checkpoint
discovery, class_remap propagation, manifest, tmp cleanup) with no GPU and no
model download. A genuine Ultralytics training run is covered separately by
``test_trainer_smoke.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from src.services.training import jobs as train_jobs
from src.services.training.jobs import TrainJobSpec
from src.services.training.triton_promote import build_class_id_to_name, resolve_class_remap


TRAINER_DIR = Path(__file__).resolve().parents[2] / 'docker' / 'trainer'
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

import campaign  # noqa: E402
import dataset_prep  # noqa: E402
import incumbent_compare  # noqa: E402
import job_protocol  # noqa: E402
import subset_dataset  # noqa: E402
import trainer  # noqa: E402


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A shared /jobs volume both halves of the protocol point at."""
    d = tmp_path / 'jobs'
    d.mkdir()
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(d))
    return d


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    """A minimal frozen export: 3 classes, one labeled image per split."""
    root = tmp_path / 'export'
    for split in ('train', 'val', 'test'):
        (root / 'images' / split).mkdir(parents=True)
        (root / 'labels' / split).mkdir(parents=True)
        (root / 'images' / split / f'{split}0.jpg').write_bytes(b'not-a-real-jpeg')
        # Dense export ids 0/1/2 -> registry ids 5/9/12 (see export_id_map).
        (root / 'labels' / split / f'{split}0.txt').write_text(
            '0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n2 0.75 0.75 0.1 0.1\n'
        )
    (root / 'data.yaml').write_text(
        'path: .\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        'nc: 3\n'
        "names: ['widget', 'serial_plate', 'gadget']\n"
    )
    # A registry id space that is deliberately NOT the dense export id space:
    # this is exactly the translation subset_dataset must not skip.
    (root / 'class_registry.json').write_text(
        json.dumps(
            {
                'classes': [
                    {'class_id': 5, 'class_name': 'widget'},
                    {'class_id': 9, 'class_name': 'serial_plate'},
                    {'class_id': 12, 'class_name': 'gadget'},
                ],
                'export_id_map': {'5': 0, '9': 1, '12': 2},
            }
        )
    )
    (root / 'manifest.json').write_text(json.dumps({'frozen_test_sha': 'sha-frozen-test'}))
    return root


def _write_job(**overrides: Any) -> str:
    """Submit a job through the *API's* writer and return the job id."""
    spec = TrainJobSpec(**overrides)
    return asyncio.run(train_jobs.write_job(spec))


# =============================================================================
# job.json: API writer -> trainer parser
# =============================================================================


def test_api_written_job_json_parses_in_trainer(jobs_dir: Path, export_dir: Path) -> None:
    """The exact bytes ``POST /train/start`` writes must parse in the trainer."""
    job_id = _write_job(
        dataset_export_dir=str(export_dir),
        model_size='n',
        profile='probe',
        include_classes=[5, 12],
        single_cls=False,
        cuda_visible_devices='0',
        hyperparameters={'epochs': 3, 'batch': 4, 'optimizer': 'MuSGD'},
    )

    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    assert spec.job_id == job_id
    assert spec.dataset_export_dir == export_dir
    assert spec.include_classes == [5, 12]
    assert spec.model_family == 'yolo26'
    assert spec.model_size == 'n'
    assert spec.profile == 'probe'
    assert spec.hyperparameters['epochs'] == 3
    # Lineage the API auto-fills at submit time and the manifest re-publishes.
    assert spec.raw['frozen_test_sha'] == 'sha-frozen-test'


def test_trainer_sibling_paths_match_the_api_side(jobs_dir: Path, export_dir: Path) -> None:
    """status / cancel / log / manifest paths are a shared naming contract."""
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    assert spec.status_path == train_jobs._status_path(job_id)
    assert spec.cancel_path == train_jobs._cancel_path(job_id)
    assert spec.run_log_path == train_jobs._log_path(job_id)
    assert spec.manifest_path == train_jobs._manifest_path(job_id)


def test_api_cancel_sentinel_is_seen_by_the_trainer(jobs_dir: Path, export_dir: Path) -> None:
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    assert not spec.cancel_path.exists()

    asyncio.run(train_jobs.write_cancel(job_id))

    assert spec.cancel_path.exists()


@pytest.mark.parametrize(
    ('mutation', 'expected'),
    [
        ({'job_id': ''}, 'job_id'),
        ({'model_family': 'yolo99'}, 'model_family'),
        ({'model_size': 'z'}, 'model_size'),
        ({'profile': 'turbo'}, 'profile'),
        ({'dataset_export_dir': ''}, 'dataset_export_dir'),
        ({'dataset_export_dir': '/nope/does/not/exist'}, 'dataset_export_dir'),
        ({'include_classes': ['five']}, 'include_classes'),
        ({'hyperparameters': {'optimizer': 'auto'}}, 'optimizer'),
    ],
)
def test_broken_job_json_is_rejected(
    jobs_dir: Path, export_dir: Path, mutation: dict[str, Any], expected: str
) -> None:
    """A hand-written or stale job file must fail loudly, not half-run."""
    payload: dict[str, Any] = {
        'job_id': 'job-broken',
        'model_family': 'yolo26',
        'model_size': 'n',
        'profile': 'probe',
        'dataset_export_dir': str(export_dir),
        'hyperparameters': {},
    }
    payload.update(mutation)
    path = jobs_dir / 'job-broken.job.json'
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=expected):
        job_protocol.parse_and_validate_job(path)


def test_invalid_job_gets_a_terminal_failed_status(jobs_dir: Path) -> None:
    """Otherwise the watcher would re-read the same broken file forever."""
    (jobs_dir / 'job-bad.job.json').write_text('{not json')

    assert trainer.process_one(jobs_dir) is True

    status = asyncio.run(train_jobs.read_status('job-bad'))
    assert status is not None
    assert status.state == 'failed'
    assert 'validation' in (status.error or '')
    # Terminal -> no longer pending.
    assert job_protocol.list_pending_jobs(jobs_dir) == []


# =============================================================================
# on_fit_epoch_end metric semantics
# =============================================================================


class _SpecStub:
    """Just enough of ``JobSpec`` for ``_make_ultralytics_callbacks``:
    ``on_fit_epoch_end`` itself never reads ``spec`` at all, but the shared
    closure factory takes one positional argument."""

    cancel_path = Path('/nonexistent/does-not-exist.cancel')


class _FakeUltralyticsTrainer:
    def __init__(self, epoch: int, map50: float, map50_95: float) -> None:
        self.epoch = epoch  # 0-indexed, as Ultralytics reports it
        self.metrics = {'metrics/mAP50(B)': map50, 'metrics/mAP50-95(B)': map50_95}


def test_on_fit_epoch_end_tells_the_best_checkpoint_revalidation_apart_from_a_real_epoch() -> None:
    """Ultralytics' final_eval() re-validates best.pt and fires
    on_fit_epoch_end once more after training, WITHOUT advancing
    trainer.epoch. Before the fix, this call silently overwrote
    last_epoch_metric with the best checkpoint's own metrics (not the true
    last epoch's) -- live evidence: last_metric mAP50-95 0.857 vs
    results.csv's actual last-epoch row of 0.855. It must instead land in
    a distinct, single coherent best_checkpoint_metric row."""
    state = job_protocol.StatusState(job_id='x', state='running')
    _, _, on_fit_epoch_end = trainer._make_ultralytics_callbacks(_SpecStub(), state, total_epochs=2)

    # Epoch 1 (Ultralytics reports epoch=0).
    on_fit_epoch_end(_FakeUltralyticsTrainer(epoch=0, map50=0.70, map50_95=0.40))
    # Epoch 2, the true LAST training epoch (Ultralytics reports epoch=1).
    on_fit_epoch_end(_FakeUltralyticsTrainer(epoch=1, map50=0.855, map50_95=0.80))
    # final_eval()'s post-training re-validation of best.pt: same epoch=1,
    # different (better) metrics because it validates the BEST checkpoint,
    # not necessarily the last in-training epoch's weights.
    on_fit_epoch_end(_FakeUltralyticsTrainer(epoch=1, map50=0.86, map50_95=0.857))

    assert state.last_epoch_metric == {'epoch': 2, 'map50': 0.855, 'map50_95': 0.80}
    assert state.best_checkpoint_metric == {'epoch': 2, 'map50': 0.86, 'map50_95': 0.857}
    assert state.current_epoch == 2  # not clobbered by the revalidation call


# =============================================================================
# status.json: trainer writer -> API reader
# =============================================================================


def test_trainer_status_payload_parses_in_the_api_reader(jobs_dir: Path, export_dir: Path) -> None:
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    state = job_protocol.StatusState(job_id=job_id, state='running')
    state.current_epoch = 2
    state.total_epochs = 3
    state.eval = {'map50': 0.71, 'map50_95': 0.42}
    state.checkpoint_path = '/runs/x/weights/best.pt'
    job_protocol.write_status_now(spec.status_path, state)

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None
    assert status.state == 'running'
    assert status.current_epoch == 2
    assert status.total_epochs == 3
    assert status.checkpoint_path == '/runs/x/weights/best.pt'
    # The API back-fills best_checkpoint_metric from eval when the trainer omits it.
    assert status.best_checkpoint_metric == {'map50': 0.71, 'map50_95': 0.42}
    assert status.heartbeat_at is not None


@pytest.mark.parametrize(
    'state_name', sorted(job_protocol.TERMINAL_STATES | job_protocol.ACTIVE_STATES)
)
def test_every_trainer_state_is_accepted_by_the_api_model(
    jobs_dir: Path, export_dir: Path, state_name: str
) -> None:
    """A state the trainer can write that the API rejects would blank the UI."""
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    job_protocol.write_status_now(
        spec.status_path, job_protocol.StatusState(job_id=job_id, state=state_name)
    )

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None, f'API rejected trainer state {state_name!r}'
    assert status.state == state_name


def test_run_log_is_tailable_by_the_api(jobs_dir: Path, export_dir: Path) -> None:
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    with job_protocol.tee_to_run_log(spec.run_log_path):
        print('[trainer] epoch 1/3')
        print('[trainer] epoch 2/3')

    assert asyncio.run(train_jobs.tail_run_log(job_id, lines=10))[-1] == '[trainer] epoch 2/3'


def test_pending_list_skips_terminal_jobs_and_is_fifo(jobs_dir: Path, export_dir: Path) -> None:
    first = _write_job(job_id='2026-01-01T00-00-00_a', dataset_export_dir=str(export_dir))
    second = _write_job(job_id='2026-01-01T00-00-01_b', dataset_export_dir=str(export_dir))

    assert [p.name for p in job_protocol.list_pending_jobs(jobs_dir)] == [
        f'{first}.job.json',
        f'{second}.job.json',
    ]

    job_protocol.write_status_now(
        train_jobs._status_path(first), job_protocol.StatusState(job_id=first, state='finished')
    )
    assert [p.name for p in job_protocol.list_pending_jobs(jobs_dir)] == [f'{second}.job.json']


# =============================================================================
# class_remap: the trainer half of the promote fix
# =============================================================================


def test_subset_rewrite_emits_a_remap_promote_can_read(tmp_path: Path, export_dir: Path) -> None:
    """The remap payload shape is a contract with resolve_class_remap()."""
    out = tmp_path / 'subset'
    result = subset_dataset.build_subset_view(
        export_dir=export_dir, include_classes=[12, 5], single_cls=False, out_dir=out
    )

    # include_classes order drives the new ids, not registry order.
    assert result.class_remap == {12: 0, 5: 1}
    assert result.kept_rows == 6  # 2 kept classes x 3 splits
    assert result.dropped_rows == 3  # the excluded class, once per split

    weights_dir = tmp_path / 'run' / 'weights'
    weights_dir.mkdir(parents=True)
    (weights_dir / 'class_remap.json').write_bytes((out / 'class_remap.json').read_bytes())

    remap = resolve_class_remap(job_id='j', checkpoint_path=weights_dir / 'best.pt', manifest=None)
    assert remap.source == 'weights_dir'
    assert remap.mapping == {12: 0, 5: 1}
    assert remap.names == ['gadget', 'widget']
    assert remap.include_classes == [12, 5]

    # labels.txt for the promoted model must be the subset, renumbered.
    full_registry = {5: 'widget', 9: 'serial_plate', 12: 'gadget'}
    assert build_class_id_to_name(remap=remap, full_registry=full_registry) == {
        0: 'gadget',
        1: 'widget',
    }


def test_single_cls_rewrite_collapses_to_one_class(tmp_path: Path, export_dir: Path) -> None:
    out = tmp_path / 'subset'
    subset_dataset.build_subset_view(
        export_dir=export_dir, include_classes=[9], single_cls=True, out_dir=out
    )
    weights_dir = tmp_path / 'run' / 'weights'
    weights_dir.mkdir(parents=True)
    (weights_dir / 'class_remap.json').write_bytes((out / 'class_remap.json').read_bytes())

    remap = resolve_class_remap(job_id='j', checkpoint_path=weights_dir / 'best.pt', manifest=None)
    assert remap.single_cls is True
    assert build_class_id_to_name(remap=remap, full_registry={9: 'serial_plate'}) == {0: 'object'}


def test_subset_rewrite_rejects_a_registry_id_outside_the_export(
    tmp_path: Path, export_dir: Path
) -> None:
    with pytest.raises(ValueError, match='outside this export'):
        subset_dataset.build_subset_view(
            export_dir=export_dir,
            include_classes=[5, 999],
            single_cls=False,
            out_dir=tmp_path / 'subset',
        )


def test_class_remap_copy_reports_failure_loudly_for_a_subset_run(
    jobs_dir: Path, export_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A subset run whose remap never reaches the weights dir must say so.

    Silently returning True here is the bug the promote-side fix exists for:
    promote would fall back to the full registry and ship a labels.txt that
    doesn't describe the model it is serving.
    """
    monkeypatch.setattr(job_protocol, 'TMP_ROOT', tmp_path / 'tmp')
    job_id = _write_job(dataset_export_dir=str(export_dir), include_classes=[5], single_cls=False)
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    save_dir = tmp_path / 'run'
    save_dir.mkdir()

    # Subset run, but no class_remap.json was produced -> not a silent pass.
    assert dataset_prep.copy_class_remap_to_weights_dir(spec, save_dir) is False

    # Now produce one; the copy must land next to best.pt.
    spec.subset_dir.mkdir(parents=True)
    (spec.subset_dir / 'class_remap.json').write_text(
        json.dumps({'original_to_new': {'5': 0}, 'single_cls': False, 'names': ['widget']})
    )
    assert dataset_prep.copy_class_remap_to_weights_dir(spec, save_dir) is True
    assert (save_dir / 'weights' / 'class_remap.json').is_file()


def test_class_remap_copy_is_a_no_op_for_a_whole_export_run(
    jobs_dir: Path, export_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A full-class run legitimately has no remap -- that is not a failure."""
    monkeypatch.setattr(job_protocol, 'TMP_ROOT', tmp_path / 'tmp')
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    save_dir = tmp_path / 'run'
    save_dir.mkdir()

    assert dataset_prep.copy_class_remap_to_weights_dir(spec, save_dir) is True
    assert not (save_dir / 'weights' / 'class_remap.json').exists()


# =============================================================================
# GPU device resolution
# =============================================================================


def test_gpu_ids_pass_through_when_no_order_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('OP_TRAIN_GPU_ORDER', raising=False)
    assert trainer.resolve_local_device('2') == '2'
    assert trainer.resolve_local_device('0,2') == '0,2'
    assert trainer.resolve_local_device(None) is None
    assert trainer.resolve_local_device('') is None


def test_host_gpu_ids_map_to_positional_local_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container pinned to ['0','2'] sees host GPU 2 as local index 1.

    Deriving the device string from the *count* of requested GPUs would run a
    host-GPU-2-only request on host GPU 0 instead.
    """
    monkeypatch.setenv('OP_TRAIN_GPU_ORDER', '0,2')
    assert trainer.resolve_local_device('0') == '0'
    assert trainer.resolve_local_device('2') == '1'
    assert trainer.resolve_local_device('0,2') == '0,1'


def test_unattached_gpu_request_fails_instead_of_mis_scheduling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_TRAIN_GPU_ORDER', '0,2')
    with pytest.raises(ValueError, match='not attached to this container'):
        trainer.resolve_local_device('1')


# =============================================================================
# Orientation-sensitive class resolution (replaces the reference stack's
# hardcoded plate class id)
# =============================================================================


def test_text_classes_resolve_by_name_against_the_training_data_yaml(
    jobs_dir: Path, export_dir: Path
) -> None:
    job_id = _write_job(
        dataset_export_dir=str(export_dir),
        augmentation={'enabled': True, 'text_class_names': ['Serial_Plate']},
    )
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    # Matching is case-insensitive and lands in the data.yaml's id space.
    assert dataset_prep.resolve_text_classes(spec, export_dir / 'data.yaml') == {1}


def test_text_classes_follow_the_subset_renumbering(
    tmp_path: Path, jobs_dir: Path, export_dir: Path
) -> None:
    """After a subset rewrite the same name must resolve to its NEW id."""
    out = tmp_path / 'subset'
    subset_dataset.build_subset_view(
        export_dir=export_dir, include_classes=[12, 9], single_cls=False, out_dir=out
    )
    job_id = _write_job(
        dataset_export_dir=str(export_dir),
        include_classes=[12, 9],
        augmentation={'enabled': True, 'text_class_names': ['serial_plate']},
    )
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    assert dataset_prep.resolve_text_classes(spec, out / 'data.yaml') == {1}


def test_text_classes_fall_back_to_the_deployment_default(
    jobs_dir: Path, export_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_TRAIN_TEXT_CLASS_NAMES', 'serial_plate,absent_class')
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    # Unknown names are ignored rather than raising -- a subset run drops
    # classes legitimately.
    assert dataset_prep.resolve_text_classes(spec, export_dir / 'data.yaml') == {1}


# =============================================================================
# Campaign policy
# =============================================================================


@pytest.mark.parametrize(
    ('stop_when', 'eval_block', 'expected'),
    [
        (None, {'map50': 0.9}, False),
        ({'map50_at_least': 0.7}, {'map50': 0.71}, True),
        ({'map50_at_least': 0.7}, {'map50': 0.69}, False),
        ({'map50_at_least': 0.7}, {}, False),
        ({'map50_95_at_least': 0.4}, {'map50_95': 0.5}, True),
        ({'map50_at_least': 0.1, 'map50_95_at_least': 0.9}, {'map50': 0.5, 'map50_95': 0.5}, False),
        # Unknown rule fails CLOSED -- never skip siblings on a policy we
        # don't understand.
        ({'f1_at_least': 0.1}, {'map50': 0.99}, False),
    ],
)
def test_stop_when_policy(
    stop_when: dict[str, float] | None, eval_block: dict[str, Any], expected: bool
) -> None:
    assert campaign.stop_when_satisfied(stop_when, eval_block) is expected


def test_campaign_auto_skip_marks_queued_siblings_skipped(jobs_dir: Path, export_dir: Path) -> None:
    campaign_id = 'camp-1'
    winner = _write_job(
        job_id=f'{campaign_id}_run00_probe',
        campaign_id=campaign_id,
        dataset_export_dir=str(export_dir),
        stop_when={'map50_at_least': 0.5},
    )
    loser = _write_job(
        job_id=f'{campaign_id}_run01_medium',
        campaign_id=campaign_id,
        dataset_export_dir=str(export_dir),
        stop_when={'map50_at_least': 0.5},
    )
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{winner}.job.json')
    state = job_protocol.StatusState(job_id=winner, campaign_id=campaign_id, state='finished')
    state.eval = {'map50': 0.8}

    campaign.maybe_handle_campaign(spec, state)

    sibling = asyncio.run(train_jobs.read_status(loser))
    assert sibling is not None
    assert sibling.state == 'skipped'
    assert sibling.error == 'auto_skip_threshold_met'


def test_campaign_hook_is_a_no_op_for_a_standalone_job(jobs_dir: Path, export_dir: Path) -> None:
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    state = job_protocol.StatusState(job_id=job_id, state='finished')
    state.eval = {'map50': 0.9}

    campaign.maybe_handle_campaign(spec, state)  # must not raise or write


# =============================================================================
# Incumbent comparison
# =============================================================================


def test_comparison_is_skipped_when_no_incumbent_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh deployment with nothing to compare against must not error."""
    monkeypatch.delenv('OP_TRAIN_INCUMBENT_MODELS', raising=False)
    assert incumbent_compare.resolve_incumbent_model_name() is None


def test_results_csv_metrics_accept_both_ultralytics_spellings(tmp_path: Path) -> None:
    new_style = tmp_path / 'new.csv'
    new_style.write_text('epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.25\n2,0.8,0.4\n')
    row = incumbent_compare.read_results_csv_last_row(new_style)
    assert row is not None
    assert incumbent_compare.extract_top_level_metrics(row) == {'map50': 0.8, 'map50_95': 0.4}

    legacy = tmp_path / 'legacy.csv'
    legacy.write_text('epoch,metrics/mAP_0.5,metrics/mAP_0.5:0.95\n1,0.6,0.3\n')
    row = incumbent_compare.read_results_csv_last_row(legacy)
    assert row is not None
    assert incumbent_compare.extract_top_level_metrics(row) == {'map50': 0.6, 'map50_95': 0.3}

    assert incumbent_compare.read_results_csv_last_row(tmp_path / 'missing.csv') is None


def test_box_iou_matches_hand_computed_overlap() -> None:
    assert incumbent_compare.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert incumbent_compare.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # Overlap 5x10=50; union 100+100-50=150.
    assert incumbent_compare.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


# =============================================================================
# End-to-end run_job against a stub Ultralytics
# =============================================================================


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
    monkeypatch.setattr(trainer, 'RUNS_ROOT', str(tmp_path / 'runs'))
    monkeypatch.setattr(job_protocol, 'TMP_ROOT', tmp_path / 'tmp')
    # A local file-backed tracking store, so the real MLflow code path runs
    # (and is therefore covered) without reaching for a network service that
    # isn't there.
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'file://{tmp_path / "mlruns"}')
    return _StubYOLO


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
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    trainer.run_job(spec)

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None
    assert status.state == 'finished', status.error
    assert status.checkpoint_path is not None
    assert status.checkpoint_path.endswith('weights/best.pt')
    assert status.eval == {'map50': 0.77, 'map50_95': 0.44}
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
    assert manifest['lineage']['class_remap'] is None  # whole-export run
    assert manifest['lineage']['training_seed'] == 7
    assert manifest['lineage']['deterministic'] is True
    assert manifest['results']['final_state'] == 'finished'
    assert manifest['results']['checkpoint_sha256']
    assert manifest['promoted_to'] is None

    # A full-class run must not leave a remap for promote to trip over.
    assert (
        resolve_class_remap(
            job_id=job_id, checkpoint_path=Path(status.checkpoint_path), manifest=manifest
        ).source
        == 'none'
    )
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
    monkeypatch.setattr(lineage, '_docker_client', lambda: _FakeDockerClient())

    job_id = _write_job(
        dataset_export_dir=str(export_dir),
        model_size='n',
        profile='probe',
        hyperparameters={'epochs': 1, 'batch': 2, 'optimizer': 'MuSGD', 'seed': 3},
    )
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

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
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

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
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    asyncio.run(train_jobs.write_cancel(job_id))
    # The real cancel path raises out of the on_train_epoch_end callback.
    stub_ultralytics.raise_on_train = trainer.CancelRequestedError('cancelled')

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
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
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
