"""The eval block (test split vs training-time validation), on_fit_epoch_end metric semantics
and the incumbent comparison.
"""

from __future__ import annotations

from pathlib import Path

import incumbent_compare
import job_status
import pytest
import trainer


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
    state = job_status.StatusState(job_id='x', state='running')
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


class _FakeMetric:
    """Stand-in for Ultralytics' ``Metric`` (``DetMetrics.box``)."""

    def __init__(
        self,
        *,
        map50: float,
        map_: float,
        mp: float,
        mr: float,
        ap_class_index: list[int],
        p: list[float],
        r: list[float],
        f1: list[float],
        ap50: list[float],
    ) -> None:
        self.map50 = map50
        self.map = map_
        self.mp = mp
        self.mr = mr
        self.ap_class_index = ap_class_index
        self.p = p
        self.r = r
        self.f1 = f1
        self.ap50 = ap50


class _FakeDetMetrics:
    """Stand-in for the object ``model.val(...)`` returns."""

    def __init__(self, box: _FakeMetric, names: dict[int, str], nt_per_class: list[int]) -> None:
        self.box = box
        self.names = names
        self.nt_per_class = nt_per_class


def _fake_test_val_results() -> _FakeDetMetrics:
    box = _FakeMetric(
        map50=0.62,
        map_=0.41,
        mp=0.71,
        mr=0.55,
        ap_class_index=[0, 1],
        p=[0.8, 0.6],
        r=[0.7, 0.5],
        f1=[0.75, 0.55],
        ap50=[0.79, 0.5],
    )
    return _FakeDetMetrics(box, names={0: 'widget', 1: 'gadget'}, nt_per_class=[12, 8])


def _write_results_csv(save_dir: Path, map50: float, map50_95: float) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'results.csv').write_text(
        f'epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,{map50},{map50_95}\n'
    )


def test_extract_test_summary_reads_detmetrics_box_properties() -> None:
    assert incumbent_compare.extract_test_summary(_fake_test_val_results()) == {
        'map50': pytest.approx(0.62),
        'map50_95': pytest.approx(0.41),
        'precision': pytest.approx(0.71),
        'recall': pytest.approx(0.55),
    }


def test_extract_test_summary_empty_when_box_missing() -> None:
    class _NoBox:
        pass

    assert incumbent_compare.extract_test_summary(_NoBox()) == {}


def test_populate_eval_block_prefers_the_test_split_over_the_training_time_val_split(
    tmp_path: Path,
) -> None:
    """The bug: eval.map50/map50_95 must be the frozen TEST split (from a
    fresh val() pass), not the per-epoch VALIDATION split recorded in
    results.csv -- even though both are available."""
    save_dir = tmp_path / 'run'
    # Deliberately distinct from the test numbers so a conflation is
    # unmistakable rather than a lucky coincidence.
    _write_results_csv(save_dir, map50=0.9191, map50_95=0.85096)
    state = job_status.StatusState(job_id='j-test-split', state='running')

    trainer.populate_eval_block(
        state, save_dir, val_results=_fake_test_val_results(), data_yaml_path=None
    )

    assert state.eval is not None
    assert state.eval['split'] == 'test'
    assert state.eval['map50'] == pytest.approx(0.62)
    assert state.eval['map50_95'] == pytest.approx(0.41)
    assert state.eval['precision'] == pytest.approx(0.71)
    assert state.eval['recall'] == pytest.approx(0.55)
    assert len(state.eval['per_class']) == 2
    # The training-time validation numbers are kept, but clearly named --
    # never under the headline map50/map50_95 keys.
    assert state.eval['val_last'] == pytest.approx({'map50': 0.9191, 'map50_95': 0.85096})
    assert state.eval['map50'] != pytest.approx(0.9191)
    assert state.eval['map50_95'] != pytest.approx(0.85096)


def test_populate_eval_block_falls_back_to_val_when_the_test_pass_is_missing(
    tmp_path: Path,
) -> None:
    save_dir = tmp_path / 'run'
    _write_results_csv(save_dir, map50=0.9191, map50_95=0.85096)
    state = job_status.StatusState(job_id='j-no-test', state='running')

    trainer.populate_eval_block(state, save_dir, val_results=None, data_yaml_path=None)

    assert state.eval is not None
    assert state.eval['split'] == 'val'
    assert 'per_class' not in state.eval
    assert state.eval['map50'] == pytest.approx(0.9191)
    assert state.eval['map50_95'] == pytest.approx(0.85096)
    assert state.eval['val_last'] == pytest.approx({'map50': 0.9191, 'map50_95': 0.85096})


def test_populate_eval_block_falls_back_to_val_when_per_class_is_unavailable(
    tmp_path: Path,
) -> None:
    """A test pass that ran but returned no usable per-class rows (older
    Ultralytics, or a degenerate ``box``) must not be labeled 'test' --
    that would silently ship an empty/absent per_class under a 'test'
    label for the promote gate to trip over."""
    save_dir = tmp_path / 'run'
    _write_results_csv(save_dir, map50=0.5, map50_95=0.3)
    empty_box = _FakeMetric(
        map50=0.7, map_=0.5, mp=0.6, mr=0.4, ap_class_index=[], p=[], r=[], f1=[], ap50=[]
    )
    val_results = _FakeDetMetrics(empty_box, names={}, nt_per_class=[])
    state = job_status.StatusState(job_id='j-no-per-class', state='running')

    trainer.populate_eval_block(state, save_dir, val_results=val_results, data_yaml_path=None)

    assert state.eval is not None
    assert state.eval['split'] == 'val'
    assert 'per_class' not in state.eval
    assert state.eval['map50'] == pytest.approx(0.5)  # the val number, not the test box's 0.7
