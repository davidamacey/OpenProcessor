"""The status ``eval`` block: test-split vs validation-split metrics and the incumbent comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import incumbent_compare
from logutil import get_logger


if TYPE_CHECKING:
    from pathlib import Path

    from job_status import StatusState


logger = get_logger('trainer.trainer_eval')


def populate_eval_block(
    state: StatusState,
    save_dir: Path,
    val_results: Any | None = None,
    data_yaml_path: Path | None = None,
    eval_head: str | None = None,
) -> None:
    """Parse Ultralytics' artifacts off disk into ``state.eval`` (+ ``compare``).

    ``results.csv``'s last row is Ultralytics' per-epoch **validation** metric
    (the split reserved for early-stopping/model-selection during training,
    recorded every epoch). ``val_results`` -- when the caller passed one -- is
    a fresh ``model.val(..., split='test')`` pass against the frozen holdout
    the run was never trained or tuned against. These are NOT
    interchangeable: labeling the val number as "test" overstates how the
    model will do on unseen data.

    The overall ``eval.map50`` / ``eval.map50_95`` (+ ``precision`` /
    ``recall`` / ``per_class`` when available) come from the test pass
    whenever it succeeded (``eval.split == 'test'``). When it didn't run or
    produced no usable ``box`` metrics, they fall back to the training-time
    validation numbers (``eval.split == 'val'``, no ``per_class`` -- those
    would silently be val-split numbers mislabeled as test). The val numbers
    are always additionally kept, clearly named, under ``eval.val_last`` so a
    consumer that specifically wants the training-time curve still can. The
    confusion-matrix PNG path (server filesystem; the API rewrites this to a
    servable URL before it reaches the wire) rounds out the block. When an
    incumbent is configured and reachable, the side-by-side comparison lands
    in ``state.compare``.

    ``eval_head`` (from :func:`_finalize_run`) records which detection head
    that same test-split ``val()`` pass scored -- ``"end2end"`` when the
    loaded checkpoint's NMS-free one-to-one head was explicitly forced to
    match what's actually served (see ``_finalize_run``), ``None`` for a
    model family with no such distinction. Making this explicit means a
    comparison result never silently depends on Ultralytics' own
    checkpoint-dependent ``.val()`` default.
    """
    eval_block: dict[str, Any] = {}
    row = incumbent_compare.read_results_csv_last_row(save_dir / 'results.csv')
    val_last = incumbent_compare.extract_top_level_metrics(row) if row is not None else {}

    test_summary: dict[str, float] = {}
    per_class: list[dict[str, Any]] = []
    if val_results is not None:
        test_summary = incumbent_compare.extract_test_summary(val_results)
        per_class = incumbent_compare.per_class_from_val_results(val_results)

    if test_summary and per_class:
        eval_block.update(test_summary)
        eval_block['split'] = 'test'
        eval_block['per_class'] = per_class
    else:
        eval_block.update(val_last)
        eval_block['split'] = 'val'

    if val_last:
        eval_block['val_last'] = val_last

    cm_path = save_dir / 'confusion_matrix.png'
    if cm_path.is_file():
        eval_block['confusion_matrix_path'] = str(cm_path)

    if eval_head is not None:
        eval_block['head'] = eval_head

    if eval_block:
        with state.lock:
            state.eval = eval_block

    if val_results is None or data_yaml_path is None or not per_class:
        return
    try:
        compare_block = incumbent_compare.build_compare_block(
            save_dir=save_dir,
            candidate_per_class=per_class,
            candidate_summary={k: v for k, v in eval_block.items() if k in ('map50', 'map50_95')},
            data_yaml_path=data_yaml_path,
        )
    except Exception as exc:  # a missing Triton must not tank the run
        logger.warning('compare: comparison block failed', error=str(exc))
        return
    if compare_block is not None:
        with state.lock:
            state.compare = compare_block
