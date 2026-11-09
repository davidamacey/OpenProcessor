"""The promote export follows the detection head: NMS-free for a one-to-one head, embedded NMS
for any other.

On Ultralytics 8.4.x a bare ``export(format='onnx')`` of a YOLO26 checkpoint writes the raw
one-to-many head, which promote refuses (``src/services/training/onnx_head.py``); the head the
engine serves has to be asked for with ``nms=False``
(https://docs.ultralytics.com/modes/export/).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import trainer
import trainer_fit


if TYPE_CHECKING:
    from pathlib import Path


def _model_class(head: object, calls: list[dict[str, Any]]) -> type:
    class FakeYolo:
        def __init__(self, path: str = '') -> None:
            self.path = path
            self.model = SimpleNamespace(model=[object(), head])

        def export(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            return 'best.onnx'

    return FakeYolo


def _spec(tmp_path: Path, **hyperparameters: Any) -> Any:
    return SimpleNamespace(
        hyperparameters=hyperparameters, run_log_path=tmp_path / 'run.log', model_family='yolo26'
    )


def _export(tmp_path: Path, head: object, **hyperparameters: Any) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    model = _model_class(head, calls)()
    trainer_fit._export_onnx(_spec(tmp_path, **hyperparameters), model, tmp_path / 'best.pt')
    (kwargs,) = calls
    return kwargs


ONE2ONE_HEAD = SimpleNamespace(one2one_cv2=[], cv2=[])
NMS_HEAD = SimpleNamespace(cv2=[], cv3=[])


def test_a_one_to_one_head_is_exported_nms_free(tmp_path: Path) -> None:
    kwargs = _export(tmp_path, ONE2ONE_HEAD)
    assert kwargs['nms'] is False
    assert 'half' not in kwargs


def test_any_other_head_embeds_nms_at_the_served_thresholds(tmp_path: Path) -> None:
    kwargs = _export(tmp_path, NMS_HEAD)
    assert kwargs['nms'] is True
    assert (kwargs['agnostic_nms'], kwargs['iou'], kwargs['conf'], kwargs['max_det']) == (
        True,
        0.7,
        0.25,
        300,
    )
    assert 'half' not in kwargs


@pytest.mark.parametrize('head', [ONE2ONE_HEAD, NMS_HEAD])
def test_the_dynamic_batch_shape_and_the_export_batch_are_unchanged(
    tmp_path: Path, head: object
) -> None:
    kwargs = _export(tmp_path, head)
    assert (kwargs['format'], kwargs['dynamic'], kwargs['simplify']) == ('onnx', True, True)
    assert kwargs['batch'] == trainer_fit.DEFAULT_EXPORT_BATCH
    assert _export(tmp_path, head, export_batch=16)['batch'] == 16


def test_the_test_split_validation_scores_the_served_one_to_one_head() -> None:
    inner = SimpleNamespace(model=[object(), ONE2ONE_HEAD], end2end=False)
    eval_model = SimpleNamespace(model=inner)
    assert trainer._select_served_head(eval_model, 'yolo26') == 'end2end'
    assert inner.end2end is True


def test_a_head_without_the_one_to_one_branch_keeps_its_own_head() -> None:
    inner = SimpleNamespace(model=[object(), NMS_HEAD], end2end=False)
    assert trainer._select_served_head(SimpleNamespace(model=inner), 'yolo26') is None
    assert inner.end2end is False
