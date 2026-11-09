"""``export/ultralytics_export.py``: one export for every Ultralytics detection family.

Offline: the model is a stand-in whose ``export`` records its arguments and writes a small ONNX,
and the TensorRT build is replaced by a function that writes a plan file. What is checked is what
this repository decides: the documented ``export`` arguments of each head kind
(https://docs.ultralytics.com/modes/export/), the head check, where the ONNX, the engine, the
labels and the config land, and that a failure is an error result, never a half-served model.
The official-weights rows are in ``tests/export/test_family_matrix.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper

from src.domain.model_families import NMS, ONE2ONE, export_kwargs


EXPORT_DIR = Path(__file__).resolve().parents[2] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import ultralytics_export  # noqa: E402  # follows the sys.path setup


def _fused_onnx(path: Path, dims: list[int | str], *, outputs: int = 1) -> Path:
    outs = [
        helper.make_tensor_value_info(f'output{i}', TensorProto.FLOAT, dims) for i in range(outputs)
    ]
    graph = helper.make_graph(
        [helper.make_node('Identity', ['images'], [o.name]) for o in outs],
        'head',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, dims)],
        outs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


class FakeYolo:
    """What ``ultralytics.YOLO`` exposes to the export: ``export``, ``names`` and the head."""

    def __init__(self, pt_file: Path, head_kind: str, dims: list[int | str] | None = None) -> None:
        self.pt_file = pt_file
        self.names = {0: 'car', 1: 'plate'}
        head = SimpleNamespace(one2one_cv2=[]) if head_kind == ONE2ONE else SimpleNamespace()
        self.model = SimpleNamespace(model=[SimpleNamespace(), head])
        self.dims = dims or ['batch', 300, 6]
        self.export_calls: list[dict[str, Any]] = []

    def export(self, **kwargs: Any) -> str:
        self.export_calls.append(kwargs)
        return str(_fused_onnx(self.pt_file.with_suffix('.onnx'), self.dims))


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """The module's directories under ``tmp_path`` and a fake TensorRT build."""
    models, weights = tmp_path / 'models', tmp_path / 'pytorch_models'
    weights.mkdir()
    monkeypatch.setattr(ultralytics_export, 'MODELS_DIR', models)
    built: list[tuple[Path, Path, int]] = []

    def build(onnx_path: Path, plan_path: Path, max_batch: int) -> bool:
        built.append((onnx_path, plan_path, max_batch))
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_bytes(b'plan')
        return True

    stub = ModuleType('yolo_trt_build')
    stub.build_fused_engine = build  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'yolo_trt_build', stub)
    return SimpleNamespace(models=models, weights=weights, built=built, stub=stub)


def _export(
    repo: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    head_kind: str,
    *,
    formats: list[str],
    dims: list[int | str] | None = None,
    max_batch: int = 8,
) -> tuple[dict[str, Any], FakeYolo]:
    pt = repo.weights / 'car.pt'
    pt.write_bytes(b'x' * 10)
    fake = FakeYolo(pt, head_kind, dims)
    monkeypatch.setattr(ultralytics_export, 'load_model', lambda _path: fake)
    config = {'pt_file': str(pt), 'triton_name': 'car', 'max_batch': max_batch}
    return ultralytics_export.export_model('car', config, formats), fake


@pytest.mark.parametrize('kind', [NMS, ONE2ONE])
def test_the_export_call_is_the_documented_one_for_the_head_kind(
    kind: str, repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, fake = _export(repo, monkeypatch, kind, formats=['onnx'], max_batch=8)

    assert result['status'] == 'success', result
    assert fake.export_calls == [
        {
            'format': 'onnx',
            'imgsz': 640,
            'dynamic': True,
            'simplify': True,
            'device': 'cpu',
            **export_kwargs(kind, 8),
        }
    ]
    assert 'end2end' not in fake.export_calls[0]


@pytest.mark.parametrize(
    ('kind', 'engine'), [(NMS, 'car_trt_end2end'), (ONE2ONE, 'car_trt')], ids=['nms', 'one2one']
)
def test_an_engine_export_lands_the_onnx_the_engine_the_labels_and_the_config(
    kind: str, engine: str, repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _export(repo, monkeypatch, kind, formats=['trt'])

    assert result['status'] == 'success', result
    onnx_path = repo.models / 'car_end2end' / '1' / 'model.onnx'
    plan = repo.models / engine / '1' / 'model.plan'
    assert onnx_path.is_file()
    assert repo.built == [(onnx_path, plan, 8)]
    assert (repo.models / engine / 'labels.txt').read_text() == 'car\nplate\n'
    config = (repo.models / engine / 'config.pbtxt').read_text()
    assert f'name: "{engine}"' in config
    assert 'max_batch_size: 8' in config
    assert 'name: "output0"' in config
    assert 'dims: [ 300, 6 ]' in config
    assert 'TYPE_UINT8' in config
    assert result['head_kind'] == kind


def test_formats_onnx_stops_before_the_engine(
    repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _export(repo, monkeypatch, NMS, formats=['onnx'])

    assert result['status'] == 'success'
    assert (repo.models / 'car_end2end' / '1' / 'model.onnx').is_file()
    assert repo.built == []
    assert not (repo.models / 'car_trt_end2end').exists()


def test_a_raw_head_is_an_error_and_no_engine_is_built(
    repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _export(repo, monkeypatch, NMS, formats=['trt'], dims=['batch', 84, 'anchors'])

    assert result['status'] == 'error'
    assert 'raw one-to-many head' in result['error']
    assert repo.built == []
    assert not (repo.models / 'car_trt_end2end').exists()


def test_a_failed_engine_build_is_an_error_and_writes_no_config(
    repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.stub.build_fused_engine = lambda *_a: False  # type: ignore[attr-defined]

    result, _ = _export(repo, monkeypatch, NMS, formats=['trt'])

    assert result['status'] == 'error'
    assert not (repo.models / 'car_trt_end2end' / 'config.pbtxt').exists()


def test_a_missing_weights_file_is_an_error_naming_the_model(repo: SimpleNamespace) -> None:
    config = {'pt_file': str(repo.weights / 'gone.pt'), 'triton_name': 'car', 'max_batch': 8}

    result = ultralytics_export.export_model('car', config, ['trt'])

    assert result['status'] == 'error'
    assert result['triton_name'] == 'car'
    assert 'gone.pt' in result['error']


def test_inspect_refuses_the_raw_one_to_many_head(tmp_path: Path) -> None:
    raw = _fused_onnx(tmp_path / 'raw.onnx', ['batch', 84, 'anchors'])

    with pytest.raises(RuntimeError, match=r'raw one-to-many head.*\[84, -1\]|\[84, -1\].*raw'):
        ultralytics_export.inspect_fused_output(raw)


def test_inspect_accepts_the_fused_300_by_6_head(tmp_path: Path) -> None:
    fused = _fused_onnx(tmp_path / 'fused.onnx', ['batch', 300, 6])

    assert ultralytics_export.inspect_fused_output(fused) == ('output0', [300, 6])


def test_inspect_still_refuses_several_outputs(tmp_path: Path) -> None:
    many = _fused_onnx(tmp_path / 'many.onnx', ['batch', 300, 6], outputs=2)

    with pytest.raises(RuntimeError, match='single fused output'):
        ultralytics_export.inspect_fused_output(many)


def test_any_failed_result_fails_the_run() -> None:
    from yolo_export_report import any_export_failed

    assert any_export_failed([{'model': 'a', 'status': 'success'}]) is False
    assert any_export_failed([]) is False
    assert (
        any_export_failed([{'model': 'a', 'status': 'success'}, {'model': 'b', 'status': 'error'}])
        is True
    )


def test_the_summary_prints_an_error_result_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from yolo_export_report import print_summary

    caplog.set_level(logging.INFO)
    print_summary([{'model': 'small', 'triton_name': 'x', 'status': 'error', 'error': 'no file'}])

    assert 'no file' in caplog.text


def test_an_identical_config_is_not_rewritten(tmp_path: Path) -> None:
    """``models/*/config.pbtxt`` is tracked: an export run that changes nothing must not dirty it."""
    from yolo_triton_config import save_fused_config

    model_dir = tmp_path / 'car_trt_end2end'
    first = save_fused_config(model_dir, 'car_trt_end2end', 64, 'output0', [300, 6])
    before = first.stat().st_mtime_ns

    again = save_fused_config(model_dir, 'car_trt_end2end', 64, 'output0', [300, 6])

    assert again == first
    assert again.stat().st_mtime_ns == before


def test_a_differing_config_is_kept_unless_overwrite_is_asked(tmp_path: Path) -> None:
    from yolo_triton_config import save_fused_config

    model_dir = tmp_path / 'car_trt_end2end'
    config = save_fused_config(model_dir, 'car_trt_end2end', 64, 'output0', [300, 6])

    side = save_fused_config(model_dir, 'car_trt_end2end', 32, 'output0', [300, 6])
    assert side.name == 'config.pbtxt.generated'
    assert 'max_batch_size: 64' in config.read_text()

    replaced = save_fused_config(
        model_dir, 'car_trt_end2end', 32, 'output0', [300, 6], overwrite=True
    )
    assert replaced == config
    assert 'max_batch_size: 32' in config.read_text()
