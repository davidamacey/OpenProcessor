"""Tests for the bake-off quantize leg (scripts/curation/bakeoff/quantize.py)
and how the job runner reports quantize / CoreML stage failures.

The Ultralytics export itself is stubbed (it needs a real checkpoint and,
for fp16, a GPU); INT8 QDQ quantization runs for real on a tiny ONNX conv
graph so the calibration reader and quantizer wiring are exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.curation.bakeoff import bakeoff_runner, quant_stage, quantize


def _img(path: Path, w: int = 64, h: int = 48) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(path.name))
    cv2.imwrite(str(path), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    return path


def _dataset(root: Path, split: str, n_pos: int, n_bg: int) -> Path:
    for i in range(n_pos + n_bg):
        _img(root / 'images' / split / f'f{i:03d}.png')
        lbl = root / 'labels' / split / f'f{i:03d}.txt'
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text('0 0.5 0.5 0.2 0.2\n' if i < n_pos else '')
    return root


# --- request validation ------------------------------------------------------


def test_validate_request_errors(tmp_path: Path) -> None:
    pt = tmp_path / 'best.pt'
    with pytest.raises(quantize.QuantizeError, match='checkpoint is required'):
        quantize.validate_request(['fp32_onnx'], None, None)
    with pytest.raises(quantize.QuantizeError, match='weights not found'):
        quantize.validate_request(['fp32_onnx'], pt, None)
    pt.write_bytes(b'x')
    with pytest.raises(quantize.QuantizeError, match='unknown formats'):
        quantize.validate_request(['fp8_onnx'], pt, None)
    with pytest.raises(quantize.QuantizeError, match='calibration dataset'):
        quantize.validate_request(['int8_onnx'], pt, None)
    assert quantize.validate_request(['fp16_onnx'], pt, None) == (['fp16_onnx'], pt)


def test_quantize_error_is_catchable_not_systemexit() -> None:
    assert issubclass(quantize.QuantizeError, Exception)
    assert not issubclass(quantize.QuantizeError, SystemExit)


def test_run_rejects_path_like_model_id(tmp_path: Path) -> None:
    with pytest.raises(quantize.QuantizeError, match='invalid model_id'):
        quantize.run(
            '../x',
            ['fp32_onnx'],
            tmp_path,
            pt_override=None,
            calib_override=None,
            n_calib_override=None,
        )


# --- calibration selection ---------------------------------------------------


def test_calibration_mixes_backgrounds_and_is_deterministic(tmp_path: Path) -> None:
    root = _dataset(tmp_path / 'ds', 'train', n_pos=30, n_bg=10)
    a = quantize.select_calibration_images(root, 'train', 20)
    b = quantize.select_calibration_images(root, 'train', 20)
    assert a == b
    assert len(a) == 20
    n_bg = sum(1 for p in a if int(p.stem[1:]) >= 30)
    assert n_bg == 3  # round(20 * 0.15)


def test_calibration_missing_split_is_a_clear_error(tmp_path: Path) -> None:
    root = _dataset(tmp_path / 'ds', 'test', n_pos=2, n_bg=0)
    with pytest.raises(quantize.QuantizeError, match='images/train'):
        quantize.select_calibration_images(root, 'train', 5)


# --- run(): manifest + artifacts (exporter stubbed) ----------------------------


def _tiny_conv_onnx(path: Path, size: int) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(
        np.random.default_rng(0).standard_normal((4, 3, 3, 3)).astype(np.float32), 'w'
    )
    node = helper.make_node('Conv', ['images', 'w'], ['out'], pads=[1, 1, 1, 1])
    graph = helper.make_graph(
        [node],
        'tiny',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info('out', TensorProto.FLOAT, [1, 4, size, size])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_run_writes_artifacts_and_manifest(tmp_path: Path, monkeypatch) -> None:
    size = 32
    exported: list[tuple[str, bool]] = []

    def fake_export(pt_path: Path, out: Path, imgsz: int, *, half: bool) -> Path:
        assert pt_path.name == 'source.pt'  # works from the writable copy
        assert imgsz == size
        exported.append((out.name, half))
        _tiny_conv_onnx(out, size)
        return out

    monkeypatch.setattr(quantize, '_ultralytics_export_onnx', fake_export)
    pt = tmp_path / 'best.pt'
    pt.write_bytes(b'weights')
    calib = _dataset(tmp_path / 'export', 'train', n_pos=4, n_bg=2)

    manifest = quantize.run(
        'cand',
        ['fp32_onnx', 'fp16_onnx', 'int8_onnx'],
        tmp_path / 'quant',
        pt_override=pt,
        calib_override=calib,
        n_calib_override=5,
        imgsz=size,
    )
    out = tmp_path / 'quant' / 'cand'
    assert exported == [('fp32.onnx', False), ('fp16.onnx', True)]
    assert [a['format'] for a in manifest['artifacts']] == ['fp32_onnx', 'fp16_onnx', 'int8_onnx']
    assert (out / 'int8_qdq.onnx').is_file()
    assert (out / 'source.pt').read_bytes() == b'weights'
    assert manifest['calibration']['n_images'] == 5
    assert manifest['calibration']['split'] == 'train'
    assert json.loads((out / 'manifest.json').read_text())['model_id'] == 'cand'
    assert len(json.loads((out / 'calibration_manifest.json').read_text())) == 5
    import onnx

    ops = {n.op_type for n in onnx.load(str(out / 'int8_qdq.onnx')).graph.node}
    assert 'QuantizeLinear' in ops


# --- runner: stage failures are reported, not swallowed ------------------------


def _fake_scoring(monkeypatch) -> None:
    def fake_task(ds_path, ds_out, model, gpu):
        ds_out.mkdir(parents=True, exist_ok=True)
        return model['name'], True, None

    monkeypatch.setattr(bakeoff_runner, '_run_task', fake_task)


def test_runner_reports_coreml_as_failed_stage(tmp_path: Path, monkeypatch) -> None:
    _fake_scoring(monkeypatch)
    ds = tmp_path / 'ds'
    ds.mkdir()
    monkeypatch.setattr(quant_stage, '_quantize_and_variant_models', lambda *_a: [])
    status = bakeoff_runner.run_job(
        {
            'job_id': 'c1',
            'verify_frozen': False,
            'out_dir': str(tmp_path / 'out'),
            'datasets': [str(ds)],
            'models': [{'backend': 'ultralytics', 'name': 'm1'}],
            'quantize': {'checkpoint': '/x.pt', 'coreml': True},
        }
    )
    assert status['state'] == 'done'
    assert {'stage': 'coreml', 'error': bakeoff_runner.COREML_UNAVAILABLE} in status['failed']


def test_runner_quantize_failure_is_recorded(tmp_path: Path, monkeypatch) -> None:
    _fake_scoring(monkeypatch)
    ds = tmp_path / 'ds'
    ds.mkdir()
    status = bakeoff_runner.run_job(
        {
            'job_id': 'q1',
            'verify_frozen': False,
            'out_dir': str(tmp_path / 'out'),
            'datasets': [str(ds)],
            'models': [{'backend': 'ultralytics', 'name': 'm1'}],
            'quantize': {'model_id': 'cand', 'checkpoint': str(tmp_path / 'missing.pt')},
        }
    )
    assert status['state'] == 'done'  # the plain model was still scored
    [failure] = [f for f in status['failed'] if f.get('stage') == 'quantize']
    assert 'weights not found' in failure['error']


def test_runner_quantize_only_job_errors_when_export_fails(tmp_path: Path) -> None:
    ds = tmp_path / 'ds'
    ds.mkdir()
    status = bakeoff_runner.run_job(
        {
            'job_id': 'q2',
            'verify_frozen': False,
            'out_dir': str(tmp_path / 'out'),
            'datasets': [str(ds)],
            'quantize': {'model_id': 'cand'},
        }
    )
    assert status['state'] == 'error'
    assert 'quantize: QuantizeError: a checkpoint is required' in status['error']
    on_disk = json.loads((tmp_path / 'out' / 'status.json').read_text())
    assert on_disk['state'] == 'error'


def test_runner_quant_variants_default_under_job_out_dir(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}

    def fake_run(model_id, formats, out_root, **kw):
        seen.update(model_id=model_id, out_root=out_root, **kw)
        (out_root / model_id).mkdir(parents=True)
        (out_root / model_id / 'fp16.onnx').write_bytes(b'x')

    monkeypatch.setattr(quantize, 'run', fake_run)
    models = quant_stage._quantize_and_variant_models(
        {'model_id': 'cand', 'checkpoint': '/c.pt', 'formats': ['fp16_onnx'], 'imgsz': 320},
        [{'name': 'd', 'path': '/data/d'}],
        tmp_path / 'job',
    )
    assert seen['out_root'] == tmp_path / 'job' / 'quant'
    assert seen['calib_override'] == Path('/data/d')
    assert seen['imgsz'] == 320
    assert [m['name'] for m in models] == ['ours_fp16_onnx']
