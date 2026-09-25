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
from scripts.curation.bakeoff.freeze import test_sha as label_sha


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


# --- TensorRT .plan export (G-21: no `trtexec --fp16`, removed in TRT 11.1) --


def test_export_trt_plan_never_passes_fp16_flag(tmp_path: Path, monkeypatch) -> None:
    """trtexec on TensorRT 11.1 (strongly typed) has no --fp16 flag; passing
    it makes trtexec exit non-zero. Precision comes from the ONNX itself."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        calls.append(cmd)

    fake_trtexec = tmp_path / 'trtexec'
    fake_trtexec.write_bytes(b'')
    monkeypatch.setattr(quantize.shutil, 'which', lambda _name: str(fake_trtexec))
    monkeypatch.setattr(quantize, 'subprocess', type('S', (), {'run': staticmethod(fake_run)}))
    onnx_path = tmp_path / 'fp16.onnx'
    onnx_path.write_bytes(b'x')
    out = quantize.export_trt_plan(onnx_path, tmp_path / 'model.plan', int8=False)
    assert out == tmp_path / 'model.plan'
    assert len(calls) == 1
    assert '--fp16' not in calls[0]
    assert not any(arg.startswith('--fp16') for arg in calls[0])


def test_run_prefers_fp16_onnx_as_plan_source_over_fp32(tmp_path: Path, monkeypatch) -> None:
    """When both fp32.onnx and fp16.onnx exist and int8 wasn't requested,
    the plan is built from the fp16-baked ONNX -- otherwise TRT 11.1 has
    no way to bake reduced precision in after the fact."""

    def fake_export(pt_path: Path, out: Path, imgsz: int, *, half: bool) -> Path:
        out.write_bytes(b'x')
        return out

    plan_srcs: list[Path] = []

    def fake_plan(onnx_path: Path, out: Path, *, int8: bool) -> Path:
        plan_srcs.append(onnx_path)
        out.write_bytes(b'plan')
        return out

    monkeypatch.setattr(quantize, '_ultralytics_export_onnx', fake_export)
    monkeypatch.setattr(quantize, 'export_trt_plan', fake_plan)
    pt = tmp_path / 'best.pt'
    pt.write_bytes(b'weights')

    quantize.run(
        'cand',
        ['fp32_onnx', 'fp16_onnx', 'plan'],
        tmp_path / 'quant',
        pt_override=pt,
        calib_override=None,
        n_calib_override=None,
    )
    assert plan_srcs == [tmp_path / 'quant' / 'cand' / 'fp16.onnx']
    assert (tmp_path / 'quant' / 'cand' / 'model.plan').is_file()


# --- runner: stage failures are reported, not swallowed ------------------------


def _fake_scoring(monkeypatch) -> None:
    def fake_task(ds_path, ds_out, ds_id, model, gpu):
        ds_out.mkdir(parents=True, exist_ok=True)
        return model['model'], True, None

    monkeypatch.setattr(bakeoff_runner, '_run_task', fake_task)


def _v2_spec(tmp_path: Path, *, models: list, quant: dict) -> dict:
    """A job spec v2 over one (empty) dataset whose test split hash is current."""
    ds = tmp_path / 'ds'
    ds.mkdir(exist_ok=True)
    return {
        'schema_version': 2,
        'job_id': 'q',
        'profile': 'generic',
        'out_dir': str(tmp_path / 'out'),
        'datasets': [
            {
                'id': 'export:ds',
                'dir_name': 'export__ds',
                'path': str(ds),
                'test_label_sha': label_sha(ds)[0],
                'frozen_test_sha': '',
                'eval_class_ids': [],
            }
        ],
        'models': models,
        'quantize': quant,
    }


_MODEL = {'model': 'custom:m1', 'backend': 'ultralytics', 'class_map_by_dataset': {}}


def test_runner_reports_coreml_as_failed_stage(tmp_path: Path, monkeypatch) -> None:
    _fake_scoring(monkeypatch)
    monkeypatch.setattr(quant_stage, '_quantize_and_variant_models', lambda *_a: [])
    status = bakeoff_runner.run_job(
        _v2_spec(tmp_path, models=[_MODEL], quant={'checkpoint': '/x.pt', 'coreml': True})
    )
    assert status['state'] == 'done'
    assert {
        'stage': 'coreml',
        'dataset': None,
        'model': None,
        'error': bakeoff_runner.COREML_UNAVAILABLE,
    } in status['failed']


def test_runner_quantize_failure_is_recorded(tmp_path: Path, monkeypatch) -> None:
    _fake_scoring(monkeypatch)
    status = bakeoff_runner.run_job(
        _v2_spec(
            tmp_path,
            models=[_MODEL],
            quant={'run_id': 'cand', 'checkpoint': str(tmp_path / 'missing.pt')},
        )
    )
    assert status['state'] == 'done'  # the plain model was still scored
    [failure] = [f for f in status['failed'] if f.get('stage') == 'quantize']
    assert 'weights not found' in failure['error']


def test_runner_quantize_only_job_errors_when_export_fails(tmp_path: Path) -> None:
    status = bakeoff_runner.run_job(_v2_spec(tmp_path, models=[], quant={'run_id': 'cand'}))
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
        {
            'run_id': 'cand',
            'model_key_prefix': 'run:cand',
            'checkpoint': '/c.pt',
            'formats': ['fp16_onnx'],
            'imgsz': 320,
            'class_map_by_dataset': {'export:d': {'0': 3}},
        },
        [{'id': 'export:d', 'path': '/data/d'}],
        tmp_path / 'job',
    )
    assert seen['model_id'] == 'cand'
    assert seen['out_root'] == tmp_path / 'job' / 'quant'
    assert seen['calib_override'] == Path('/data/d')
    assert seen['imgsz'] == 320
    [variant] = models
    assert variant['model'] == 'run:cand:fp16_onnx'
    assert variant['backend'] == 'onnxruntime'
    assert variant['class_map_by_dataset'] == {'export:d': {'0': 3}}
    assert variant['backend_options']['coords_normalized'] is False
