"""PE text tower served from a TensorRT plan: config rendering (offline, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'export'))

import pe_text_constants
import pe_text_engine as engine
import pe_text_triton_config as cfgmod


def test_default_platform_is_a_tensorrt_plan_on_gpu() -> None:
    out = cfgmod.render_config(cfgmod.PETextTritonConfig())
    assert 'platform: "tensorrt_plan"' in out
    assert 'kind: KIND_GPU' in out
    assert 'onnxruntime' not in out.replace('onnxruntime_onnx fallback', '')


def test_tensor_contract_is_unchanged_by_the_platform() -> None:
    for platform in ('tensorrt_plan', 'onnxruntime_onnx'):
        out = cfgmod.render_config(
            cfgmod.PETextTritonConfig(platform=platform, kind='KIND_CPU')
            if platform == 'onnxruntime_onnx'
            else cfgmod.PETextTritonConfig(platform=platform)
        )
        assert f'name: "{pe_text_constants.INPUT_TENSOR}"' in out
        assert 'data_type: TYPE_INT64' in out
        assert 'dims: [ -1 ]' in out
        assert f'name: "{pe_text_constants.OUTPUT_TENSOR}"' in out
        assert 'data_type: TYPE_FP32' in out


def test_ort_fallback_may_run_on_cpu() -> None:
    out = cfgmod.render_config(
        cfgmod.PETextTritonConfig(platform='onnxruntime_onnx', kind='KIND_CPU')
    )
    assert 'platform: "onnxruntime_onnx"' in out
    assert 'kind: KIND_CPU' in out


def test_tensorrt_plan_refuses_a_cpu_instance() -> None:
    with pytest.raises(ValueError, match='KIND_GPU'):
        cfgmod.PETextTritonConfig(platform='tensorrt_plan', kind='KIND_CPU')


def test_unknown_platform_is_refused() -> None:
    with pytest.raises(ValueError, match='platform'):
        cfgmod.PETextTritonConfig(platform='pytorch_libtorch')


def test_model_file_name_follows_the_platform() -> None:
    assert cfgmod.model_file_name('tensorrt_plan') == 'model.plan'
    assert cfgmod.model_file_name('onnxruntime_onnx') == 'model.onnx'


# --- engine profile and parity gate -----------------------------------------


def test_profile_is_batch_one_hot_with_dynamic_tokens() -> None:
    assert engine.profile_shapes(32) == {'min': (1, 2), 'opt': (1, 12), 'max': (32, 32)}
    assert engine.profile_shapes(16)['max'] == (16, pe_text_constants.CONTEXT_LENGTH)


def test_gate_prompts_are_distinct_and_numerous() -> None:
    prompts = engine.gate_prompts()
    assert len(prompts) == len(set(prompts)) >= 100


def _rows(n: int = 50) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(n, 8))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    return ref, [np.zeros((1, 3), dtype=np.int64) + i for i in range(n)]


def test_gate_refuses_a_failing_engine_and_does_not_install() -> None:
    ref, rows = _rows()
    bad = ref + np.random.default_rng(1).normal(scale=0.8, size=ref.shape)
    installed: list[bytes] = []
    with pytest.raises(engine.EngineGateError, match='not installing'):
        engine.build_and_gate(
            build=lambda: b'plan',
            run_engine=lambda _p, r: [bad[i : i + 1] for i in range(len(r))],
            reference=lambda row: ref[int(row[0, 0]) : int(row[0, 0]) + 1],
            token_rows=rows,
            install=lambda p: installed.append(p),  # type: ignore[arg-type,return-value]
        )
    assert installed == []


def test_gate_installs_a_matching_engine() -> None:
    ref, rows = _rows()
    noisy = ref + np.random.default_rng(2).normal(scale=1e-4, size=ref.shape)
    installed: list[bytes] = []
    report = engine.build_and_gate(
        build=lambda: b'plan',
        run_engine=lambda _p, r: [noisy[i : i + 1] for i in range(len(r))],
        reference=lambda row: ref[int(row[0, 0]) : int(row[0, 0]) + 1],
        token_rows=rows,
        install=lambda p: installed.append(p),  # type: ignore[arg-type,return-value]
    )
    assert report.passes()
    assert installed == [b'plan']


def test_gate_treats_nan_as_failure() -> None:
    ref, _ = _rows()
    bad = ref.copy()
    bad[3, 0] = np.nan
    assert not engine.compare(ref, bad).passes()
