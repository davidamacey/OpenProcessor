"""The installer's PE engine build must produce the input datatype models/pe_image_encoder/config.pbtxt
declares (uint8) and fail at install time when it does not, and a model install in a checkout must
not die silently."""

from __future__ import annotations

import re
import shutil
from typing import TYPE_CHECKING

import pytest
from installer_harness import REPO_ROOT
from test_model_setup import LIB, ONNX_SHA, STAMP_STUBS, _installed_pe, _put_onnx, _stamp_row


if TYPE_CHECKING:
    from pathlib import Path


NORMALIZE = '--mean 0.5 0.5 0.5 --std 0.5 0.5 0.5'
FLAGS_OK = 'min1/opt8/max32 336x336 workspace8G strongly-typed input=uint8'


def _build_script(tmp_path: Path, *, fp16_fails: bool, check_fails: bool = False) -> str:
    """The real `pe` group; every dc call is appended to calls.log."""
    _put_onnx(tmp_path)
    (tmp_path / 'models' / 'pe_image_encoder').mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / 'models/pe_image_encoder/config.pbtxt', tmp_path / 'models/pe_image_encoder'
    )
    return (
        f'source {LIB}; OP_DIR="{tmp_path}"; mkdir -p "{tmp_path}/.install"; sleep() {{ :; }}; '
        f'triton_load_and_wait() {{ return 0; }}; {STAMP_STUBS} '
        'dc() { '
        '  echo "$*" >> "$OP_DIR/calls.log"; '
        '  if [[ "$*" == *"cp /app/pytorch_models/pe_image_encoder.onnx"* ]]; then touch "$OP_DIR/fp32_staged"; fi; '
        f'  if [[ "$*" == *trtexec* && {"true" if fp16_fails else "false"} == true && ! -e "$OP_DIR/fp32_staged" ]]; then return 1; fi; '
        f'  if [[ "$*" == *check_input_dtype* && {"true" if check_fails else "false"} == true ]]; then '
        '    echo "input datatype mismatch" >&2; return 1; fi; '
        '  return 0; }; '
        'model_setup_run_group pe; echo "rc=$?"'
    )


def _calls(tmp_path: Path) -> list[str]:
    return (tmp_path / 'calls.log').read_text().splitlines()


def _trtexec_inputs(calls: list[str]) -> list[str]:
    return [m.group(1) for c in calls if (m := re.search(r'trtexec --onnx=/models/(\S+)', c))]


def _assert_prefix_before_every_trtexec(calls: list[str], expected_builds: int) -> None:
    builds = [i for i, c in enumerate(calls) if 'trtexec' in c and '--onnx=' in c]
    assert len(builds) == expected_builds
    for build in builds:
        before = calls[:build]
        assert any(
            'uint8_input.py' in c and NORMALIZE in c and 'pe_image_encoder.build.u8.onnx' in c
            for c in before
        ), calls
        assert any('check_input_dtype.py' in c for c in before), calls
    assert set(_trtexec_inputs(calls)) == {'pe_image_encoder.build.u8.onnx'}


def test_fp16_branch_applies_the_uint8_prefix_before_trtexec(tmp_path: Path, bash) -> None:
    result = bash(_build_script(tmp_path, fp16_fails=False))
    assert 'rc=0' in result.stdout
    calls = _calls(tmp_path)
    assert any('trt_utils.py' in c for c in calls)  # the bake runs first
    _assert_prefix_before_every_trtexec(calls, 1)
    bake = next(i for i, c in enumerate(calls) if 'trt_utils.py' in c)
    assert bake < next(i for i, c in enumerate(calls) if 'uint8_input.py' in c)


def test_fp32_fallback_branch_applies_the_uint8_prefix_too(tmp_path: Path, bash) -> None:
    result = bash(_build_script(tmp_path, fp16_fails=True))
    assert 'rc=0' in result.stdout
    _assert_prefix_before_every_trtexec(_calls(tmp_path), 2)
    row = (tmp_path / '.install' / 'precision.tsv').read_text().rstrip().split('\t')
    assert row[1] == 'fp32_fallback'
    assert row[6] == FLAGS_OK


def test_clean_build_stamp_records_the_input_datatype(tmp_path: Path, bash) -> None:
    bash(_build_script(tmp_path, fp16_fails=False))
    row = (tmp_path / '.install' / 'precision.tsv').read_text().rstrip().split('\t')
    assert row[1] == 'fp16'
    assert row[5] == ONNX_SHA
    assert row[6] == FLAGS_OK


def test_a_datatype_mismatch_fails_the_install_and_records_no_stamp(tmp_path: Path, bash) -> None:
    result = bash(_build_script(tmp_path, fp16_fails=False, check_fails=True))
    out = result.stdout + result.stderr
    assert 'rc=0' not in out
    logs = ''.join(p.read_text() for p in (tmp_path / '.install' / 'logs').glob('pe_image_trt*'))
    assert 'input datatype mismatch' in out + logs
    assert 'would not match config.pbtxt' in out + logs
    assert not (tmp_path / '.install' / 'precision.tsv').exists()
    assert not any('trtexec' in c and '--onnx=' in c for c in _calls(tmp_path))


def _with_config(tmp_path: Path) -> None:
    cfg = tmp_path / 'models' / 'pe_image_encoder' / 'config.pbtxt'
    cfg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / 'models/pe_image_encoder/config.pbtxt', cfg)


def test_a_stamp_without_the_config_datatype_is_stale(tmp_path: Path, bash) -> None:
    prelude = _installed_pe(tmp_path, stamp=_stamp_row())  # flags: "flags", the pre-fix shape
    _with_config(tmp_path)
    result = bash(
        prelude + 'model_setup_stamp_stale pe_image_encoder pytorch_models/pe_image_encoder.onnx'
    )
    assert result.returncode == 0
    assert 'input datatype is not the config' in result.stdout


def test_a_stamp_with_the_config_datatype_is_current(tmp_path: Path, bash) -> None:
    stamp = _stamp_row().replace('\tflags\n', f'\t{FLAGS_OK}\n')
    prelude = _installed_pe(tmp_path, stamp=stamp)
    _with_config(tmp_path)
    result = bash(
        prelude + 'model_setup_stamp_stale pe_image_encoder pytorch_models/pe_image_encoder.onnx'
    )
    assert result.returncode == 1


def test_installer_and_build_script_use_the_same_normalization() -> None:
    build = (REPO_ROOT / 'export/build_pe_trt.sh').read_text()
    lib = (REPO_ROOT / 'scripts/lib/model_setup.sh').read_text()
    assert f'PE_NORMALIZE="{NORMALIZE}"' in build
    assert f'_MS_PE_NORMALIZE="{NORMALIZE}"' in lib


# ---- the check script itself, on real ONNX files ----


def _float_onnx(path: Path) -> None:
    onnx = pytest.importorskip('onnx')
    from onnx import TensorProto, helper

    node = helper.make_node('Identity', ['images'], ['out'])
    graph = helper.make_graph(
        [node],
        'g',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, ['b', 3, 8, 8])],
        [helper.make_tensor_value_info('out', TensorProto.FLOAT, ['b', 3, 8, 8])],
    )
    onnx.save(helper.make_model(graph), str(path))


def _run_check(onnx_path: Path, config: Path):
    import subprocess
    import sys

    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / 'export/check_input_dtype.py'),
            str(onnx_path),
            str(config),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_input_dtype_rejects_a_float_onnx_for_a_uint8_config(tmp_path: Path) -> None:
    _float_onnx(tmp_path / 'f.onnx')
    done = _run_check(tmp_path / 'f.onnx', REPO_ROOT / 'models/pe_image_encoder/config.pbtxt')
    assert done.returncode == 1
    assert 'TYPE_FP32' in done.stderr
    assert 'TYPE_UINT8' in done.stderr


def test_check_input_dtype_accepts_the_prefixed_onnx(tmp_path: Path) -> None:
    pytest.importorskip('onnx')
    import subprocess
    import sys

    _float_onnx(tmp_path / 'f.onnx')
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / 'export/uint8_input.py'),
            str(tmp_path / 'f.onnx'),
            str(tmp_path / 'u.onnx'),
            *NORMALIZE.split(),
        ],
        check=True,
        capture_output=True,
    )
    done = _run_check(tmp_path / 'u.onnx', REPO_ROOT / 'models/pe_image_encoder/config.pbtxt')
    assert done.returncode == 0, done.stderr


# ---- every shipped config that declares a uint8 engine input has a build that applies the prefix ----

# config directory -> the sources that build its engine
UINT8_BUILDERS = {
    'pe_image_encoder': ['export/build_pe_trt.sh', 'scripts/lib/model_setup.sh'],
    'yolov11_small_trt_end2end': ['export/yolo_trt_build.py'],
    'yolo26_small_trt': ['export/export_yolo26.py'],
}


def _uint8_tensorrt_configs() -> set[str]:
    found = set()
    for cfg in (REPO_ROOT / 'models').glob('*/config.pbtxt'):
        text = cfg.read_text()
        block = re.search(r'^input\s*\[(.*?)^\]', text, re.DOTALL | re.MULTILINE)
        if 'tensorrt_plan' in text and block and 'TYPE_UINT8' in block.group(1):
            found.add(cfg.parent.name)
    return found


def test_every_uint8_tensorrt_config_has_a_known_builder() -> None:
    assert _uint8_tensorrt_configs() == set(UINT8_BUILDERS)


@pytest.mark.parametrize('model', sorted(UINT8_BUILDERS))
def test_each_uint8_engine_builder_applies_the_prefix(model: str) -> None:
    for rel in UINT8_BUILDERS[model]:
        text = (REPO_ROOT / rel).read_text()
        assert re.search(r'uint8_input(\.py)?\b|add_uint8_input', text), (
            f'{rel} never adds the uint8 input'
        )


def test_dual_head_exporter_declares_uint8_and_applies_the_prefix() -> None:
    assert 'TYPE_UINT8' in (REPO_ROOT / 'export/dual_head_triton.py').read_text()
    assert 'add_uint8_input(' in (REPO_ROOT / 'export/dual_head_model.py').read_text()


# ---- #270: models install in a checkout without .install/state.json ----


def test_models_install_in_a_checkout_without_state_json_does_not_exit_silently(
    shimmed, tmp_path
) -> None:
    checkout = tmp_path / 'checkout'
    (checkout / 'src').mkdir(parents=True)
    (checkout / 'src' / 'main.py').write_text('')
    shutil.copytree(REPO_ROOT / 'scripts' / 'lib', checkout / 'scripts' / 'lib')
    shutil.copy2(REPO_ROOT / 'openprocessor', checkout / 'openprocessor')
    for name in ('docker-compose.yml', 'docker-compose.dev.yml'):
        (checkout / name).write_text('services: {}\n')
    (checkout / '.env').write_text('COMPOSE_PROJECT_NAME=opcheckout-test\n')
    shimmed.flag('allow_mutations')
    import subprocess

    done = subprocess.run(
        [str(checkout / 'openprocessor'), 'models', 'install', '--only', 'preflight'],
        check=False,
        cwd=str(checkout),
        env=shimmed.env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = done.stdout + done.stderr
    assert 'no .install/state.json' in out
    assert 'docker compose' in shimmed.log.read_text()  # it went on to act, not died at the sed
