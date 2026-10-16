"""``export/build_pe_trt.sh`` (``make pe-build-trt``) never ships an FP32 engine silently (#277).

The FP16 bake or the FP16 trtexec build failing is a hard failure unless the caller
says ``ALLOW_FP32=1``; the uint8 input prefix is always applied; stubs stand in for
python and trtexec so no GPU is needed.
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

import pytest
from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path

SCRIPT = REPO_ROOT / 'export' / 'build_pe_trt.sh'

PY_STUB = """#!/bin/bash
# stands in for the venv python: dispatches on the script basename
name="$(basename "$1")"
echo "py $*" >> "$STUB_LOG"
case "$name" in
  trt_utils.py)
    if [ "${STUB_BAKE:-fp16}" = "crash" ]; then echo "boom" >&2; exit 1; fi
    if [ "${STUB_BAKE:-fp16}" = "fp16" ]; then echo FP16MARK > "$3"; else cp "$2" "$3"; fi; echo "${STUB_BAKE:-fp16}" >&2; echo "$3" ;;
  uint8_input.py)
    if [ "${STUB_U8:-ok}" = "fail" ]; then echo "no uint8" >&2; exit 1; fi
    cp "$2" "$3"; echo "$3" ;;
  *) : ;;
esac
"""

TRTEXEC_STUB = """#!/bin/bash
echo "trtexec $*" >> "$STUB_LOG"
onnx=""
for a in "$@"; do case "$a" in --onnx=*) onnx="${a#--onnx=}";; --saveEngine=*) plan="${a#--saveEngine=}";; esac; done
if [ "${STUB_TRT:-ok}" = "fp16_fails" ] && grep -q FP16MARK "$onnx"; then exit 1; fi
echo plan > "$plan"
"""


def _run(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess[str], str]:
    onnx = tmp_path / 'pe_image_encoder.onnx'
    onnx.write_text('FP32')
    py = tmp_path / 'py'
    py.write_text(PY_STUB)
    trt = tmp_path / 'trtexec'
    trt.write_text(TRTEXEC_STUB)
    py.chmod(0o755)
    trt.chmod(0o755)
    log = tmp_path / 'calls.log'
    log.touch()
    full = {
        **os.environ,
        'ONNX_PATH': str(onnx),
        'MODELS_DIRS': str(tmp_path / 'models'),
        'PYTHON': str(py),
        'TRTEXEC': str(trt),
        'STUB_LOG': str(log),
        **env,
    }
    done = subprocess.run(
        ['bash', str(SCRIPT)], env=full, capture_output=True, text=True, check=False
    )
    return done, log.read_text()


def test_clean_fp16_build_installs_the_plan_with_uint8_input(tmp_path: Path) -> None:
    done, calls = _run(tmp_path)
    assert done.returncode == 0, done.stderr
    assert 'uint8_input.py' in calls
    assert (tmp_path / 'models/pe_image_encoder/1/model.plan').exists()
    assert 'fp16' in done.stdout


def test_fp16_trtexec_failure_fails_loudly_and_installs_nothing(tmp_path: Path) -> None:
    done, calls = _run(tmp_path, STUB_TRT='fp16_fails')
    assert done.returncode != 0
    assert 'FP16' in done.stderr
    assert 'ALLOW_FP32=1' in done.stderr
    assert (
        calls.count('\ntrtexec ') + calls.startswith('trtexec ') == 1
    )  # no silent second, FP32 build
    assert not (tmp_path / 'models/pe_image_encoder/1/model.plan').exists()


def test_bake_falling_back_to_fp32_fails_loudly(tmp_path: Path) -> None:
    done, _ = _run(tmp_path, STUB_BAKE='fp32')
    assert done.returncode != 0
    assert 'ALLOW_FP32=1' in done.stderr
    assert not (tmp_path / 'models/pe_image_encoder/1/model.plan').exists()


def test_uint8_prefix_failure_fails_loudly(tmp_path: Path) -> None:
    done, calls = _run(tmp_path, STUB_U8='fail')
    assert done.returncode != 0
    assert 'uint8' in done.stderr
    assert 'trtexec ' not in calls


@pytest.mark.parametrize('extra', [{'STUB_TRT': 'fp16_fails'}, {'STUB_BAKE': 'fp32'}])
def test_explicit_opt_in_still_builds_fp32_and_says_so(
    tmp_path: Path, extra: dict[str, str]
) -> None:
    done, _ = _run(tmp_path, ALLOW_FP32='1', **extra)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / 'models/pe_image_encoder/1/model.plan').exists()
    assert 'FP32' in done.stderr
