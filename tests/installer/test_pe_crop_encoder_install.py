"""The installer builds, stamps, loads and requires the crop encoder exactly as it does the frame
encoder: without ``pe_crop_encoder`` an install's engine readiness never turns green."""

from __future__ import annotations

import re
import shutil
from typing import TYPE_CHECKING

import pytest
from installer_harness import REPO_ROOT
from test_model_setup import LIB, ONNX_BYTES, ONNX_SHA, STAMP_STUBS, _put_onnx


if TYPE_CHECKING:
    from pathlib import Path

FRAME = 'pe_image_encoder'
CROP = 'pe_crop_encoder'
NORMALIZE = '--mean 0.5 0.5 0.5 --std 0.5 0.5 0.5'


def _config_input_size(model: str) -> int:
    text = (REPO_ROOT / 'models' / model / 'config.pbtxt').read_text()
    dims = re.search(r'dims:\s*\[\s*3,\s*(\d+),\s*(\d+)\s*\]', text)
    assert dims, model
    assert dims.group(1) == dims.group(2)
    return int(dims.group(1))


def _group_script(tmp_path: Path, *, fp16_fails_for: str = '') -> str:
    """The real `pe` group; every dc call goes to calls.log. A trtexec of the model named in
    ``fp16_fails_for`` fails until its FP32 ONNX has been staged."""
    _put_onnx(tmp_path)
    (tmp_path / 'pytorch_models' / f'{CROP}.onnx').write_bytes(ONNX_BYTES)
    for model in (FRAME, CROP):
        (tmp_path / 'models' / model).mkdir(parents=True)
        shutil.copy(REPO_ROOT / 'models' / model / 'config.pbtxt', tmp_path / 'models' / model)
    return (
        f'source {LIB}; OP_DIR="{tmp_path}"; mkdir -p "{tmp_path}/.install"; sleep() {{ :; }}; '
        'triton_load_and_wait() { echo "$1" >> "$OP_DIR/loaded.log"; return 0; }; '
        f'{STAMP_STUBS} '
        'dc() { '
        '  echo "$*" >> "$OP_DIR/calls.log"; '
        '  if [[ "$*" == *"cp /app/pytorch_models/"* ]]; then touch "$OP_DIR/fp32_staged"; fi; '
        f'  if [[ -n "{fp16_fails_for}" && "$*" == *trtexec* && "$*" == *"{fp16_fails_for}"* '
        '        && ! -e "$OP_DIR/fp32_staged" ]]; then return 1; fi; '
        '  return 0; }; '
        'model_setup_run_group pe; echo "rc=$?"'
    )


def _calls(tmp_path: Path) -> list[str]:
    return (tmp_path / 'calls.log').read_text().splitlines()


def _rows(tmp_path: Path) -> dict[str, list[str]]:
    text = (tmp_path / '.install' / 'precision.tsv').read_text()
    return {r.split('\t')[0]: r.split('\t') for r in text.splitlines()}


def test_the_pe_group_lists_the_crop_encoder_in_every_table(bash) -> None:
    out = bash(
        f'source {LIB}; OP_DIR=/x; model_setup_group_models pe; model_setup_group_outputs pe; '
        'model_setup_group_engines pe; model_setup_required_models "core curation"'
    ).stdout
    assert f'{FRAME} {CROP} pe_text_encoder' in out
    assert f'/x/models/{CROP}/1/model.plan' in out
    assert f'{CROP}\tpytorch_models/{CROP}.onnx\tfp16' in out
    assert f'{FRAME},{CROP}' in out.splitlines()[-1]
    assert 'pe_text_encoder' not in out.splitlines()[-1]


def test_the_core_tier_does_not_require_the_crop_encoder(bash) -> None:
    out = bash(f'source {LIB}; OP_DIR=/x; model_setup_required_models "core"').stdout
    assert CROP not in out


def test_the_group_exports_the_crop_onnx_through_the_documented_path(tmp_path: Path, bash) -> None:
    out = bash(_group_script(tmp_path)).stdout
    assert 'rc=0' in out
    assert any('export_pe_image_encoder.py --crop' in c for c in _calls(tmp_path))


def test_the_crop_engine_is_built_at_its_own_input_size_into_its_own_model(
    tmp_path: Path, bash
) -> None:
    bash(_group_script(tmp_path))
    size = _config_input_size(CROP)
    assert size == 224
    builds = {
        m.group(1): c
        for c in _calls(tmp_path)
        if (m := re.search(r'--saveEngine=/models/(\w+)/1/model\.plan', c))
    }
    assert set(builds) == {FRAME, CROP}
    crop = builds[CROP]
    assert f'--onnx=/models/{CROP}.build.u8.onnx' in crop
    for shape in ('min', 'opt', 'max'):
        assert re.search(rf'--{shape}Shapes=images:\d+x3x{size}x{size}\b', crop)
    assert 'x3x336x336' not in crop
    assert 'x3x336x336' in builds[FRAME]


def test_the_crop_engine_gets_the_uint8_prefix_and_the_config_datatype_check(
    tmp_path: Path, bash
) -> None:
    bash(_group_script(tmp_path))
    calls = _calls(tmp_path)
    build = next(i for i, c in enumerate(calls) if f'/models/{CROP}/1/model.plan' in c)
    before = calls[:build]
    assert any(
        'uint8_input.py' in c and NORMALIZE in c and f'{CROP}.build.u8.onnx' in c for c in before
    )
    assert any(
        'check_input_dtype.py' in c and f'/app/models/{CROP}/config.pbtxt' in c for c in before
    )


def test_both_engines_are_stamped_and_the_crop_flags_name_its_size(tmp_path: Path, bash) -> None:
    bash(_group_script(tmp_path))
    rows = _rows(tmp_path)
    assert set(rows) == {FRAME, CROP}
    for model in (FRAME, CROP):
        assert rows[model][1] == 'fp16'
        assert rows[model][3:6] == ['v110100', '8.6', ONNX_SHA]
        assert 'input=uint8' in rows[model][6]
    assert '336x336' in rows[FRAME][6]
    assert '224x224' in rows[CROP][6]


def test_a_crop_fp16_failure_falls_back_to_fp32_and_degrades_the_group(
    tmp_path: Path, bash
) -> None:
    out = bash(_group_script(tmp_path, fp16_fails_for=CROP))
    assert 'rc=0' in out.stdout
    assert '[WARN]' in out.stdout + out.stderr
    rows = _rows(tmp_path)
    assert rows[FRAME][1] == 'fp16'
    assert rows[CROP][1] == 'fp32_fallback'
    statuses = dict(
        ln.split('\t')[:2] for ln in (tmp_path / '.install' / 'groups.tsv').read_text().splitlines()
    )
    assert statuses['pe'] == 'degraded'


def test_the_group_loads_both_encoders_into_the_server(tmp_path: Path, bash) -> None:
    bash(_group_script(tmp_path))
    assert set((tmp_path / 'loaded.log').read_text().split()) == {FRAME, CROP, 'pe_text_encoder'}


def _installed(tmp_path: Path, *, crop_stamp: str | None) -> str:
    """An installed, READY PE group whose frame engine is stamped current."""
    models = tmp_path / 'models'
    for model in (FRAME, CROP, 'pe_text_encoder'):
        (models / model / '1').mkdir(parents=True, exist_ok=True)
        (models / model / '1' / 'model.plan').write_text('x')
    for model in (FRAME, CROP):
        shutil.copy(REPO_ROOT / 'models' / model / 'config.pbtxt', models / model)
    (tmp_path / '.install').mkdir(exist_ok=True)
    (tmp_path / '.install' / 'state.json').write_text('{"triton_image_digest": "sha256:abc"}')
    (tmp_path / '.install' / 'groups.tsv').write_text('pe\tok\t5\n')
    (tmp_path / 'pytorch_models').mkdir(exist_ok=True)
    for model in (FRAME, CROP):
        (tmp_path / 'pytorch_models' / f'{model}.onnx').write_bytes(ONNX_BYTES)

    def row(model: str) -> str:
        return (
            f'{model}\tfp16\t\tv110100\t8.6\t{ONNX_SHA}\t'
            'min1/opt8/max32 336x336 target=default workspace=8G strongly-typed input=uint8\n'
        )

    stamps = row(FRAME) + (crop_stamp if crop_stamp is not None else '')
    (tmp_path / '.install' / 'precision.tsv').write_text(stamps)
    return (
        f'source {LIB}; OP_DIR="{tmp_path}"; MODEL_SETUP_TRITON_DIGEST=sha256:abc; {STAMP_STUBS}'
        'triton_model_ready() { return 0; }; triton_load_and_wait() { return 0; }; '
        'dc() { return 0; }; '
    )


def test_an_install_that_has_only_the_frame_stamp_rebuilds_the_group(tmp_path: Path, bash) -> None:
    """`models repair` on an install made before the crop encoder existed must build it."""
    out = bash(_installed(tmp_path, crop_stamp=None) + 'model_setup_run_group pe').stdout
    assert 'skipped' not in out
    assert f'no build stamp; {CROP}' in out


def test_a_current_crop_stamp_lets_the_group_skip(tmp_path: Path, bash) -> None:
    stamp = (
        f'{CROP}\tfp16\t\tv110100\t8.6\t{ONNX_SHA}\t'
        'min1/opt8/max32 224x224 target=default workspace=4G strongly-typed input=uint8\n'
    )
    out = bash(_installed(tmp_path, crop_stamp=stamp) + 'model_setup_run_group pe').stdout
    assert 'up to date, skipped' in out


def test_an_unstamped_crop_engine_is_never_adopted(tmp_path: Path, bash) -> None:
    prelude = _installed(tmp_path, crop_stamp=None)
    out = bash(prelude + f'_ms_adopt_unstamped {CROP} pytorch_models/{CROP}.onnx fp16; echo rc=$?')
    assert 'rc=1' in out.stdout


@pytest.mark.parametrize('model', [FRAME, CROP])
def test_the_workspace_and_batch_of_a_pe_build_come_from_the_trt_settings(
    model: str, bash, tmp_path: Path
) -> None:
    out = bash(
        f'source {LIB}; OP_DIR="{tmp_path}"; GPU_PROFILE=small; '
        f'_ms_triton() {{ echo "TRTEXEC $*"; }}; _ms_pe_trtexec {model} x.onnx; _ms_pe_flags {model}'
    ).stdout
    size = _config_input_size(model)
    assert '--memPoolSize=workspace:2G' in out
    assert f'--maxShapes=images:16x3x{size}x{size}' in out
    assert f'min1/opt8/max16 {size}x{size} target=small workspace=2G' in out


def test_the_build_script_and_the_installer_agree_on_each_input_size() -> None:
    build = (REPO_ROOT / 'export/build_pe_trt.sh').read_text()
    for role, model in (('frame', FRAME), ('crop', CROP)):
        m = re.search(rf'{role}\)\s+MODEL_NAME="{model}";\s+DEFAULT_IMAGE_SIZE=(\d+)', build)
        assert m, role
        assert int(m.group(1)) == _config_input_size(model), role
