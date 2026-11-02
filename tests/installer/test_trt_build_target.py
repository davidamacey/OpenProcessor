"""Build target, workspace and the engine stamp (#42).

An engine stamp records the TensorRT build target and workspace; a changed
target or workspace rebuilds, a profile switch that leaves them alone does not.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from test_model_setup import LIB, STAMP_STUBS


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'export'))
import trt_build_settings as tbs  # noqa: E402  # path set up above


GB = 1 << 30
MODEL = 'arcface_w600k_r50'
ONNX_REL = 'pytorch_models/arcface_w600k_r50.onnx'
ONNX_BYTES = b'onnx-of-arcface'
SHA = hashlib.sha256(ONNX_BYTES).hexdigest()


def _pre(tmp_path: Path, flags: str) -> str:
    """A shell prelude with a stamp for ``MODEL`` carrying ``flags``."""
    (tmp_path / '.install').mkdir(exist_ok=True)
    (tmp_path / ONNX_REL).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ONNX_REL).write_bytes(ONNX_BYTES)
    row = f'{MODEL}\tfp16\t\tv110100\t8.6\t{SHA}\t{flags}\n'
    (tmp_path / '.install' / 'precision.tsv').write_text(row)
    return f'source {LIB}; OP_DIR="{tmp_path}"; {STAMP_STUBS}'


def _stale(bash, tmp_path: Path, flags: str, env: str = '') -> tuple[bool, str]:
    out = bash(
        _pre(tmp_path, flags)
        + f'{env or ":"}; if why="$(model_setup_stamp_stale {MODEL} {ONNX_REL})"; '
        + 'then echo "STALE:$why"; else echo CURRENT; fi'
    ).stdout.strip()
    assert out.startswith(('STALE', 'CURRENT')), out
    return out.startswith('STALE'), out


@pytest.mark.parametrize('target', ['default', 'small'])
@pytest.mark.parametrize('model', ['yolo', 'pe_image_encoder', 'paddleocr_rec_trt', 'arcface'])
def test_shell_workspace_table_agrees_with_python(
    bash, monkeypatch: pytest.MonkeyPatch, target: str, model: str
) -> None:
    out = bash(
        f'source {LIB}; OP_TRT_BUILD_TARGET={target} trt_workspace_gb {model}'
    ).stdout.strip()
    monkeypatch.setenv('OP_TRT_BUILD_TARGET', target)
    assert int(out) * GB == tbs.workspace_bytes(model)


@pytest.mark.parametrize('default', [128, 64, 32, 16, 8, 4])
@pytest.mark.parametrize('target', ['default', 'small'])
def test_shell_max_batch_agrees_with_python(
    bash, monkeypatch: pytest.MonkeyPatch, target: str, default: int
) -> None:
    out = bash(
        f'source {LIB}; OP_TRT_BUILD_TARGET={target} trt_max_batch pe_image_encoder {default}'
    ).stdout.strip()
    monkeypatch.setenv('OP_TRT_BUILD_TARGET', target)
    assert int(out) == tbs.max_batch('pe_image_encoder', default)


@pytest.mark.parametrize('bad', ['0', 'x', '-3', '2G'])
def test_shell_rejects_a_bad_workspace(bash, bad: str) -> None:
    out = bash(f'source {LIB}; OP_TRT_WORKSPACE_GB={bad} trt_workspace_gb yolo; echo "rc=$?"')
    assert 'rc=1' in out.stdout


def test_shell_rejects_a_bad_target(bash) -> None:
    out = bash(f'source {LIB}; OP_TRT_BUILD_TARGET=huge trt_target; echo "rc=$?"')
    assert 'rc=1' in out.stdout


def test_a_matching_default_stamp_is_current(bash, tmp_path: Path) -> None:
    stale, out = _stale(bash, tmp_path, 'trt-python-api fp16 target=default workspace=4G')
    assert not stale, out


def test_a_default_stamp_is_stale_for_the_small_target(bash, tmp_path: Path) -> None:
    stale, out = _stale(
        bash,
        tmp_path,
        'trt-python-api fp16 target=default workspace=4G',
        env='OP_TRT_BUILD_TARGET=small',
    )
    assert stale
    assert 'built for target default, now small' in out


def test_the_small_profile_alone_makes_a_default_stamp_stale(bash, tmp_path: Path) -> None:
    stale, out = _stale(
        bash, tmp_path, 'trt-python-api fp16 target=default workspace=4G', env='GPU_PROFILE=small'
    )
    assert stale
    assert 'now small' in out


def test_a_small_stamp_is_current_under_the_small_profile(bash, tmp_path: Path) -> None:
    stale, out = _stale(
        bash, tmp_path, 'trt-python-api fp16 target=small workspace=2G', env='GPU_PROFILE=small'
    )
    assert not stale, out


@pytest.mark.parametrize('profile', ['minimal', 'standard', 'full'])
def test_switching_between_the_other_profiles_rebuilds_nothing(
    bash, tmp_path: Path, profile: str
) -> None:
    stale, out = _stale(
        bash,
        tmp_path,
        'trt-python-api fp16 target=default workspace=4G',
        env=f'GPU_PROFILE={profile}',
    )
    assert not stale, out


def test_a_changed_workspace_override_is_stale(bash, tmp_path: Path) -> None:
    stale, out = _stale(
        bash,
        tmp_path,
        'trt-python-api fp16 target=default workspace=4G',
        env='OP_TRT_WORKSPACE_GB=3',
    )
    assert stale
    assert 'built with workspace 4G, now 3G' in out


def test_an_old_format_stamp_is_current_under_the_default_target(bash, tmp_path: Path) -> None:
    """Stamps written before #42 were built with the default target and workspace."""
    stale, out = _stale(bash, tmp_path, 'trt-python-api fp16')
    assert not stale, out


def test_an_old_format_stamp_is_stale_for_the_small_target(bash, tmp_path: Path) -> None:
    stale, out = _stale(bash, tmp_path, 'trt-python-api fp16', env='OP_TRT_BUILD_TARGET=small')
    assert stale
    assert 'built for target default, now small' in out


def test_an_old_format_stamp_is_stale_for_another_workspace(bash, tmp_path: Path) -> None:
    stale, out = _stale(bash, tmp_path, 'trt-python-api fp16', env='OP_TRT_WORKSPACE_GB=3')
    assert stale
    assert 'built with workspace 4G, now 3G' in out


def test_an_old_format_pe_stamp_keeps_its_8g_default(bash, tmp_path: Path) -> None:
    (tmp_path / '.install').mkdir()
    (tmp_path / 'pytorch_models').mkdir()
    (tmp_path / 'pytorch_models' / 'pe_image_encoder.onnx').write_bytes(ONNX_BYTES)
    row = f'pe_image_encoder\tfp16\t\tv110100\t8.6\t{SHA}\tmin1/opt8/max32 336x336 workspace8G strongly-typed\n'
    (tmp_path / '.install' / 'precision.tsv').write_text(row)
    out = bash(
        f'source {LIB}; OP_DIR="{tmp_path}"; {STAMP_STUBS} '
        'if model_setup_stamp_stale pe_image_encoder pytorch_models/pe_image_encoder.onnx; '
        'then echo STALE; else echo CURRENT; fi'
    ).stdout
    assert 'CURRENT' in out


def test_the_recorded_stamp_carries_target_and_workspace(bash, tmp_path: Path) -> None:
    (tmp_path / '.install').mkdir()
    (tmp_path / ONNX_REL).parent.mkdir(parents=True)
    (tmp_path / ONNX_REL).write_bytes(ONNX_BYTES)
    bash(
        f'source {LIB}; OP_DIR="{tmp_path}"; GPU_PROFILE=small; {STAMP_STUBS} '
        '_ms_config_input_dtype() { echo FP32; }; _ms_record_group_stamps faces'
    )
    rows = (tmp_path / '.install' / 'precision.tsv').read_text()
    assert 'target=small workspace=2G' in rows


def test_the_pe_stamp_and_the_trtexec_call_come_from_one_source(bash, tmp_path: Path) -> None:
    out = bash(
        f'source {LIB}; OP_DIR="{tmp_path}"; GPU_PROFILE=small; '
        '_ms_triton() { echo "TRTEXEC $*"; }; _ms_pe_trtexec x.onnx; _ms_pe_flags'
    ).stdout
    assert '--memPoolSize=workspace:2G' in out
    assert '--maxShapes=images:16x3x336x336' in out
    assert 'min1/opt8/max16 336x336 target=small workspace=2G' in out


def test_the_default_pe_build_is_unchanged(bash, tmp_path: Path) -> None:
    out = bash(
        f'source {LIB}; OP_DIR="{tmp_path}"; '
        '_ms_triton() { echo "TRTEXEC $*"; }; _ms_pe_trtexec x.onnx'
    ).stdout
    assert '--memPoolSize=workspace:8G' in out
    assert '--maxShapes=images:32x3x336x336' in out


def test_export_containers_receive_the_target_and_workspace(bash, tmp_path: Path) -> None:
    out = bash(
        f'source {LIB}; OP_DIR="{tmp_path}"; GPU_PROFILE=small; OP_TRT_WORKSPACE_GB=3; '
        'dc() { echo "DC $*"; }; _ms_trt_env; _ms_export python x.py'
    ).stdout
    assert '-e OP_TRT_BUILD_TARGET=small' in out
    assert '-e OP_TRT_WORKSPACE_GB=3' in out
    assert out.index('-e OP_TRT_BUILD_TARGET') < out.index('model-export')


def test_the_oom_hint_names_the_small_target(bash) -> None:
    out = bash(f'source {LIB}; permanent_hint permanent:oom').stdout
    assert '--build-target small' in out


def test_the_stamp_reads_the_card_that_built_the_engine(bash, tmp_path: Path) -> None:
    """Python engines build on API_GPU_ID, the PE trtexec on TRITON_GPU_ID (J6)."""
    out = bash(
        f'source {LIB}; OP_DIR="{tmp_path}"; TRITON_GPU_ID=1,2; API_GPU_ID=0; '
        'nvidia-smi() { echo "$*" >> "$OP_DIR/smi"; echo 8.6; }; '
        f'_ms_compute_cap {MODEL} >/dev/null; _ms_compute_cap pe_image_encoder >/dev/null; '
        f'cat "$OP_DIR/smi"'
    ).stdout.splitlines()
    assert out[0].endswith('-i 0')
    assert out[1].endswith('-i 1')


def test_a_pre_stamp_engine_is_not_adopted_for_a_non_default_target(bash, tmp_path: Path) -> None:
    (tmp_path / '.install').mkdir()
    (tmp_path / ONNX_REL).parent.mkdir(parents=True)
    (tmp_path / ONNX_REL).write_bytes(ONNX_BYTES)
    plan = tmp_path / 'models' / MODEL / '1' / 'model.plan'
    plan.parent.mkdir(parents=True)
    plan.write_text('x')
    (tmp_path / 'models' / MODEL / 'config.pbtxt').write_text(
        'input [\n  {\n    name: "x"\n    data_type: TYPE_FP32\n  }\n]\n'
    )
    script = (
        f'source {LIB}; OP_DIR="{tmp_path}"; {STAMP_STUBS} triton_model_ready() {{ return 0; }}; '
    )
    assert (
        'ADOPTED'
        in bash(script + f'_ms_adopt_unstamped {MODEL} {ONNX_REL} fp16 && echo ADOPTED').stdout
    )
    (tmp_path / '.install' / 'precision.tsv').unlink()
    assert (
        'ADOPTED'
        not in bash(
            script
            + f'GPU_PROFILE=small; _ms_adopt_unstamped {MODEL} {ONNX_REL} fp16 && echo ADOPTED'
        ).stdout
    )


CLI = ROOT / 'scripts' / 'lib' / 'cli_core.sh'


def test_models_install_flags_reach_the_environment(bash) -> None:
    out = bash(
        f'source {CLI}; cmd_models_install() {{ echo "only=$1 target=$OP_TRT_BUILD_TARGET ws=$OP_TRT_WORKSPACE_GB"; }}; '
        'cmd_models install --only faces --build-target small --workspace 3'
    ).stdout
    assert 'only=faces target=small ws=3' in out


def test_models_install_rejects_an_unknown_flag(bash) -> None:
    out = bash(f'source {CLI}; cmd_models install --frobnicate; echo "rc=$?"')
    assert 'rc=2' in out.stdout


def test_env_template_documents_the_knobs() -> None:
    text = (ROOT / 'env.template').read_text()
    assert 'OP_TRT_BUILD_TARGET' in text
    assert 'OP_TRT_WORKSPACE_GB' in text


def test_check_gpu_memory_reports_free_not_total(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """The export check logged the card's total as "free"; stub the toolchain it imports."""
    from unittest.mock import MagicMock

    for name in (
        'tensorrt',
        'ultralytics',
        'ultralytics.cfg',
        'ultralytics.engine',
        'ultralytics.engine.exporter',
    ):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    for name in ('yolo_formats', 'yolo_trt_build'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import yolo_formats

    torch = yolo_formats.torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda _d: (1 * GB, 48 * GB), raising=False)
    monkeypatch.setattr(
        torch.cuda,
        'get_device_properties',
        lambda _d: type('P', (), {'total_memory': 48 * GB})(),
    )
    with caplog.at_level('WARNING'):
        assert yolo_formats.check_gpu_memory(4.0) is True
    assert any('Low GPU memory: 1.0GB free' in r.message for r in caplog.records)
