"""Engine build stamps for the yolo, mobileclip, faces and ocr groups (#415).

Same contract as the PE group (test_model_setup.py): a build records a stamp
per engine, and a stamp that no longer matches the host or the ONNX makes the
group rebuild instead of being skipped as up to date.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
from test_model_setup import LIB, STAMP_STUBS


if TYPE_CHECKING:
    from pathlib import Path


# group -> {engine model: (ONNX under OP_DIR, precision the build is designed for)}
ENGINES: dict[str, dict[str, tuple[str, str]]] = {
    'yolo': {
        'yolov11_small_trt_end2end': ('models/yolov11_small_end2end/1/model.onnx', 'fp16'),
    },
    'mobileclip': {
        'mobileclip2_s2_image_encoder': (
            'pytorch_models/mobileclip2_s2_image_encoder.onnx',
            'fp16',
        ),
        'mobileclip2_s2_text_encoder': ('pytorch_models/mobileclip2_s2_text_encoder.onnx', 'fp32'),
    },
    'faces': {
        'scrfd_10g_bnkps': ('pytorch_models/scrfd_10g_bnkps.onnx', 'fp16'),
        'arcface_w600k_r50': ('pytorch_models/arcface_w600k_r50.onnx', 'fp16'),
    },
    'ocr': {
        'paddleocr_det_trt': ('pytorch_models/paddleocr/ppocr_det_v5_mobile.onnx', 'fp32'),
        'paddleocr_rec_trt': ('pytorch_models/paddleocr/ppocr_rec_v5_mobile.onnx', 'fp32'),
    },
}
GROUP_MODELS = {
    'yolo': ['yolov11_small_trt_end2end'],
    'mobileclip': ['mobileclip2_s2_image_encoder', 'mobileclip2_s2_text_encoder'],
    'faces': ['scrfd_10g_bnkps', 'arcface_w600k_r50'],
    'ocr': ['paddleocr_det_trt', 'paddleocr_rec_trt', 'ocr_pipeline'],
}
PLANS = {
    'yolo': ['yolov11_small_trt_end2end'],
    'mobileclip': ['mobileclip2_s2_image_encoder', 'mobileclip2_s2_text_encoder'],
    'faces': ['scrfd_10g_bnkps', 'arcface_w600k_r50'],
    'ocr': ['paddleocr_det_trt', 'paddleocr_rec_trt'],
}
GROUPS = list(ENGINES)


def _onnx_bytes(model: str) -> bytes:
    return f'onnx-of-{model}'.encode()


def _sha(model: str) -> str:
    return hashlib.sha256(_onnx_bytes(model)).hexdigest()


def _put_onnx(tmp_path: Path, group: str) -> None:
    for model, (rel, _) in ENGINES[group].items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_onnx_bytes(model))


def _row(model: str, precision: str, *, cc: str = '8.6', sha: str | None = None) -> str:
    ws = 8 if model == 'paddleocr_rec_trt' else 4
    flags = f'trt-python-api target=default workspace={ws}G'
    return f'{model}\t{precision}\t\tv110100\t{cc}\t{sha or _sha(model)}\t{flags}\n'


def _good_stamps(group: str) -> dict[str, str]:
    return {m: _row(m, prec) for m, (_, prec) in ENGINES[group].items()}


def _installed(tmp_path: Path, group: str, stamps: dict[str, str]) -> str:
    """An installed, READY, up-to-date-looking group; returns the bash prelude."""
    for model in PLANS[group]:
        plan = tmp_path / 'models' / model / '1' / 'model.plan'
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text('x')
    (tmp_path / '.install').mkdir(exist_ok=True)
    (tmp_path / '.install' / 'state.json').write_text('{"triton_image_digest": "sha256:abc"}')
    (tmp_path / '.install' / 'groups.tsv').write_text(f'{group}\tok\t5\n')
    (tmp_path / '.install' / 'precision.tsv').write_text(''.join(stamps.values()))
    _put_onnx(tmp_path, group)
    return (
        f'source {LIB}; OP_DIR="{tmp_path}"; MODEL_SETUP_TRITON_DIGEST=sha256:abc; {STAMP_STUBS}'
        'triton_model_ready() { return 0; }; triton_load_and_wait() { return 0; }; '
        'dc() { return 0; }; '
    )


@pytest.mark.parametrize('group', GROUPS)
def test_build_records_a_full_stamp_for_every_engine(tmp_path: Path, bash, group: str) -> None:
    _put_onnx(tmp_path, group)
    (tmp_path / '.install').mkdir()
    out = bash(
        f'source {LIB}; OP_DIR="{tmp_path}"; sleep() {{ :; }}; {STAMP_STUBS} '
        'triton_load_and_wait() { return 0; }; dc() { return 0; }; '
        f'model_setup_run_group {group}; echo "rc=$?"'
    )
    assert 'rc=0' in out.stdout
    rows = {
        r[0]: r
        for r in (
            line.split('\t')
            for line in (tmp_path / '.install' / 'precision.tsv').read_text().splitlines()
        )
    }
    assert set(rows) == set(ENGINES[group])
    for model, (_, precision) in ENGINES[group].items():
        assert rows[model][1] == precision
        assert rows[model][3:6] == ['v110100', '8.6', _sha(model)]
        assert rows[model][6]


@pytest.mark.parametrize('group', GROUPS)
def test_a_matching_stamp_lets_the_group_skip(tmp_path: Path, bash, group: str) -> None:
    prelude = _installed(tmp_path, group, _good_stamps(group))
    assert 'up to date, skipped' in bash(prelude + f'model_setup_run_group {group}').stdout


def _mismatches(group: str) -> dict[str, dict[str, str]]:
    first = next(iter(ENGINES[group]))
    last = list(ENGINES[group])[-1]
    other_sha = {**_good_stamps(group), first: _row(first, ENGINES[group][first][1], sha='0' * 64)}
    other_arch = {**_good_stamps(group), last: _row(last, ENGINES[group][last][1], cc='7.5')}
    fallback = {**_good_stamps(group), first: _row(first, 'fp32_fallback')}
    return {
        'ONNX changed': other_sha,
        'compute capability 7.5': other_arch,
        'built as fp32_fallback': fallback,
    }


@pytest.mark.parametrize('group', GROUPS)
def test_a_stale_or_missing_stamp_forces_a_rebuild(tmp_path: Path, bash, group: str) -> None:
    for reason, stamps in _mismatches(group).items():
        prelude = _installed(tmp_path, group, stamps)
        out = bash(prelude + f'model_setup_run_group {group}').stdout
        assert 'skipped' not in out, reason
        assert 'engine stamp is stale' in out, reason
        assert reason in out, reason


def test_precision_report_lists_only_recorded_fallbacks(tmp_path: Path, bash) -> None:
    stamps = {
        'mobileclip2_s2_text_encoder': _row('mobileclip2_s2_text_encoder', 'fp32'),
        'scrfd_10g_bnkps': _row('scrfd_10g_bnkps', 'fp32_fallback'),
    }
    (tmp_path / '.install').mkdir()
    (tmp_path / '.install' / 'precision.tsv').write_text(''.join(stamps.values()))
    out = bash(f'source {LIB}; OP_DIR="{tmp_path}" model_setup_precision_report').stdout
    assert out.strip() == 'scrfd_10g_bnkps\tfp32_fallback'


# Engines built before per-engine stamps (v0.6.0 stamped only pe_image_encoder).
PE_ROW = 'pe_image_encoder\tfp16\t\tv110100\t8.6\t' + 'a' * 64 + '\tflags\n'


def _unstamped(tmp_path: Path, group: str, *, pe_row: str | None = PE_ROW) -> str:
    """An installed group whose engines have no ledger rows, only (optionally) the PE row."""
    prelude = _installed(tmp_path, group, {})
    for model in ENGINES[group]:
        cfg = tmp_path / 'models' / model / 'config.pbtxt'
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text('input [\n  {\n    name: "x"\n    data_type: TYPE_FP32\n  }\n]\n')
    if pe_row is not None:
        (tmp_path / '.install' / 'precision.tsv').write_text(pe_row)
    # Any docker compose call is a rebuild step: leave a marker.
    return prelude + 'dc() { touch "$OP_DIR/rebuilt"; return 0; }; '


def _rows(tmp_path: Path) -> dict[str, list[str]]:
    text = (tmp_path / '.install' / 'precision.tsv').read_text()
    return {r[0]: r for r in (line.split('\t') for line in text.splitlines())}


@pytest.mark.parametrize('with_pe_row', [True, False])
@pytest.mark.parametrize('group', GROUPS)
def test_a_valid_unstamped_engine_is_adopted_not_rebuilt(
    tmp_path: Path, bash, group: str, with_pe_row: bool
) -> None:
    prelude = _unstamped(tmp_path, group, pe_row=PE_ROW if with_pe_row else None)
    out = bash(prelude + f'model_setup_run_group {group}').stdout
    assert 'up to date, skipped' in out
    assert not (tmp_path / 'rebuilt').exists()
    rows = _rows(tmp_path)
    for model, (_, precision) in ENGINES[group].items():
        assert rows[model][1] == precision
        assert rows[model][2].startswith('adopted: built before per-engine stamps')
        assert rows[model][3:6] == ['v110100', '8.6', _sha(model)]
    if with_pe_row:
        assert rows['pe_image_encoder'][5] == 'a' * 64  # the reference row is untouched
    # once adopted the engine is an ordinary stamped one
    again = bash(prelude + f'model_setup_run_group {group}').stdout
    assert 'up to date, skipped' in again


def test_an_adopted_yolo_stamp_records_the_config_input_datatype(tmp_path: Path, bash) -> None:
    prelude = _unstamped(tmp_path, 'yolo')
    cfg = tmp_path / 'models' / 'yolov11_small_trt_end2end' / 'config.pbtxt'
    cfg.write_text('input [\n  {\n    name: "images"\n    data_type: TYPE_UINT8\n  }\n]\n')
    assert 'up to date, skipped' in bash(prelude + 'model_setup_run_group yolo').stdout
    assert 'input=uint8' in _rows(tmp_path)['yolov11_small_trt_end2end'][6]


def _unadoptable(tmp_path: Path, group: str, case: str) -> str:
    first = next(iter(ENGINES[group]))
    pe_row = PE_ROW
    extra = ''
    if case == 'tensorrt differs from the PE stamp':
        pe_row = PE_ROW.replace('v110100', 'v100000')
    elif case == 'compute capability differs from the PE stamp':
        pe_row = PE_ROW.replace('\t8.6\t', '\t7.5\t')
    elif case == 'not READY in Triton':
        extra = 'triton_model_ready() { return 1; }; '
    elif case == 'no GPU compute capability':
        extra = '_ms_compute_cap() { :; }; '
    elif case == 'no TensorRT version':
        extra = '_ms_trt_version() { :; }; '
    prelude = _unstamped(tmp_path, group, pe_row=pe_row) + extra
    if case == 'ONNX missing':
        (tmp_path / ENGINES[group][first][0]).unlink()
    elif case == 'engine missing':
        (tmp_path / 'models' / PLANS[group][0] / '1' / 'model.plan').unlink()
    elif case == 'no config input datatype':
        (tmp_path / 'models' / first / 'config.pbtxt').unlink()
    elif case == 'engine empty':
        (tmp_path / 'models' / PLANS[group][0] / '1' / 'model.plan').write_text('')
    return prelude


UNADOPTABLE = [
    'tensorrt differs from the PE stamp',
    'compute capability differs from the PE stamp',
    'not READY in Triton',
    'no GPU compute capability',
    'no TensorRT version',
    'ONNX missing',
    'engine missing',
    'engine empty',
    'no config input datatype',
]


@pytest.mark.parametrize('case', UNADOPTABLE)
@pytest.mark.parametrize('group', GROUPS)
def test_an_unstamped_engine_that_cannot_be_verified_is_rebuilt(
    tmp_path: Path, bash, group: str, case: str
) -> None:
    prelude = _unadoptable(tmp_path, group, case)
    out = bash(prelude + f'model_setup_run_group {group}').stdout
    assert 'skipped' not in out, case
    assert 'engine stamp is stale (no build stamp' in out, case
    assert 'adopted' not in (tmp_path / '.install' / 'precision.tsv').read_text(), case
    assert (tmp_path / 'rebuilt').exists(), case


def test_a_missing_pe_stamp_is_never_adopted(tmp_path: Path, bash) -> None:
    prelude = (
        f'source {LIB}; OP_DIR="{tmp_path}"; MODEL_SETUP_TRITON_DIGEST=sha256:abc; {STAMP_STUBS}'
        'triton_model_ready() { return 0; }; triton_load_and_wait() { return 0; }; '
        'dc() { touch "$OP_DIR/rebuilt"; return 0; }; '
    )
    (tmp_path / '.install').mkdir()
    (tmp_path / '.install' / 'state.json').write_text('{"triton_image_digest": "sha256:abc"}')
    for rel in ('models/pe_image_encoder/1/model.plan', 'pytorch_models/pe_image_encoder.onnx'):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text('x')
    out = bash(prelude + 'model_setup_run_group pe').stdout
    assert 'engine stamp is stale (no build stamp' in out
    assert (tmp_path / 'rebuilt').exists()


def test_a_brace_style_config_input_is_adopted_not_rebuilt(tmp_path: Path, bash) -> None:
    # the shipped arcface config writes `input {` rather than `input [`
    prelude = _unstamped(tmp_path, 'faces')
    for model in ENGINES['faces']:
        cfg = tmp_path / 'models' / model / 'config.pbtxt'
        cfg.write_text(
            'input {\n  name: "input"\n  data_type: TYPE_FP32\n  dims: [3, 112, 112]\n}\n'
        )
    out = bash(prelude + 'model_setup_run_group faces').stdout
    assert 'up to date, skipped' in out
    assert not (tmp_path / 'rebuilt').exists()
    assert 'input=fp32' in _rows(tmp_path)['arcface_w600k_r50'][6]


@pytest.mark.parametrize('group', ['mobileclip', 'faces', 'ocr'])
def test_every_engine_of_a_group_is_stamped_when_docker_reads_stdin(
    tmp_path: Path, bash, group: str
) -> None:
    # `docker compose run` forwards stdin even with -T: the stamp loop must not lose its
    # remaining engines to the TensorRT version probe
    _put_onnx(tmp_path, group)
    (tmp_path / '.install').mkdir()
    out = bash(
        f'source {LIB}; OP_DIR="{tmp_path}"; _ms_compute_cap() {{ echo 8.6; }}; '
        'dc() { cat > /dev/null; echo "TensorRT v110100"; }; '
        f'_ms_record_group_stamps {group}; echo "rc=$?"'
    )
    assert 'rc=0' in out.stdout
    assert set(_rows(tmp_path)) == set(ENGINES[group])
