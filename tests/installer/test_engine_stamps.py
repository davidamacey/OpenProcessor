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
    return f'{model}\t{precision}\t\tv110100\t{cc}\t{sha or _sha(model)}\ttrt-python-api\n'


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
    missing = {m: s for m, s in _good_stamps(group).items() if m != last}
    fallback = {**_good_stamps(group), first: _row(first, 'fp32_fallback')}
    return {
        'ONNX changed': other_sha,
        'compute capability 7.5': other_arch,
        'no build stamp': missing,
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
