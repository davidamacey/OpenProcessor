"""TensorRT build settings (#42): one place decides workspace and engine max batch.

The workspace limit and the engine max-batch profile used to be literals in ten
exporters. These tests pin the table, the small build target, the fail-closed
parsing, the pairing with the small profile's Triton max_batch_size, and a
ratchet that keeps new literals out.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import trt_build_settings as tbs  # noqa: E402  # path set up above


GB = 1 << 30


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('OP_TRT_WORKSPACE_GB', 'OP_TRT_BUILD_TARGET', 'GPU_PROFILE'):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ('model', 'gb'),
    [
        ('yolo', 4),
        ('scrfd_10g_bnkps', 4),
        ('arcface_w600k_r50', 4),
        ('mobileclip2_s2_image_encoder', 4),
        ('mobileclip2_s2_text_encoder', 4),
        ('paddleocr_det_trt', 4),
        ('paddleocr_rec_trt', 8),
        ('pe_image_encoder', 8),
    ],
)
def test_default_workspace_matches_what_the_exporters_used_to_hardcode(model: str, gb: int) -> None:
    assert tbs.workspace_bytes(model) == gb * GB


def test_default_max_batch_is_untouched() -> None:
    assert tbs.max_batch('arcface_w600k_r50', 128) == 128


def test_small_target_uses_2gb_workspace_for_every_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_TRT_BUILD_TARGET', 'small')
    sizes = {tbs.workspace_bytes(m) for m in ('pe_image_encoder', 'yolo', 'paddleocr_rec_trt')}
    assert sizes == {2 * GB}


def test_small_profile_implies_the_small_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GPU_PROFILE', 'small')
    assert tbs.build_target() == 'small'
    monkeypatch.setenv('GPU_PROFILE', 'standard')
    assert tbs.build_target() == 'default'


def test_explicit_target_beats_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GPU_PROFILE', 'small')
    monkeypatch.setenv('OP_TRT_BUILD_TARGET', 'default')
    assert tbs.build_target() == 'default'


@pytest.mark.parametrize(
    ('default', 'expected'),
    [(128, 64), (64, 32), (32, 16), (16, 16), (8, 8), (4, 4)],
)
def test_small_target_halves_max_batch_but_never_below_the_profile_floor(
    monkeypatch: pytest.MonkeyPatch, default: int, expected: int
) -> None:
    monkeypatch.setenv('OP_TRT_BUILD_TARGET', 'small')
    assert tbs.max_batch('m', default) == expected


def test_floor_equals_the_small_profiles_triton_max_batch_size() -> None:
    profile = json.loads((ROOT / 'config_templates/profiles/small.json').read_text())
    assert profile['max_batch_size'] == tbs.SMALL_PROFILE_MAX_BATCH


def test_override_applies_to_all_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_TRT_WORKSPACE_GB', '3')
    assert tbs.workspace_bytes('pe_image_encoder') == 3 * GB
    assert tbs.workspace_bytes('yolo') == 3 * GB


def test_override_beats_the_small_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_TRT_BUILD_TARGET', 'small')
    monkeypatch.setenv('OP_TRT_WORKSPACE_GB', '6')
    assert tbs.workspace_bytes('yolo') == 6 * GB


def test_empty_values_are_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_TRT_WORKSPACE_GB', '')
    monkeypatch.setenv('OP_TRT_BUILD_TARGET', '')
    assert tbs.workspace_bytes('yolo') == 4 * GB
    assert tbs.build_target() == 'default'


@pytest.mark.parametrize('bad', ['0', '-1', 'abc', '2.5', '1e3', '4G', ' '])
def test_invalid_workspace_raises(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv('OP_TRT_WORKSPACE_GB', bad)
    with pytest.raises(ValueError, match='OP_TRT_WORKSPACE_GB'):
        tbs.workspace_bytes('yolo')


@pytest.mark.parametrize('bad', ['small2', 'Small', 'tiny', '1'])
def test_invalid_target_raises(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv('OP_TRT_BUILD_TARGET', bad)
    with pytest.raises(ValueError, match='OP_TRT_BUILD_TARGET'):
        tbs.build_target()


_SCAN_DIRS = ('export', 'scripts')
_LITERAL = re.compile(
    r'WORKSPACE_GB\s*=\s*\d'
    r'|set_memory_pool_limit\([^)]*(<<|\*\s*\(1)'
    r'|memPoolSize=workspace:\s*[0-9]'
    r'|TRT_WORKSPACE=\s*"?[0-9]'
)


def _scan() -> tuple[list[str], int]:
    hits: list[str] = []
    scanned = 0
    for d in _SCAN_DIRS:
        for path in (ROOT / d).rglob('*'):
            if path.suffix not in {'.py', '.sh'} or 'venv' in path.parts:
                continue
            scanned += 1
            for n, line in enumerate(path.read_text(errors='ignore').splitlines(), 1):
                if _LITERAL.search(line):
                    hits.append(f'{path.relative_to(ROOT)}:{n}: {line.strip()}')
    return hits, scanned


def test_no_workspace_literal_survives_outside_the_helper() -> None:
    hits, scanned = _scan()
    assert scanned > 50, 'the scan found almost no files; the ratchet would pass vacuously'
    allowed = ('export/trt_build_settings.py', 'scripts/lib/model_setup.sh')
    stray = [h for h in hits if not h.startswith(allowed)]
    assert not stray, 'workspace literals must come from trt_build_settings:\n' + '\n'.join(stray)
