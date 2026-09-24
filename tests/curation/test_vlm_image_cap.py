"""``OP_VLM_MAX_IMAGES_PER_CALL`` — the per-request VLM image cap is
deployment config, not a hardcoded 8.

The cap has to be <= the serving engine's per-prompt image limit or every
over-sized request 400s upstream, so it is both the labeler's default
chunk size *and* the hard clamp on explicit values. Import-time config,
so the end-to-end check runs in a fresh interpreter.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs this repo's own interpreter on a fixed snippet
import sys
from pathlib import Path

import pytest

from src.services.labeling import vlm_client


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_unset_keeps_default_of_8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_VLM_MAX_IMAGES_PER_CALL', raising=False)
    assert vlm_client._env_max_images_per_call() == 8


def test_env_sets_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VLM_MAX_IMAGES_PER_CALL', '6')
    assert vlm_client._env_max_images_per_call() == 6


@pytest.mark.parametrize('raw', ['six', '0', '-2'])
def test_malformed_cap_raises(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv('OP_VLM_MAX_IMAGES_PER_CALL', raw)
    with pytest.raises(ValueError, match='OP_VLM_MAX_IMAGES_PER_CALL'):
        vlm_client._env_max_images_per_call()


def test_env_cap_governs_labeler_default_and_hard_clamp() -> None:
    snippet = (
        'from src.services.labeling.vlm_labeler import VlmLabeler\n'
        'print(VlmLabeler().max_images_per_call, '
        'VlmLabeler(max_images_per_call=8).max_images_per_call, '
        'VlmLabeler(max_images_per_call=2).max_images_per_call)\n'
    )
    env = {**os.environ, 'OP_VLM_MAX_IMAGES_PER_CALL': '6'}
    out = subprocess.run(  # nosec B603
        [sys.executable, '-c', snippet],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == '6 6 2'
