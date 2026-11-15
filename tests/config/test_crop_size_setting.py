"""``PE_CROP_SIZE`` is exactly 224 (the default) or 336: the two builds of the crop encoder, each
its own vector space. Anything else refuses start-up naming the variable."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _settings_size(value: str | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != 'PE_CROP_SIZE'}
    if value is not None:
        env['PE_CROP_SIZE'] = value
    return subprocess.run(
        [
            sys.executable,
            '-c',
            'from src.config.settings import TritonModelConfig as C; print(C.PE_CROP_SIZE)',
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    ('value', 'size'), [(None, '224'), ('', '224'), ('224', '224'), ('336', '336')]
)
def test_the_two_supported_sizes_and_the_default_start(value: str | None, size: str) -> None:
    done = _settings_size(value)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == size


@pytest.mark.parametrize('value', ['300', '448', '0', '-224', '224.0', 'big'])
def test_any_other_size_refuses_start_up_naming_the_variable(value: str) -> None:
    done = _settings_size(value)
    assert done.returncode != 0
    assert 'PE_CROP_SIZE' in done.stderr
    assert '224, 336' in done.stderr
