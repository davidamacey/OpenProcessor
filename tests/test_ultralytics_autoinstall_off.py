"""Ultralytics never installs a package at run time (``YOLO_AUTOINSTALL=False``).

The setting is read from the environment when ``ultralytics.utils`` is imported
(``AUTOINSTALL = env_bool("YOLO_AUTOINSTALL", True)``); with it on, a missing requirement is
``pip install``-ed from inside an export, a probe or a training run. Every image that carries
Ultralytics sets it, and the test environment does too.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - the test interpreter with a fixed script
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize('dockerfile', ['Dockerfile', 'docker/trainer/Dockerfile'])
def test_every_image_with_ultralytics_turns_autoinstall_off(dockerfile: str) -> None:
    text = (REPO_ROOT / dockerfile).read_text(encoding='utf-8')
    assert 'YOLO_AUTOINSTALL=False' in text


def test_the_test_environment_turns_autoinstall_off() -> None:
    assert os.environ.get('YOLO_AUTOINSTALL') == 'False'


def test_ultralytics_reads_the_setting_at_import() -> None:
    env = {**os.environ, 'YOLO_AUTOINSTALL': 'False'}
    result = subprocess.run(  # nosec B603 - the test interpreter, fixed script
        [sys.executable, '-I', '-c', 'import ultralytics.utils as u; print(u.AUTOINSTALL)'],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=env,
    )
    assert result.stdout.strip().splitlines()[-1] == 'False'
