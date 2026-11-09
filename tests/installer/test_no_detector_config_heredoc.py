"""The detector's ``config.pbtxt`` is written by the exporter (``export/ultralytics_export.py``),
never by an installer heredoc: a shell copy of the tensor list drifts from the engine it describes
(it still declared the four EfficientNMS outputs after the engine became one ``output0 [300, 6]``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path

LIB = REPO_ROOT / 'scripts' / 'lib'
DETECTOR = 'yolov11_small_trt_end2end'
SCRFD = 'scrfd_10g_bnkps'


def _generate(tmp_path: Path, model: str, *, existing: str | None = None) -> Path:
    shutil.copytree(REPO_ROOT / 'config_templates', tmp_path / 'config_templates')
    (tmp_path / 'models' / model).mkdir(parents=True)
    config = tmp_path / 'models' / model / 'config.pbtxt'
    if existing is not None:
        config.write_text(existing)
    done = subprocess.run(
        [
            'bash',
            '-c',
            (
                f'PROJECT_DIR="{tmp_path}"; source "{LIB}/config.sh"; '
                f'generate_model_config {model} standard 0'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={'PATH': '/usr/bin:/bin'},
    )
    assert done.returncode == 0, done.stderr + done.stdout
    return config


def test_no_shell_library_writes_a_detector_config() -> None:
    offenders = [
        path.name
        for path in sorted(LIB.glob('*.sh'))
        if re.search(rf'name:\s*"{DETECTOR}"', path.read_text())
    ]
    assert not offenders, offenders


def test_profile_generation_does_not_write_the_detector_config(tmp_path: Path) -> None:
    assert not _generate(tmp_path, DETECTOR).exists()


def test_profile_generation_still_tunes_an_existing_detector_config(tmp_path: Path) -> None:
    config = _generate(
        tmp_path, DETECTOR, existing='name: "x"\nmax_batch_size: 64\ninstance_group [\n]\n'
    )
    assert 'max_batch_size: 32' in config.read_text()  # the standard profile


def test_the_other_generators_still_write_their_missing_configs(tmp_path: Path) -> None:
    assert f'name: "{SCRFD}"' in _generate(tmp_path, SCRFD).read_text()
