"""``gpu_image_prep`` instance count (#268): one value per installer profile.

Two instances on one card raised ingest from 31.8 to 48.0 images/s for one more copy of the model
(GPU_MIB_PER_INSTANCE each); a third added 0.6 percent. The count is a profile value like
``ocr_pipeline_instances`` so a small card is never given the second copy.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'models' / 'gpu_image_prep' / 'config.pbtxt'
PROFILES = ROOT / 'config_templates' / 'profiles'
NAMES = ('minimal', 'standard', 'full')
KEY = 'gpu_image_prep_instances'
EXPECTED = {'minimal': 1, 'standard': 1, 'full': 2}
VRAM_FLOOR_MIB = {'minimal': 6 * 1024, 'standard': 12 * 1024, 'full': 48 * 1024}
MODELS_MIB = {'minimal': 8192, 'standard': 12288, 'full': 17408}


def _profile(name: str) -> dict:
    return json.loads((PROFILES / f'{name}.json').read_text())


def _mib_per_instance() -> int:
    match = re.search(
        r'GPU_MIB_PER_INSTANCE"\s*value:\s*{\s*string_value:\s*"(\d+)"', CONFIG.read_text()
    )
    assert match
    return int(match.group(1))


def _committed_count() -> int:
    counts = re.findall(r'^\s*count:\s*(\d+)', CONFIG.read_text(), re.M)
    assert len(counts) == 1
    return int(counts[0])


@pytest.mark.parametrize('name', NAMES)
def test_profile_carries_the_count(name: str) -> None:
    assert _profile(name)[KEY] == EXPECTED[name]


@pytest.mark.parametrize('name', NAMES)
def test_profile_with_the_gpu_decode_on_fits_its_smallest_card(name: str) -> None:
    profile = _profile(name)
    if profile['ingest_gpu_decode']:
        used = MODELS_MIB[name] + profile[KEY] * _mib_per_instance()
        assert used <= VRAM_FLOOR_MIB[name]


def test_committed_config_is_the_full_profile_value() -> None:
    # A checkout that never runs the installer (the code default has the GPU decode on) gets the
    # measured best count; the smaller profiles are rewritten by the installer.
    assert _committed_count() == _profile('full')[KEY] == 2


def _generate(tmp_path: Path, name: str) -> str:
    models = tmp_path / 'models'
    shutil.copytree(ROOT / 'models' / 'gpu_image_prep', models / 'gpu_image_prep')
    shutil.copytree(ROOT / 'config_templates', tmp_path / 'config_templates')
    script = (
        f'export PROJECT_DIR={tmp_path}; '
        f'source {ROOT}/scripts/lib/gpu.sh; source {ROOT}/scripts/lib/config.sh; '
        f'generate_model_config gpu_image_prep {name} 1'
    )
    done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr + done.stdout
    return (models / 'gpu_image_prep' / 'config.pbtxt').read_text()


@pytest.mark.parametrize('name', NAMES)
def test_installer_applies_the_profile_count(name: str, tmp_path: Path) -> None:
    text = _generate(tmp_path, name)
    assert re.search(rf'count:\s*{EXPECTED[name]}\b', text), text
    assert re.search(r'gpus:\s*\[\s*1\s*\]', text)


def test_generate_all_configs_covers_gpu_image_prep() -> None:
    assert '"gpu_image_prep"' in (ROOT / 'scripts' / 'lib' / 'config.sh').read_text()


def test_docs_state_the_footprint_the_config_implies() -> None:
    full = _profile('full')[KEY] * _mib_per_instance()
    assert full == 17408
    for doc in ('docs/PERFORMANCE.md', 'INSTALLATION.md'):
        text = (ROOT / doc).read_text()
        assert f'{full:,} MiB' in text, f'{doc} does not state {full:,} MiB for the full profile'
