"""Instance counts in the committed Triton configs and the installer profiles.

``ocr_pipeline`` is a Python BLS model: one instance runs one request at a
time, so the instance count is the OCR concurrency. The count is bounded by
the per-profile Python-backend GPU memory budget, measured per instance and
recorded next to the count in the model config.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
PROFILES = ROOT / 'config_templates' / 'profiles'
PROFILE_NAMES = ('minimal', 'standard', 'full')
MEM_KEY = 'GPU_MIB_PER_INSTANCE'


def _instance_count(model: str) -> int:
    text = (MODELS / model / 'config.pbtxt').read_text()
    block = re.search(r'instance_group\s*\[(.*?)\n\]', text, re.S)
    assert block, f'{model}: no instance_group'
    counts = re.findall(r'^\s*count:\s*(\d+)', block.group(1), re.M)
    assert len(counts) == 1, f'{model}: expected one instance group'
    return int(counts[0])


def _mib_per_instance(model: str) -> int:
    text = (MODELS / model / 'config.pbtxt').read_text()
    match = re.search(rf'key:\s*"{MEM_KEY}"\s*value:\s*{{\s*string_value:\s*"(\d+)"', text)
    assert match, f'{model}: missing {MEM_KEY} parameter'
    return int(match.group(1))


def _profile(name: str) -> dict:
    return json.loads((PROFILES / f'{name}.json').read_text())


def _python_gpu_models() -> list[str]:
    found = []
    for cfg in sorted(MODELS.glob('*/config.pbtxt')):
        text = cfg.read_text()
        if re.search(r'^backend:\s*"python"', text, re.M) and 'KIND_GPU' in text:
            found.append(cfg.parent.name)
    return found


def test_ocr_pipeline_ships_more_than_one_instance() -> None:
    # Measured 21 images/s at one instance, 42 at two, 80 at four.
    assert _instance_count('ocr_pipeline') >= 2


def test_every_python_gpu_model_records_its_memory_per_instance() -> None:
    models = _python_gpu_models()
    assert 'ocr_pipeline' in models
    for model in models:
        assert _mib_per_instance(model) > 0


@pytest.mark.parametrize('name', PROFILE_NAMES)
def test_profile_ocr_instances_fit_the_python_backend_budget(name: str) -> None:
    profile = _profile(name)
    used = profile['ocr_pipeline_instances'] * _mib_per_instance('ocr_pipeline')
    assert profile['ocr_pipeline_instances'] >= 1
    assert used <= profile['python_backend_budget_mib'], (
        f'{name}: {used} MiB of Python instances exceeds the {name} budget'
    )


def _default_loaded() -> set[str]:
    compose = (ROOT / 'docker-compose.yml').read_text()
    return set(re.findall(r'--load-model=(\S+)', compose))


def test_committed_python_instances_fit_the_standard_budget() -> None:
    # The budget covers what Triton loads at start; a model loaded on demand (gpu_image_prep, loaded on
    # first use) records its own memory and is planned by the installer profile (ingest_gpu_decode).
    budget = _profile('standard')['python_backend_budget_mib']
    defaults = [m for m in _python_gpu_models() if m in _default_loaded()]
    assert 'ocr_pipeline' in defaults
    used = sum(_instance_count(m) * _mib_per_instance(m) for m in defaults)
    assert used <= budget


def test_profile_instances_grow_with_the_card() -> None:
    counts = [_profile(n)['ocr_pipeline_instances'] for n in PROFILE_NAMES]
    assert counts == sorted(counts)
    assert counts[0] == 1


@pytest.mark.parametrize('name', PROFILE_NAMES)
def test_installer_applies_the_profile_count_to_ocr_pipeline(name: str, tmp_path: Path) -> None:
    # config.sh derives MODELS_DIR from PROJECT_DIR, so run against a copy.
    models = tmp_path / 'models'
    shutil.copytree(MODELS / 'ocr_pipeline', models / 'ocr_pipeline')
    shutil.copytree(ROOT / 'config_templates', tmp_path / 'config_templates')
    script = (
        f'export PROJECT_DIR={tmp_path}; '
        f'source {ROOT}/scripts/lib/gpu.sh; source {ROOT}/scripts/lib/config.sh; '
        f'generate_model_config ocr_pipeline {name} 1'
    )
    done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr + done.stdout
    text = (models / 'ocr_pipeline' / 'config.pbtxt').read_text()
    assert re.search(rf'count:\s*{_profile(name)["ocr_pipeline_instances"]}\b', text), text
    assert re.search(r'gpus:\s*\[\s*1\s*\]', text)
