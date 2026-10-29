"""The GPU decode follows the hardware: each installer profile says whether its card has
room for the gpu_image_prep model (about 8.5 GB on top of the profile's Triton models), the
installer writes OP_INGEST_GPU_DECODE from it, and ``openprocessor gpu plan`` says what is on."""

from __future__ import annotations

import json
import re
import subprocess
from typing import TYPE_CHECKING

import pytest
import test_cli_project
from installer_harness import REPO_ROOT
from test_cli_project import cli
from test_cli_vlm import set_env
from test_gpu_plan import gb, plan


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst

PROFILES = REPO_ROOT / 'config_templates' / 'profiles'
GPU_IMAGE_PREP_MIB = int(
    re.search(
        r'GPU_MIB_PER_INSTANCE"\s*value:\s*{\s*string_value:\s*"(\d+)"',
        (REPO_ROOT / 'models' / 'gpu_image_prep' / 'config.pbtxt').read_text(),
    ).group(1)  # type: ignore[union-attr]
)
# The free memory each profile's own models need
# (setup-openprocessor.sh _triton_need_gb, the same figures `gpu plan` checks the primary against).
MODELS_MIB = {'minimal': 8192, 'standard': 12288, 'full': 17408}
EXPECTED = {'minimal': False, 'standard': False, 'full': True}
A6000 = '0, NVIDIA RTX A6000, 49140, 100, 8.6\n'
TI_12GB = '0, NVIDIA GeForce RTX 3080 Ti, 12288, 100, 8.6\n'


def _installer_floor_mib(name: str) -> int:
    """The least free VRAM the installer plans the profile for (``_profile_min_free_gb``)."""
    done = subprocess.run(
        [
            'bash',
            '-c',
            f'OP_SOURCE_ONLY=1 source "{REPO_ROOT}/setup-openprocessor.sh"; _profile_min_free_gb {name}',
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return int(done.stdout.strip()) * 1024


def _profile(name: str) -> dict:
    return json.loads((PROFILES / f'{name}.json').read_text())


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_profile_turns_the_gpu_decode_on_only_where_the_card_has_room(name: str) -> None:
    flag = _profile(name)['ingest_gpu_decode']
    assert flag is EXPECTED[name]
    instances = _profile(name)['gpu_image_prep_instances']
    floor = _installer_floor_mib(name)
    fits = MODELS_MIB[name] + instances * GPU_IMAGE_PREP_MIB <= floor
    # The GPU decode is on only when the profile's smallest card holds it beside the models.
    assert flag == fits


def _shell(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['bash', '-c', script], capture_output=True, text=True, check=False, timeout=30
    )


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_load_profile_exports_the_flag(name: str) -> None:
    done = _shell(
        f'export PROJECT_DIR={REPO_ROOT}; source {REPO_ROOT}/scripts/lib/gpu.sh; '
        f'source {REPO_ROOT}/scripts/lib/colors.sh; load_profile {name}; '
        'echo "$PROFILE_INGEST_GPU_DECODE"'
    )
    assert done.stdout.strip() == ('1' if EXPECTED[name] else '0'), done.stderr


def _generate_env(tmp_path: Path, name: str, existing: str | None = None) -> str:
    shutil_root = tmp_path / 'proj'
    shutil_root.mkdir()
    (shutil_root / 'config_templates').symlink_to(REPO_ROOT / 'config_templates')
    if existing is not None:
        (shutil_root / '.env').write_text(existing)
    done = _shell(
        f'export PROJECT_DIR={shutil_root}; source {REPO_ROOT}/scripts/lib/colors.sh; '
        f'source {REPO_ROOT}/scripts/lib/gpu.sh; source {REPO_ROOT}/scripts/lib/opensearch_heap.sh; '
        f'source {REPO_ROOT}/scripts/lib/config.sh; opensearch_heap_for_host() {{ echo 2g; }}; '
        f'generate_env_file {name} 0 true'
    )
    assert done.returncode == 0, done.stderr + done.stdout
    return (shutil_root / '.env').read_text()


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_setup_writes_the_flag_for_the_profile(name: str, tmp_path: Path) -> None:
    text = _generate_env(tmp_path, name)
    assert f'\nOP_INGEST_GPU_DECODE={int(EXPECTED[name])}\n' in text


def test_a_forced_regeneration_keeps_an_explicit_choice(tmp_path: Path) -> None:
    text = _generate_env(tmp_path, 'minimal', existing='OP_INGEST_GPU_DECODE=1\n')
    assert '\nOP_INGEST_GPU_DECODE=1\n' in text


@pytest.mark.parametrize(
    ('gpus', 'profile'),
    [
        ([(0, gb(8), 0)], 'minimal'),
        ([(0, gb(20), 0)], 'standard'),
        ([(0, gb(48), 0), (1, gb(48), 0)], 'full'),
        ([(0, gb(24), 0), (1, gb(24), 0)], 'standard'),
        ([(0, gb(32), 0), (1, gb(32), 0)], 'standard'),
    ],
)
def test_installer_plan_matches_the_profile_file(
    gpus: list[tuple[int, int, int]], profile: str
) -> None:
    rc, kv, _ = plan(gpus)
    assert rc == 0
    assert kv['GPU_PROFILE'] == profile
    assert kv['OP_INGEST_GPU_DECODE'] == str(int(_profile(profile)['ingest_gpu_decode']))


def test_gpu_plan_says_the_gpu_decode_is_on_by_default(shimmed: Shimmed, inst: Path) -> None:
    shimmed.gpus(A6000)
    set_env(inst, GPU_PROFILE='full', TRITON_GPU_ID='0')
    env = inst / '.env'  # a checkout whose .env has no such line
    env.write_text(
        ''.join(
            ln + '\n'
            for ln in env.read_text().splitlines()
            if not ln.startswith('OP_INGEST_GPU_DECODE=')
        )
    )
    done = cli(shimmed, inst, 'gpu', 'plan')
    assert done.returncode == 0, done.stderr
    assert 'GPU decode (ingest): on (code default' in done.stdout
    assert 'need about' not in done.stdout + done.stderr


def test_gpu_plan_says_why_it_is_off(shimmed: Shimmed, inst: Path) -> None:
    shimmed.gpus(TI_12GB)
    set_env(inst, GPU_PROFILE='standard', TRITON_GPU_ID='0', OP_INGEST_GPU_DECODE='0')
    done = cli(shimmed, inst, 'gpu', 'plan')
    assert 'GPU decode (ingest): off (OP_INGEST_GPU_DECODE=0 in .env)' in done.stdout


def test_gpu_plan_warns_but_does_not_refuse_a_forced_on_card_without_room(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.gpus('0, NVIDIA RTX A6000, 24576, 100, 8.6\n')
    set_env(inst, GPU_PROFILE='standard', TRITON_GPU_ID='0', OP_INGEST_GPU_DECODE='1')
    done = cli(shimmed, inst, 'gpu', 'plan')
    assert 'GPU decode (ingest): on (OP_INGEST_GPU_DECODE=1 in .env)' in done.stdout
    assert 'need about 20992 MiB' not in done.stdout  # 24 GB holds 12288 + 8704
    shimmed.gpus(TI_12GB)
    done = cli(shimmed, inst, 'gpu', 'plan')
    assert 'need about 20992 MiB' in done.stdout + done.stderr
    assert 'ingest stays on the CPU path' in done.stdout + done.stderr


def test_gpu_plan_sizes_the_gpu_decode_by_the_installed_instance_count(
    shimmed: Shimmed, inst: Path
) -> None:
    shimmed.gpus(A6000)
    set_env(inst, GPU_PROFILE='full', TRITON_GPU_ID='0', OP_INGEST_GPU_DECODE='1')
    cfg = inst / 'models' / 'gpu_image_prep' / 'config.pbtxt'
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        (REPO_ROOT / 'models' / 'gpu_image_prep' / 'config.pbtxt')
        .read_text()
        .replace('count: 2', 'count: 3')
    )
    done = cli(shimmed, inst, 'gpu', 'plan')
    assert f'about {3 * GPU_IMAGE_PREP_MIB} MiB more on the primary card' in done.stdout
