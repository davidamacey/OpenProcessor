"""OpenSearch heap sizing from host RAM (projects plan 2.3, installer plan 11.1 #8).

The rule lives in exactly one place, scripts/lib/opensearch_heap.sh:
heap = clamp(floor(RAM_GiB / 2), 2, 30) GB. Both config generators (the
one-line installer and scripts/lib/config.sh) use it, and neither ever
overwrites an OPENSEARCH_HEAP the user already set.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAP_LIB = REPO_ROOT / 'scripts/lib/opensearch_heap.sh'
CONFIG_LIB = REPO_ROOT / 'scripts/lib/config.sh'
SETUP = REPO_ROOT / 'setup-openprocessor.sh'

GIB_KIB = 1024 * 1024


def _expected_heap(ram_gib: int) -> str:
    return f'{min(30, max(2, ram_gib // 2))}g'


def _meminfo(tmp_path: Path, ram_gib: int) -> Path:
    f = tmp_path / f'meminfo_{ram_gib}'
    f.write_text(
        f'MemTotal:       {ram_gib * GIB_KIB} kB\n'
        'MemFree:         1234567 kB\n'
        'MemAvailable:    2345678 kB\n'
    )
    return f


def _bash(script: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['bash', '-c', script],
        check=False,
        capture_output=True,
        text=True,
        env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', **env},
        timeout=60,
    )


@pytest.mark.parametrize(
    ('ram_gib', 'heap'),
    [(3, '2g'), (7, '3g'), (16, '8g'), (32, '16g'), (64, '30g'), (128, '30g')],
)
def test_heap_for_host_is_half_of_ram_clamped(tmp_path: Path, ram_gib: int, heap: str) -> None:
    assert _expected_heap(ram_gib) == heap
    result = _bash(
        f'source "{HEAP_LIB}"; opensearch_heap_for_host',
        OP_MEMINFO_PATH=str(_meminfo(tmp_path, ram_gib)),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == heap


@pytest.mark.parametrize(
    ('heap', 'per_gb', 'budget'),
    [
        ('1g', '', '20'),
        ('2g', '', '40'),
        ('4g', '', '80'),
        ('8g', '', '160'),
        ('3g', '30', '90'),
        ('2048m', '', '40'),
    ],
)
def test_soft_shard_budget(heap: str, per_gb: str, budget: str) -> None:
    result = _bash(f'source "{HEAP_LIB}"; opensearch_shard_budget "{heap}" "{per_gb}"')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == budget


def test_heap_for_host_fails_without_memtotal(tmp_path: Path) -> None:
    bad = tmp_path / 'meminfo'
    bad.write_text('MemFree: 1 kB\n')
    result = _bash(f'source "{HEAP_LIB}"; opensearch_heap_for_host', OP_MEMINFO_PATH=str(bad))
    assert result.returncode != 0
    assert result.stdout.strip() == ''


def test_clamp_logic_lives_in_exactly_one_place() -> None:
    setup = SETUP.read_text()
    config = CONFIG_LIB.read_text()
    assert 'opensearch_heap_gb' not in setup, 'the inline heap function must be gone'
    assert 'opensearch_heap_for_host' in setup
    assert 'opensearch_heap_for_host' in config
    assert 'scripts/lib/opensearch_heap.sh' in setup
    assert 'opensearch_heap.sh' in config
    for text in (setup, config):
        assert 'MemTotal' not in text
    manifest = (REPO_ROOT / 'release-manifest.txt').read_text().split()
    assert 'scripts/lib/opensearch_heap.sh' in manifest


def test_gpu_profiles_no_longer_carry_a_heap() -> None:
    for name in ('minimal', 'standard', 'full'):
        assert (
            'opensearch_heap'
            not in (REPO_ROOT / f'config_templates/profiles/{name}.json').read_text()
        )
    assert 'PROFILE_HEAP' not in (REPO_ROOT / 'scripts/lib/gpu.sh').read_text()
    assert 'PROFILE_HEAP' not in CONFIG_LIB.read_text()


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / 'proj'
    shutil.copytree(REPO_ROOT / 'config_templates', proj / 'config_templates')
    return proj


def _generate(proj: Path, meminfo: Path, fn: str) -> subprocess.CompletedProcess:
    return _bash(
        f'PROJECT_DIR="{proj}"; source "{CONFIG_LIB}"; {fn}',
        OP_MEMINFO_PATH=str(meminfo),
    )


def test_config_sh_env_file_heap_comes_from_host_ram(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    result = _generate(proj, _meminfo(tmp_path, 32), 'generate_env_file standard 0')
    assert result.returncode == 0, result.stderr
    assert re.search(r'^OPENSEARCH_HEAP=16g$', (proj / '.env').read_text(), re.MULTILINE)


def test_config_sh_forced_regen_keeps_a_user_heap(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    (proj / '.env').write_text('GPU_PROFILE=standard\nOPENSEARCH_HEAP=5g\n')
    result = _generate(proj, _meminfo(tmp_path, 32), 'generate_env_file standard 0 true')
    assert result.returncode == 0, result.stderr
    env = (proj / '.env').read_text()
    assert re.search(r'^OPENSEARCH_HEAP=5g$', env, re.MULTILINE)
    assert 'OPENSEARCH_HEAP=4g' not in env


def test_config_sh_override_interpolates_opensearch_heap(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    result = _generate(proj, _meminfo(tmp_path, 32), 'generate_compose_override standard 0')
    assert result.returncode == 0, result.stderr
    override = (proj / 'docker-compose.override.yml').read_text()
    assert (
        '"OPENSEARCH_JAVA_OPTS=-Xms${OPENSEARCH_HEAP:-2g} -Xmx${OPENSEARCH_HEAP:-2g}"' in override
    )
    assert not re.search(r'-Xm[sx][0-9]', override)


def test_compose_interpolates_heap_and_hard_codes_no_xmx() -> None:
    compose = (REPO_ROOT / 'docker-compose.yml').read_text()
    assert '"OPENSEARCH_JAVA_OPTS=-Xms${OPENSEARCH_HEAP:-2g} -Xmx${OPENSEARCH_HEAP:-2g}"' in compose
    assert not re.search(r'-Xm[sx](?!\$\{OPENSEARCH_HEAP)', compose)


def test_test_harness_compose_keeps_its_fixed_heap() -> None:
    test_compose = (REPO_ROOT / 'docker/test/compose.yml').read_text()
    assert 'OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m' in test_compose
    assert 'OPENSEARCH_HEAP' not in test_compose


def test_env_template_documents_heap_and_shard_knob() -> None:
    text = (REPO_ROOT / 'env.template').read_text()
    assert re.search(r'^OPENSEARCH_HEAP=2g$', text, re.MULTILINE)
    assert re.search(r'^# OP_SHARDS_PER_HEAP_GB=20$', text, re.MULTILINE)
    assert 'RAM/8' in text
