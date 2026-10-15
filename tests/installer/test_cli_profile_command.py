"""``openprocessor profile <name>`` applies the profile it names (issue #281).

It rewrites the Triton model configs and records the choice in an existing ``.env``
(GPU_PROFILE and OP_INGEST_GPU_DECODE, the two profile keys something reads). It writes no
``docker-compose.override.yml`` and no TRITON_WORKERS / MAX_BATCH_SIZE / SHM_SIZE: nothing read them.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path


def _project(tmp_path: Path, env: str) -> Path:
    proj = tmp_path / 'proj'
    shutil.copytree(REPO_ROOT / 'config_templates', proj / 'config_templates')
    for cfg in (REPO_ROOT / 'models').glob('*/config.pbtxt'):
        dest = proj / 'models' / cfg.parent.name
        dest.mkdir(parents=True)
        shutil.copy(cfg, dest / 'config.pbtxt')
    (proj / '.env').write_text(env)
    return proj


def _profile(proj: Path, *args: str) -> subprocess.CompletedProcess:
    script = (
        f'PROJECT_DIR="{proj}"; SCRIPT_DIR="{REPO_ROOT}/scripts"; '
        'source "$SCRIPT_DIR/lib/colors.sh"; source "$SCRIPT_DIR/lib/gpu.sh"; '
        'load_config_lib() { source "$SCRIPT_DIR/lib/config.sh"; }; '
        'source "$SCRIPT_DIR/lib/cli_dev.sh"; cmd_profile ' + ' '.join(args)
    )
    return subprocess.run(
        ['bash', '-c', script],
        check=False,
        capture_output=True,
        text=True,
        env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'},
        timeout=60,
    )


def test_profile_updates_the_profile_keys_of_an_existing_env(tmp_path: Path) -> None:
    proj = _project(tmp_path, 'GPU_PROFILE=minimal\nOP_INGEST_GPU_DECODE=0\nOPENSEARCH_HEAP=5g\n')
    done = _profile(proj, 'full', '0')
    assert done.returncode == 0, done.stderr
    env = (proj / '.env').read_text()
    assert re.search(r'^GPU_PROFILE=full$', env, re.MULTILINE), env
    assert re.search(r'^OP_INGEST_GPU_DECODE=1$', env, re.MULTILINE), env
    assert re.search(r'^OPENSEARCH_HEAP=5g$', env, re.MULTILINE), env
    assert 'GPU_PROFILE=minimal' not in env


def test_profile_appends_the_keys_when_the_env_lacks_them(tmp_path: Path) -> None:
    proj = _project(tmp_path, 'API_PORT=4603\n')
    assert _profile(proj, 'standard', '0').returncode == 0
    env = (proj / '.env').read_text()
    assert re.search(r'^GPU_PROFILE=standard$', env, re.MULTILINE), env
    assert re.search(r'^API_PORT=4603$', env, re.MULTILINE), env


def test_profile_writes_no_override_file_and_no_dead_keys(tmp_path: Path) -> None:
    proj = _project(tmp_path, 'GPU_PROFILE=minimal\n')
    assert _profile(proj, 'full', '0').returncode == 0
    assert not (proj / 'docker-compose.override.yml').exists()
    assert not re.search(r'^(TRITON_WORKERS|MAX_BATCH_SIZE|SHM_SIZE)=', (proj / '.env').read_text())


def test_profile_restart_hint_names_the_real_command(tmp_path: Path) -> None:
    proj = _project(tmp_path, 'GPU_PROFILE=minimal\n')
    done = _profile(proj, 'full', '0')
    assert './openprocessor restart triton-server' in done.stdout + done.stderr
    assert 'scripts/openprocessor.sh' not in done.stdout + done.stderr


def test_generated_env_and_template_carry_no_dead_profile_keys(tmp_path: Path) -> None:
    proj = tmp_path / 'p'
    shutil.copytree(REPO_ROOT / 'config_templates', proj / 'config_templates')
    script = (
        f'PROJECT_DIR="{proj}"; source "{REPO_ROOT}/scripts/lib/config.sh"; '
        'generate_env_file standard 0'
    )
    done = subprocess.run(
        ['bash', '-c', script], check=False, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    dead = re.compile(r'^#?\s*(TRITON_WORKERS|MAX_BATCH_SIZE|SHM_SIZE)=', re.MULTILINE)
    assert not dead.search((proj / '.env').read_text())
    assert not dead.search((REPO_ROOT / 'env.template').read_text())
    assert 'generate_compose_override' not in (REPO_ROOT / 'scripts/lib/config.sh').read_text()
    for name in ('minimal', 'standard', 'full'):
        text = (REPO_ROOT / f'config_templates/profiles/{name}.json').read_text()
        assert '"workers"' not in text
        assert '"shm_size"' not in text
