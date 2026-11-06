"""ENV-4: one .env generator (scripts/lib/env_generator.sh) for the installer and scripts/setup.sh."""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
from typing import TYPE_CHECKING

from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path

LIB = REPO_ROOT / 'scripts' / 'lib'
_ACTIVE_KEY = re.compile(r'^([A-Z_][A-Z0-9_]*)=', re.MULTILINE)


def _bash(script: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['bash', '-c', script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={'PATH': '/usr/bin:/bin', **env},
    )


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / 'proj'
    shutil.copytree(REPO_ROOT / 'config_templates', proj / 'config_templates')
    shutil.copy(REPO_ROOT / 'env.template', proj / 'env.template')
    return proj


def _checkout_generate(proj: Path, tmp_path: Path, force: str = 'false') -> None:
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text('MemTotal:       33554432 kB\n')
    done = _bash(
        f'PROJECT_DIR="{proj}"; source "{LIB}/config.sh"; generate_env_file standard 1 {force}',
        OP_MEMINFO_PATH=str(meminfo),
    )
    assert done.returncode == 0, done.stderr + done.stdout


def test_checkout_generator_writes_mode_600_and_every_active_template_key(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    _checkout_generate(proj, tmp_path)
    env = proj / '.env'
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    wanted = set(_ACTIVE_KEY.findall((proj / 'env.template').read_text()))
    assert wanted
    assert wanted <= set(_ACTIVE_KEY.findall(env.read_text()))
    text = env.read_text()
    assert re.search(r'^GPU_ID=1$', text, re.MULTILINE)
    assert re.search(r'^OPENSEARCH_HEAP=16g$', text, re.MULTILINE)


def test_the_file_is_never_world_readable_while_being_written(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    done = _bash(
        f'umask 022; source "{LIB}/env_generator.sh"; env_generate "{proj}/env.template" "{proj}/.env"'
    )
    assert done.returncode == 0, done.stderr
    assert stat.S_IMODE((proj / '.env').stat().st_mode) == 0o600


def test_an_existing_value_is_kept_and_new_template_keys_are_added(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    (proj / '.env').write_text('API_PORT=9999\nMY_OWN=1\n')
    (proj / '.env').chmod(0o644)
    done = _bash(
        f'source "{LIB}/env_generator.sh"; env_generate "{proj}/env.template" "{proj}/.env"'
    )
    assert done.returncode == 0, done.stderr
    text = (proj / '.env').read_text()
    assert re.search(r'^API_PORT=9999$', text, re.MULTILINE)
    assert 'API_PORT=4603' not in text
    assert 'MY_OWN=1' in text
    assert re.search(r'^TRITON_HTTP_PORT=4600$', text, re.MULTILINE)
    assert stat.S_IMODE((proj / '.env').stat().st_mode) == 0o600


def test_upsert_replaces_one_key_and_never_puts_the_value_in_argv(tmp_path: Path) -> None:
    env = tmp_path / '.env'
    env.write_text('A=1\nB=2\n')
    done = _bash(
        f'source "{LIB}/env_generator.sh"; env_set_key "{env}" B "x y"; env_get_key "{env}" B'
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'x y'
    assert env.read_text() == 'A=1\nB=x y\n'


def test_the_installer_and_setup_share_the_one_library() -> None:
    installer = (REPO_ROOT / 'setup-openprocessor.sh').read_text()
    assert 'scripts/lib/env_generator.sh' in installer
    assert 'env_generate "${OP_DIR}/env.template" "$ENV_FILE"' in installer
    assert 'env_create_or_merge' not in installer
    assert 'install -m 600 "${OP_DIR}/env.template"' not in installer
    assert 'env_generator.sh' in (LIB / 'config.sh').read_text()
    assert 'cat > "$env_file"' not in (LIB / 'config.sh').read_text()
    manifest = (REPO_ROOT / 'release-manifest.txt').read_text().split()
    assert 'scripts/lib/env_generator.sh' in manifest
