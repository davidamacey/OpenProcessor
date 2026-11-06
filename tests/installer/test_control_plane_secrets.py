"""The control-plane secrets (Postgres, Valkey, Flower) are created once, mode 600, never in argv.

Spike S0 rule: files are owned by the installing user at mode 600 inside 700 directories. Postgres
reads its superuser password as root (it keeps DAC_OVERRIDE); Valkey copies its conf to a tmpfs;
the application image runs as uid 1000, like the existing secrets/vlm files.
"""

from __future__ import annotations

import os
import stat
import subprocess
from typing import TYPE_CHECKING

import pytest
from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path

LIB = REPO_ROOT / 'scripts' / 'lib' / 'secrets_control_plane.sh'
FILES = (
    'postgres/super_password',
    'postgres/owner_password',
    'postgres/app_password',
    'valkey/password',
    'valkey/valkey.conf',
    'flower/auth',
)
WRAPPED = ('od', 'tr', 'head', 'base64', 'cat', 'mkdir', 'chmod', 'install', 'printf', 'tee', 'mv')


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _shims(tmp_path: Path) -> tuple[Path, Path]:
    """Wrappers that log every argv, so a test can prove a secret never reached one."""
    bin_dir, log = tmp_path / 'shim-bin', tmp_path / 'argv.log'
    bin_dir.mkdir(exist_ok=True)
    for name in WRAPPED:
        real = next(
            (p for p in (f'/usr/bin/{name}', f'/bin/{name}') if os.access(p, os.X_OK)), None
        )
        if real is None:
            continue
        shim = bin_dir / name
        shim.write_text(f'#!/bin/bash\nprintf "%s\\n" "{name} $*" >> "{log}"\nexec {real} "$@"\n')
        shim.chmod(0o755)
    return bin_dir, log


def _run(
    root: Path, tmp_path: Path, call: str = 'ensure_control_plane_secrets'
) -> tuple[subprocess.CompletedProcess, Path]:
    bin_dir, log = _shims(tmp_path)
    done = subprocess.run(
        ['bash', '-c', f'source "{LIB}"; {call} "{root}"'],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={'PATH': f'{bin_dir}:/usr/bin:/bin', 'HOME': str(tmp_path)},
    )
    return done, log


def _values(root: Path) -> dict[str, str]:
    return {f: (root / 'secrets' / f).read_text() for f in FILES}


def test_fresh_install_creates_every_file_at_600_in_700_directories(tmp_path: Path) -> None:
    done, _ = _run(tmp_path, tmp_path)
    assert done.returncode == 0, done.stderr
    for rel in FILES:
        path = tmp_path / 'secrets' / rel
        assert path.is_file(), rel
        assert _mode(path) == 0o600, rel
        assert path.stat().st_size > 8, rel
    for d in ('secrets', 'secrets/postgres', 'secrets/valkey', 'secrets/flower'):
        assert _mode(tmp_path / d) == 0o700, d


def test_the_three_postgres_passwords_differ_and_the_valkey_conf_matches_its_password(
    tmp_path: Path,
) -> None:
    assert _run(tmp_path, tmp_path)[0].returncode == 0
    vals = _values(tmp_path)
    pgs = {vals[f'postgres/{n}_password'] for n in ('super', 'owner', 'app')}
    assert len(pgs) == 3
    assert vals['valkey/valkey.conf'] == f'requirepass {vals["valkey/password"]}\n'
    assert vals['flower/auth'].count(':') == 1


def test_a_rerun_keeps_every_value(tmp_path: Path) -> None:
    assert _run(tmp_path, tmp_path)[0].returncode == 0
    before = _values(tmp_path)
    assert _run(tmp_path, tmp_path)[0].returncode == 0
    assert _values(tmp_path) == before


def test_upgrade_of_an_install_with_only_the_old_secrets_creates_the_new_ones(
    tmp_path: Path,
) -> None:
    (tmp_path / 'secrets' / 'vlm').mkdir(parents=True)
    (tmp_path / 'secrets' / 'vlm' / 'env').write_text('keep-me')
    (tmp_path / 'secrets').chmod(0o755)  # an older install may have left it open
    done, _ = _run(tmp_path, tmp_path)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / 'secrets' / 'vlm' / 'env').read_text() == 'keep-me'
    assert _mode(tmp_path / 'secrets') == 0o700
    for rel in FILES:
        assert _mode(tmp_path / 'secrets' / rel) == 0o600, rel


def test_repair_recreates_only_the_deleted_file(tmp_path: Path) -> None:
    assert _run(tmp_path, tmp_path)[0].returncode == 0
    before = _values(tmp_path)
    (tmp_path / 'secrets' / 'postgres' / 'app_password').unlink()
    assert _run(tmp_path, tmp_path)[0].returncode == 0
    after = _values(tmp_path)
    assert after['postgres/app_password']
    assert after['postgres/app_password'] != before['postgres/app_password']
    assert {k: v for k, v in after.items() if k != 'postgres/app_password'} == {
        k: v for k, v in before.items() if k != 'postgres/app_password'
    }


def test_an_empty_file_is_regenerated_not_kept(tmp_path: Path) -> None:
    assert _run(tmp_path, tmp_path)[0].returncode == 0
    (tmp_path / 'secrets' / 'valkey' / 'password').write_text('')
    assert _run(tmp_path, tmp_path)[0].returncode == 0
    vals = _values(tmp_path)
    assert vals['valkey/password']
    assert vals['valkey/valkey.conf'] == f'requirepass {vals["valkey/password"]}\n'


def test_no_value_appears_in_argv_or_in_the_output(tmp_path: Path) -> None:
    done, log = _run(tmp_path, tmp_path)
    assert done.returncode == 0, done.stderr
    argv_log = log.read_text()
    assert argv_log, 'the shims saw nothing; the test would prove nothing'
    for rel, value in _values(tmp_path).items():
        for token in value.replace(':', '\n').replace(' ', '\n').split():
            if len(token) >= 16:
                assert token not in argv_log, rel
                assert token not in done.stdout + done.stderr, rel


def test_the_cropwright_secret_helper_still_loads_beside_it() -> None:
    done = subprocess.run(
        [
            'bash',
            '-c',
            (
                f'source "{REPO_ROOT}/scripts/lib/cropwright_auth.sh"; source "{LIB}"; '
                'declare -F ensure_control_plane_secrets >/dev/null'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    'key',
    ['OP_DB_APP_PASSWORD_FILE', 'OP_DB_OWNER_PASSWORD_FILE', 'OP_DB_SUPER_PASSWORD_FILE'],
)
def test_config_show_redacts_the_password_file_settings(tmp_path: Path, key: str) -> None:
    (tmp_path / '.env').write_text(f'{key}=/run/secrets/op_pg/x\nAPI_PORT=4603\n')
    done = subprocess.run(
        [
            'bash',
            '-c',
            (
                f'PROJECT_DIR="{tmp_path}"; source "{REPO_ROOT}/scripts/lib/colors.sh"; '
                f'source "{REPO_ROOT}/scripts/lib/cli_core.sh"; cmd_config show'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr
    assert f'{key}=***' in done.stdout
    assert 'API_PORT=4603' in done.stdout


def test_the_installing_users_ids_are_recorded_in_an_existing_env_and_kept(tmp_path: Path) -> None:
    (tmp_path / '.env').write_text('API_PORT=4603\nOP_UID=4242\n')
    assert _run(tmp_path, tmp_path)[0].returncode == 0
    text = (tmp_path / '.env').read_text()
    assert 'OP_UID=4242\n' in text, 'a value already there is never overwritten'
    assert f'OP_GID={os.getgid()}\n' in text
    assert 'API_PORT=4603\n' in text


def test_no_env_file_is_created_just_for_the_ids(tmp_path: Path) -> None:
    assert _run(tmp_path, tmp_path)[0].returncode == 0
    assert not (tmp_path / '.env').exists()
