"""Optional basic auth in front of Cropwright (#351): the container entrypoint, the nginx
config and the login helper the installer and `openprocessor cropwright password` share."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from typing import TYPE_CHECKING

import pytest
from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path

FRONTEND = REPO_ROOT / 'frontend'
LIB = REPO_ROOT / 'scripts' / 'lib' / 'cropwright_auth.sh'
HASH = 'cropwright:$apr1$abcdefgh$' + 'A' * 22  # shape only; never a real credential
needs_openssl = pytest.mark.skipif(shutil.which('openssl') is None, reason='openssl missing')


class Sandbox:
    """The entrypoint with its absolute container paths rewritten under tmp_path."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.src = root / 'htpasswd-src'
        self.auth_dir = root / 'etc/nginx/auth.d'
        (root / 'etc/nginx/conf.d').mkdir(parents=True)
        (root / 'etc/nginx/snippets').mkdir(parents=True)
        (root / 'html').mkdir()
        shutil.copy(FRONTEND / 'nginx.conf', root / 'etc/nginx/conf.d/default.conf')
        shutil.copy(
            FRONTEND / 'nginx-gateway.conf', root / 'etc/nginx/snippets/gateway-subpaths.conf'
        )
        text = (FRONTEND / 'docker-entrypoint.sh').read_text()
        text = text.replace('/run/secrets/cropwright_htpasswd', str(self.src))
        text = text.replace('/usr/share/nginx/html', str(root / 'html'))
        text = text.replace('/etc/nginx', str(root / 'etc/nginx'))
        self.script = root / 'entrypoint.sh'
        self.script.write_text(text)

    def run(self, **env: str) -> subprocess.CompletedProcess:
        clean = {k: v for k, v in os.environ.items() if not k.startswith(('OP_', 'CROPWRIGHT_'))}
        return subprocess.run(
            ['sh', str(self.script)],
            check=False,
            capture_output=True,
            text=True,
            env={**clean, **env},
            timeout=30,
        )


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def test_enabled_with_a_valid_login_writes_htpasswd_and_auth_basic(box: Sandbox) -> None:
    box.src.write_text(HASH + '\n')
    result = box.run(OP_CROPWRIGHT_BASIC_AUTH='true')
    assert result.returncode == 0, result.stderr
    conf = (box.auth_dir / 'basic-auth.conf').read_text()
    assert 'auth_basic "Cropwright";' in conf
    assert f'auth_basic_user_file {box.auth_dir}/htpasswd;' in conf
    written = box.auth_dir / 'htpasswd'
    assert written.read_text() == HASH + '\n'
    assert stat.S_IMODE(written.stat().st_mode) == 0o600
    assert HASH not in result.stdout + result.stderr


@pytest.mark.parametrize('contents', [None, '', '\n  \n'])
def test_enabled_without_a_login_refuses_to_start(box: Sandbox, contents: str | None) -> None:
    if contents is not None:
        box.src.write_text(contents)
    result = box.run(OP_CROPWRIGHT_BASIC_AUTH='true')
    assert result.returncode != 0
    assert 'no login is set' in result.stderr
    assert not (box.auth_dir / 'basic-auth.conf').exists()


@pytest.mark.parametrize(
    'line',
    [
        'cropwright:hunter2plaintext',
        'cropwright:{SHA}abc',
        ':$apr1$abcdefgh$' + 'A' * 22,
        'nocolon',
    ],
)
def test_enabled_with_a_malformed_login_refuses_and_does_not_echo_it(
    box: Sandbox, line: str
) -> None:
    box.src.write_text(HASH + '\n' + line + '\n')
    result = box.run(OP_CROPWRIGHT_BASIC_AUTH='true')
    assert result.returncode != 0
    assert 'not user:<apr1 or bcrypt hash>' in result.stderr
    assert line not in result.stdout + result.stderr
    assert not (box.auth_dir / 'basic-auth.conf').exists()


def test_a_bad_switch_value_refuses(box: Sandbox) -> None:
    result = box.run(OP_CROPWRIGHT_BASIC_AUTH='maybe')
    assert result.returncode != 0
    assert 'must be true/false' in result.stderr


def test_disabled_leaves_no_auth_and_removes_a_stale_one(box: Sandbox) -> None:
    box.src.write_text(HASH + '\n')
    assert box.run(OP_CROPWRIGHT_BASIC_AUTH='true').returncode == 0
    for env in ({}, {'OP_CROPWRIGHT_BASIC_AUTH': 'false'}):
        result = box.run(**env)
        assert result.returncode == 0, result.stderr
        assert list(box.auth_dir.glob('*')) == []
    rendered = (box.root / 'etc/nginx/conf.d/default.conf').read_text()
    assert 'auth_basic "' not in rendered


def test_nginx_config_applies_auth_to_every_location_but_health() -> None:
    conf = (FRONTEND / 'nginx.conf').read_text()
    assert conf.count('include /etc/nginx/auth.d/*.conf;') == 1
    assert conf.index('include /etc/nginx/auth.d/*.conf;') < conf.index('\n    location ')
    assert conf.count('auth_basic off;') == 1
    health = conf[conf.index('location = /nginx-health') :].split('}', 1)[0]
    assert 'auth_basic off;' in health
    assert 'return 200' in health
    # The gateway locations carry no auth directive of their own, so they inherit.
    assert 'auth_basic' not in (FRONTEND / 'nginx-gateway.conf').read_text()


def test_healthchecks_use_the_open_route() -> None:
    assert '/nginx-health' in (FRONTEND / 'Dockerfile').read_text()
    assert '/nginx-health' in (REPO_ROOT / 'docker-compose.yml').read_text()


@needs_openssl
def test_login_helper_output_passes_the_entrypoint_and_keeps_secrets_private(
    box: Sandbox, tmp_path: Path
) -> None:
    root = tmp_path / 'proj'
    root.mkdir()
    snippet = (
        f'source "{LIB}"; pw="$(cropwright_generate_password)" || exit 3; '
        f'cropwright_write_login "{root}" "$pw" || exit 4; printf "%s" "$pw" > "{tmp_path}/pw"'
    )
    done = subprocess.run(
        ['bash', '-c', snippet], check=False, capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, done.stderr
    password = (tmp_path / 'pw').read_text()
    assert len(password) == 24
    secrets = root / 'secrets' / 'cropwright'
    assert stat.S_IMODE((root / 'secrets').stat().st_mode) == 0o700
    assert stat.S_IMODE((secrets / 'password').stat().st_mode) == 0o600
    hashed = (secrets / 'htpasswd').read_text()
    assert hashed.startswith('cropwright:$apr1$')
    assert password not in hashed
    assert (secrets / 'password').read_text() == password
    shutil.copy(secrets / 'htpasswd', box.src)
    result = box.run(OP_CROPWRIGHT_BASIC_AUTH='true')
    assert result.returncode == 0, result.stderr
    assert password not in result.stdout + result.stderr
