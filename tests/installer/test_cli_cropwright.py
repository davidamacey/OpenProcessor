"""``openprocessor cropwright password``: rewrites the login, sets both .env keys, and never
prints the password."""

from __future__ import annotations

import fcntl
import os
import pty
import select
import stat
import subprocess
import time
from typing import TYPE_CHECKING

import test_cli_project
from test_cli_project import cli


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst


def _env(inst: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (inst / '.env').read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, _, value = line.partition('=')
            out[key] = value
    return out


def _assert_login_written(inst: Path, password: str) -> None:
    secrets = inst / 'secrets' / 'cropwright'
    assert (secrets / 'password').read_text() == password
    assert stat.S_IMODE((secrets / 'password').stat().st_mode) == 0o600
    hashed = (secrets / 'htpasswd').read_text()
    assert hashed.startswith('cropwright:$apr1$')
    assert password not in hashed
    env = _env(inst)
    assert env['OP_CROPWRIGHT_BASIC_AUTH'] == 'true'
    assert env['OP_CROPWRIGHT_BASIC_AUTH_FILE'] == './secrets/cropwright/htpasswd'
    assert password not in (inst / '.env').read_text()


def test_without_a_terminal_an_empty_answer_generates_a_password(
    shimmed: Shimmed, inst: Path
) -> None:
    # start_new_session in cli() leaves no controlling tty, so the prompt is skipped.
    result = cli(shimmed, inst, 'cropwright', 'password')
    assert result.returncode == 0, result.stderr
    password = (inst / 'secrets' / 'cropwright' / 'password').read_text()
    assert len(password) == 24
    _assert_login_written(inst, password)
    assert password not in result.stdout + result.stderr
    assert 'secrets/cropwright/password' in result.stdout + result.stderr
    first_hash = (inst / 'secrets' / 'cropwright' / 'htpasswd').read_text()
    again = cli(shimmed, inst, 'cropwright', 'password')
    assert again.returncode == 0, again.stderr
    assert (inst / 'secrets' / 'cropwright' / 'htpasswd').read_text() != first_hash


def test_a_typed_password_is_used_and_never_echoed(shimmed: Shimmed, inst: Path) -> None:
    typed = 'correct-horse-battery-staple'  # gitleaks:allow
    master, slave = pty.openpty()

    def make_ctty() -> None:
        os.setsid()
        import termios

        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

    proc = subprocess.Popen(
        [str(inst / 'openprocessor'), 'cropwright', 'password'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(shimmed.root),
        env=shimmed.env(),
        preexec_fn=make_ctty,  # noqa: PLW1509 -- attaches the pty as the controlling tty
        pass_fds=(slave,),
        text=True,
    )
    os.close(slave)
    seen = b''
    sent = False
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and proc.poll() is None:
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            try:
                seen += os.read(master, 4096)
            except OSError:
                break
            if not sent and b'New Cropwright password' in seen:
                os.write(master, typed.encode() + b'\n')
                sent = True
    out, err = proc.communicate(timeout=30)
    os.close(master)
    assert proc.returncode == 0, err
    assert sent
    _assert_login_written(inst, typed)
    assert typed not in out + err
    assert typed.encode() not in seen.replace(typed.encode() + b'\r\n', b'')
