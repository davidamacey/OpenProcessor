"""M3: under `curl | bash` stdin is the script itself, so every prompt must
read from /dev/tty. These tests pipe the script on stdin while giving the
process a real pseudo-terminal as its controlling tty, then answer the
prompt through that terminal.
"""

from __future__ import annotations

import fcntl
import os
import pty
import select
import subprocess
import termios
import time
from typing import TYPE_CHECKING

from installer_harness import PROJECT, RELEASE, SCRIPT


if TYPE_CHECKING:
    from installer_harness import Shimmed


def run_with_tty(
    shimmed: Shimmed, args: list[str], answer: str, wait_for: bytes
) -> tuple[int, bytes, str]:
    master, slave = pty.openpty()

    def make_ctty() -> None:
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

    proc = subprocess.Popen(
        ['bash', '-c', f'cat "{SCRIPT}" | bash -s -- "$@"', '_', *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(shimmed.root),
        env=shimmed.env(),
        preexec_fn=make_ctty,  # noqa: PLW1509 -- needed to attach the pty as the controlling tty
        pass_fds=(slave,),
    )
    os.close(slave)
    seen = b''
    answered = False
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            seen += chunk
            if not answered and wait_for in seen:
                os.write(master, answer.encode() + b'\n')
                answered = True
        if proc.poll() is not None and not ready:
            break
    _, err = proc.communicate(timeout=60)
    os.close(master)
    return proc.returncode, seen, err.decode()


ARGS = [
    '--dry-run',
    '--dir',
    'inst',
    '--project',
    PROJECT,
    '--version',
    RELEASE,
    '--tiers',
    'core',
    '--bind',
    '0.0.0.0',
]


def test_piped_script_reads_consent_from_the_terminal(shimmed: Shimmed) -> None:
    rc, tty_out, err = run_with_tty(shimmed, ARGS, 'expose', b"Type 'expose' to continue: ")
    assert b"Type 'expose' to continue" in tty_out
    assert rc == 0, err[-2000:]
    env = (shimmed.root / 'inst' / '.env').read_text()
    assert 'OP_BIND_ADDRESS=0.0.0.0' in env


def test_wrong_answer_on_the_terminal_refuses(shimmed: Shimmed) -> None:
    rc, tty_out, err = run_with_tty(shimmed, ARGS, 'no', b"Type 'expose' to continue: ")
    assert b"Type 'expose' to continue" in tty_out
    assert rc == 9
    assert 'bind address not confirmed' in err


def test_without_a_terminal_the_prompt_fails_with_the_flag_to_use(shimmed: Shimmed) -> None:
    result = shimmed.run(ARGS, piped=True)
    assert result.returncode == 9
    assert 'OP_ALLOW_PUBLIC_BIND=1' in result.stderr
