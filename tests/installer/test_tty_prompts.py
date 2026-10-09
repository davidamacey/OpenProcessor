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
import tempfile
import termios
import time
from typing import TYPE_CHECKING

from installer_harness import PROJECT, RELEASE, SCRIPT


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed


def run_with_tty(
    shimmed: Shimmed, args: list[str], answer: str | list[str], wait_for: bytes, **env: str
) -> tuple[int, bytes, str]:
    """Pipe the script on stdin with a pty as the controlling terminal; each
    time `wait_for` shows up on the terminal, type the next answer."""
    answers = [answer] if isinstance(answer, str) else list(answer)
    master, slave = pty.openpty()

    def make_ctty() -> None:
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

    # Files, not pipes: a dry run prints more than a pipe buffer holds, and
    # nothing reads the pipes until the process ends.
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        proc = subprocess.Popen(
            ['bash', '-c', f'cat "{SCRIPT}" | bash -s -- "$@"', '_', *args],
            stdin=subprocess.DEVNULL,
            stdout=out_f,
            stderr=err_f,
            cwd=str(shimmed.root),
            env=shimmed.env(**env),
            preexec_fn=make_ctty,  # noqa: PLW1509 -- needed to attach the pty as the controlling tty
            pass_fds=(slave,),
        )
        os.close(slave)
        seen = b''
        answered = 0
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                seen += chunk
                while answered < len(answers) and seen.count(wait_for) > answered:
                    os.write(master, answers[answered].encode() + b'\n')
                    answered += 1
            if proc.poll() is not None:
                break
        proc.wait(timeout=60)
        os.close(master)
        err_f.seek(0)
        err = err_f.read().decode()
    return proc.returncode, seen, err


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
    # A dry run writes nothing; the consented bind shows in the summary.
    assert 'services are published on 0.0.0.0' in err
    assert not (shimmed.root / 'inst').exists()


def test_wrong_answer_on_the_terminal_refuses(shimmed: Shimmed) -> None:
    rc, tty_out, err = run_with_tty(shimmed, ARGS, 'no', b"Type 'expose' to continue: ")
    assert b"Type 'expose' to continue" in tty_out
    assert rc == 9
    assert 'bind address not confirmed' in err


def test_without_a_terminal_the_prompt_fails_with_the_flag_to_use(shimmed: Shimmed) -> None:
    result = shimmed.run(ARGS, piped=True)
    assert result.returncode == 9
    assert 'OP_ALLOW_PUBLIC_BIND=1' in result.stderr


TIER_ARGS = ['--dry-run', '--dir', 'inst', '--project', PROJECT, '--version', RELEASE]
TIER_PROMPT = b'Tiers to install ['


def test_invalid_tier_answer_reprompts_instead_of_installing_the_default(shimmed: Shimmed) -> None:
    rc, tty_out, err = run_with_tty(shimmed, TIER_ARGS, ['core,curaton', 'core'], TIER_PROMPT)
    assert tty_out.count(TIER_PROMPT) == 2
    assert 'unknown tier: curaton' in err
    assert rc == 0, err[-2000:]


def test_three_invalid_tier_answers_exit_2(shimmed: Shimmed) -> None:
    rc, tty_out, err = run_with_tty(
        shimmed, TIER_ARGS, ['bogus', 'core,curaton', 'nope'], TIER_PROMPT
    )
    assert tty_out.count(TIER_PROMPT) == 3
    assert rc == 2
    assert 'no valid tier list after 3 tries' in err


def test_exported_token_is_used_interactively_without_a_prompt(
    shimmed: Shimmed, tmp_path: Path
) -> None:
    tok = tmp_path / 'tok'
    tok.write_text('hf_TESTSECRET1234567890\n')
    tok.chmod(0o600)
    rc, tty_out, err = run_with_tty(
        shimmed,
        [*TIER_ARGS, '--tiers', 'segmenter'],
        [],
        b'HuggingFace token',
        HF_TOKEN_FILE=str(tok),
    )
    assert b'HuggingFace token (input hidden)' not in tty_out
    assert rc == 0, err[-2000:]


def test_answering_no_to_lan_access_keeps_cropwright_local(shimmed: Shimmed) -> None:
    shimmed.flag('allow_mutations')
    args = [
        '--no-start',
        '--dir',
        'inst',
        '--project',
        PROJECT,
        '--version',
        RELEASE,
        '--tiers',
        'cropwright',
    ]
    rc, tty_out, err = run_with_tty(shimmed, args, 'n', b'open the Cropwright web UI? [Y/n]: ')
    assert b'Let other computers on your LAN' in tty_out
    assert rc == 0, err[-2000:]
    cw_env = (shimmed.root / 'inst' / '.env').read_text()
    assert 'CROPWRIGHT_BIND_ADDRESS=127.0.0.1' in cw_env
