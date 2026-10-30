"""Ctrl-C during an install says so and how to continue (#363)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest
from installer_harness import PROJECT, RELEASE, SCRIPT


if TYPE_CHECKING:
    from installer_harness import Shimmed

# A pytest started in the background has SIGINT ignored and the child inherits
# that; bash cannot trap a signal ignored on entry, so reset it first.
RESET_SIGINT_THEN_EXEC = (
    'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); '
    'os.execv(sys.argv[1], sys.argv[1:])'
)


def _start_hanging_install(shimmed: Shimmed) -> subprocess.Popen[str]:
    shimmed.flag('allow_mutations')
    shimmed.flag('up_hangs')
    return subprocess.Popen(
        [
            sys.executable,
            '-c',
            RESET_SIGINT_THEN_EXEC,
            '/bin/bash',
            str(SCRIPT),
            '--unattended',
            '--dir',
            'inst',
            '--project',
            PROJECT,
            '--version',
            RELEASE,
            '--tiers',
            'core',
        ],
        cwd=str(shimmed.root),
        env=shimmed.env(OP_HEALTH_TIMEOUT='1'),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


@pytest.mark.parametrize(('sig', 'code'), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_an_interrupt_says_how_to_continue(
    shimmed: Shimmed, sig: signal.Signals, code: int
) -> None:
    proc = _start_hanging_install(shimmed)
    try:
        deadline = time.monotonic() + 60
        while not (shimmed.state / 'up_started').exists():
            assert proc.poll() is None, proc.communicate()[0]
            assert time.monotonic() < deadline
            time.sleep(0.1)
        os.killpg(proc.pid, sig)
        output, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert proc.returncode == code, output
    assert 'interrupted' in output
    assert 're-run the same command' in output
