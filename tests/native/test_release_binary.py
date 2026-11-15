"""The release build of the native frame engine exists and starts (marker ``native``, no GPU).

CI builds it first (job ``native``); locally build it with
``cargo build --release --manifest-path native/Cargo.toml --locked``. A missing binary fails the
test: it never skips. Set ``CARGO_TARGET_DIR`` as for the build when it is not ``native/target``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.native

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'native' / 'target'))
BINARY = TARGET / 'release' / 'op-frame-engine'


def test_the_release_binary_starts_and_exits_cleanly() -> None:
    assert BINARY.is_file(), f'{BINARY} is missing: build the release profile first'
    done = subprocess.run(
        [str(BINARY)], capture_output=True, timeout=30, check=False
    )  # fixed path inside the checkout's build output
    assert done.returncode == 0, done.stderr.decode(errors='replace')
