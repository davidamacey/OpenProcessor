"""ExifTool is started without ``preexec_fn`` and before the process opens a gRPC channel.

``preexec_fn`` "is NOT SAFE to use in the presence of threads" and forces a real ``fork()``
(https://docs.python.org/3/library/subprocess.html); gRPC does not support ``fork()`` while its
threads have calls in flight (https://github.com/grpc/grpc/blob/master/doc/fork_support.md). The
2026-11-13 speed chain lost four photos (``INTERNAL: Stream removed``) in requests whose first
capture read started the process between two engine calls.

The real ExifTool and the real ``setpriv`` run here; only ``subprocess.Popen`` is watched.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from src.clients.exiftool_client import ExifToolClient


if TYPE_CHECKING:
    from collections.abc import Iterator


class _Spawns:
    """The ``Popen`` calls made while a test runs, argv and keyword arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def exiftool_calls(self) -> list[tuple[list[str], dict[str, Any]]]:
        return [c for c in self.calls if any('-stay_open' in a for a in c[0])]


@pytest.fixture
def spawns(monkeypatch: pytest.MonkeyPatch) -> _Spawns:
    seen = _Spawns()
    real = subprocess.Popen

    class Watching(real):  # type: ignore[valid-type, misc]
        def __init__(self, args: Any, *a: Any, **kw: Any) -> None:
            seen.calls.append((list(args), dict(kw)))
            super().__init__(args, *a, **kw)

    monkeypatch.setattr(subprocess, 'Popen', Watching)
    return seen


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    path = tmp_path / 'p.jpg'
    Image.new('RGB', (8, 8)).save(path, 'JPEG')
    return path


@pytest.fixture
def client() -> Iterator[ExifToolClient]:
    c = ExifToolClient(timeout_s=20)
    yield c
    c.close()


def test_the_process_is_started_without_preexec_fn(
    spawns: _Spawns, client: ExifToolClient, photo: Path
) -> None:
    assert client.read(photo)  # the first read starts the process, as a late start does
    [(argv, kwargs)] = spawns.exiftool_calls()
    assert kwargs.get('preexec_fn') is None, 'a real fork() of a process with gRPC calls in flight'
    assert argv[:4] == [argv[0], '--pdeathsig', 'TERM', '--']
    assert Path(argv[0]).name == 'setpriv'
    assert client.pid() is not None


def test_start_runs_the_process_before_the_first_read(
    spawns: _Spawns, client: ExifToolClient, photo: Path
) -> None:
    client.start()
    assert client.pid() is not None
    assert len(spawns.exiftool_calls()) == 1
    assert client.read(photo)
    assert len(spawns.exiftool_calls()) == 1  # the read found it running


def test_a_process_replaced_after_it_died_is_started_the_same_way(
    spawns: _Spawns, client: ExifToolClient, photo: Path
) -> None:
    assert client.read(photo)
    first = client.pid()
    assert first is not None
    os.kill(first, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while client.pid() is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert client.read(photo)  # one fresh process is tried when the read found it dead
    assert client.pid() not in (None, first)
    calls = spawns.exiftool_calls()
    assert len(calls) == 2
    assert all(kw.get('preexec_fn') is None for _argv, kw in calls)


def test_exiftool_ends_when_the_process_that_started_it_dies(tmp_path: Path) -> None:
    """The parent-death signal the removed hook set is still set: ExifTool does not exit when its
    standard input closes, so without it a killed worker leaves a process behind."""
    child = (
        'import sys, time\n'
        'from src.clients.exiftool_client import ExifToolClient\n'
        'c = ExifToolClient(); c.start()\n'
        'print(c.pid(), flush=True)\n'
        'time.sleep(60)\n'
    )
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
    parent = subprocess.Popen(
        [sys.executable, '-c', child], stdout=subprocess.PIPE, env=env, cwd=tmp_path
    )
    try:
        assert parent.stdout is not None
        exiftool_pid = int(parent.stdout.readline())
        os.kill(exiftool_pid, 0)  # running
        parent.kill()
        parent.wait(10)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(exiftool_pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        os.kill(exiftool_pid, signal.SIGKILL)
        pytest.fail('ExifTool outlived the process that started it')
    finally:
        parent.kill()
        parent.wait(10)
