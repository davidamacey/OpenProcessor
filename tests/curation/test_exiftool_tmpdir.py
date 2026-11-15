"""The private directory that holds ExifTool's named pipe is removed on every normal end of its
owner: ``close``, garbage collection without ``close``, interpreter exit, and a Celery pool child's
shutdown. ``TMPDIR`` points at ``tmp_path`` so the host's ``/tmp`` is never touched."""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from PIL import Image

from src.clients.exiftool_client import ExifToolClient
from src.services.curation import capture_read
from src.tasks import app as tasks_app


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def jpeg(tmp_path: Path) -> bytes:
    path = tmp_path / 'a.jpg'
    Image.new('RGB', (8, 8)).save(path)
    return path.read_bytes()


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    scratch = tmp_path / 'tmp'
    scratch.mkdir()
    monkeypatch.setenv('TMPDIR', str(scratch))
    monkeypatch.setattr('tempfile.tempdir', None)
    return scratch


def _left(scratch: Path) -> list[str]:
    return sorted(p.name for p in scratch.glob('op-exiftool-*'))


def test_close_removes_the_directory(scratch: Path, jpeg: bytes) -> None:
    c = ExifToolClient(timeout_s=20)
    c.read_bytes(jpeg)
    assert _left(scratch)
    c.close()
    assert not _left(scratch)


def test_garbage_collection_without_close_removes_the_directory(scratch: Path, jpeg: bytes) -> None:
    c = ExifToolClient(timeout_s=20)
    c.read_bytes(jpeg)
    assert _left(scratch)
    helper_process = c._process()
    del c
    gc.collect()
    assert not _left(scratch)
    if helper_process is not None:  # the leaked helper would otherwise outlive the test
        helper_process.kill()
        helper_process.wait(10)


def test_interpreter_exit_without_close_removes_the_directory(scratch: Path, jpeg: bytes) -> None:
    code = textwrap.dedent(
        """
        import sys
        from src.clients.exiftool_client import ExifToolClient
        c = ExifToolClient(timeout_s=20)
        c.read_bytes(sys.stdin.buffer.read())
        """
    )
    subprocess.run(
        [sys.executable, '-I', '-c', f'import sys; sys.path.insert(0, {str(ROOT)!r})\n{code}'],
        input=jpeg,
        env={'TMPDIR': str(scratch), 'PATH': '/usr/bin:/bin'},
        check=True,
        timeout=60,
        cwd=scratch,
    )
    assert not _left(scratch)


def test_a_celery_pool_child_shutdown_closes_the_capture_pool(
    scratch: Path, jpeg: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_read.shutdown_capture_pool()
    capture_read.start_capture_pool()
    try:
        pool = capture_read._pool()
        pool._client().read_bytes(jpeg)
        assert _left(scratch)
        tasks_app._on_worker_process_shutdown(pid=1)
        assert not _left(scratch)
        assert not capture_read.live_processes()
    finally:
        capture_read.shutdown_capture_pool()
