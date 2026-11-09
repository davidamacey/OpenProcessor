"""Read capture tags with ExifTool, one long-lived process per worker.

``exiftool`` is Phil Harvey's reference metadata reader (https://exiftool.org/); it is driven
through PyExifTool (https://sylikc.github.io/pyexiftool/), whose ``ExifToolHelper`` keeps one
``exiftool -stay_open True -@ -`` process and sends it batches of arguments, the documented way to
avoid paying the Perl start-up cost per file. The process starts on the first read, is shared by
every thread of the worker (a lock serialises the pipe, not any shared state between workers), and
is restarted after a failure.

Every read is bounded by ``timeout_s``. PyExifTool has no timeout of its own and keeps reading a
dead pipe forever, so a watchdog kills the process on expiry and replaces the pipe it reads with a
descriptor that cannot be read, which ends the read with an ``OSError``.

Failures are typed: :class:`ExifToolError` means the reader itself failed (binary missing, process
died, timed out, unparseable answer), an outage the caller records as "capture unknown". A file
ExifTool cannot read is not an outage: its dictionary carries ``ExifTool:Error``.
"""

from __future__ import annotations

import contextlib
import functools
import os
import tempfile
import threading
from typing import TYPE_CHECKING, Any, Final

from exiftool import ExifToolHelper
from exiftool.exceptions import ExifToolException

from src.core.logging import get_logger
from src.domain.capture_metadata import EXIFTOOL_TAGS, FILE_ERROR_KEY


if TYPE_CHECKING:
    import subprocess
    from pathlib import Path

logger = get_logger(__name__)

DEFAULT_TIMEOUT_S: Final = 10.0
#: ``-fast`` stops reading a JPEG after the metadata segments (exiftool.org, option ``-fast``).
_ARGS: Final = ('-fast',)


class ExifToolError(RuntimeError):
    """The ExifTool process could not read the file (an outage, not a property of the file)."""


def _unreadable_fd() -> int:
    """A descriptor ``select`` reports ready and ``os.read`` refuses (a write-only handle)."""
    return os.open(os.devnull, os.O_WRONLY)


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """Kill the process and make any thread blocked reading its pipes fail instead of spinning."""
    with contextlib.suppress(OSError):
        proc.kill()
    for pipe in (proc.stdout, proc.stderr):
        if pipe is None:
            continue
        fd = _unreadable_fd()
        try:
            os.dup2(fd, pipe.fileno())
        except (OSError, ValueError):
            pass
        finally:
            os.close(fd)


def _discard(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    _kill(proc)
    with contextlib.suppress(OSError, ValueError, TimeoutError):
        proc.wait(timeout=5)
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            with contextlib.suppress(OSError, ValueError):
                pipe.close()


class ExifToolClient:
    """One shared ``-stay_open`` ExifTool process; ``read`` returns the capture tags of a file."""

    def __init__(self, *, executable: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S):
        self._executable = executable
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._helper: ExifToolHelper | None = None
        self._aborted = threading.Event()

    def _process(self) -> subprocess.Popen[bytes] | None:
        # PyExifTool exposes no handle on its subprocess; the watchdog needs it to kill a hung one.
        return getattr(self._helper, '_process', None)

    def _start(self) -> ExifToolHelper:
        # -n: numeric values, GPS as signed decimals; -G: group-prefixed keys. The same two
        # arguments PyExifTool's default common_args carry, spelled out because the parser reads
        # those key names (https://exiftool.org/exiftool_pod.html).
        helper = ExifToolHelper(
            executable=self._executable, common_args=['-G', '-n'], check_execute=False
        )
        self._helper = helper
        helper.run()
        return helper

    def _run_once(self, path: str) -> dict[str, Any]:
        self._aborted.clear()
        timer = threading.Timer(self._timeout_s, self._abort)
        timer.start()
        try:
            helper = self._helper or self._start()
            rows = helper.get_tags(path, [*EXIFTOOL_TAGS, FILE_ERROR_KEY], params=list(_ARGS))
        finally:
            timer.cancel()
        return dict(rows[0]) if rows else {}

    def _abort(self) -> None:
        self._aborted.set()
        proc = self._process()
        if proc is not None:
            _kill(proc)

    def _reset(self) -> None:
        proc = self._process()
        self._helper = None
        _discard(proc)

    def read(self, path: str | Path) -> dict[str, Any]:
        """The capture tags of the file at ``path``. Raises :class:`ExifToolError` on an outage;
        one fresh process is tried when the first attempt failed, a timeout is not retried."""
        target = str(path)
        with self._lock:
            for attempt in (1, 2):
                try:
                    return self._run_once(target)
                except (ExifToolException, OSError, ValueError, RuntimeError) as exc:
                    timed_out = self._aborted.is_set()
                    self._reset()
                    logger.warning(
                        'exiftool_read_failed',
                        attempt=attempt,
                        timed_out=timed_out,
                        error=f'{type(exc).__name__}: {exc}',
                    )
                    if timed_out or attempt == 2 or isinstance(exc, FileNotFoundError):
                        msg = f'exiftool failed ({type(exc).__name__}: {exc})'
                        raise ExifToolError(msg) from exc
        raise AssertionError  # unreachable: the loop returns or raises

    def read_bytes(self, data: bytes, suffix: str = '') -> dict[str, Any]:
        """The capture tags of in-memory file ``data`` (written to a temporary file for ExifTool)."""
        with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
            handle.write(data)
            handle.flush()
            return self.read(handle.name)

    def close(self) -> None:
        with self._lock:
            helper, proc = self._helper, self._process()
            self._helper = None
            if helper is not None and helper.running:
                with contextlib.suppress(ExifToolException, OSError):
                    helper.terminate(timeout=5)
            _discard(proc)


@functools.cache
def get_exiftool_client() -> ExifToolClient:
    """The process-wide client: one ExifTool process per worker process, started on first read."""
    return ExifToolClient()


__all__ = [
    'DEFAULT_TIMEOUT_S',
    'ExifToolClient',
    'ExifToolError',
    'get_exiftool_client',
]
