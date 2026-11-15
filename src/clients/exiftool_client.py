"""Read capture tags with ExifTool, one long-lived process per worker.

``exiftool`` is Phil Harvey's reference metadata reader (https://exiftool.org/); it is driven
through PyExifTool (https://sylikc.github.io/pyexiftool/), whose ``ExifToolHelper`` keeps one
``exiftool -stay_open True -@ -`` process and sends it batches of arguments, the documented way to
avoid paying the Perl start-up cost per file. The process starts on the first read, is shared by
every thread of the worker (a lock serialises the pipe, not any shared state between workers), and
is restarted after a failure.

A photo that is only in memory reaches the process through a named pipe (:meth:`read_bytes`), not
through a whole-file temporary file: ``-stay_open ... -@ -`` takes its commands on standard input
(https://exiftool.org/exiftool_pod.html, ``-stay_open``), so the file cannot also arrive there. A
short-lived thread writes the bytes into the pipe and ends quietly when ExifTool stops reading
(``-fast`` stops after the metadata segments, which closes the pipe early); every wait of that
thread is bounded by the same ``timeout_s`` and the same watchdog.

The process is started without ``preexec_fn`` (:class:`_ExifToolHelper`). PyExifTool passes
``preexec_fn`` to ``subprocess.Popen`` to arm the parent-death signal, and the Python
documentation (https://docs.python.org/3/library/subprocess.html, ``preexec_fn``) warns it "is NOT
SAFE to use in the presence of threads" and forces a real ``fork()``; gRPC does not support
``fork()`` while its threads have calls in flight (gRPC Python fork support,
https://github.com/grpc/grpc/blob/master/doc/fork_support.md), and a start-up between two engine
calls corrupted the connection of a call in flight in the 2026-11-13 speed chain. Without it,
``Popen`` uses ``vfork()`` on Linux "when it is safe to do so" (same page, "Disable use of
posix_spawn()"), which runs no fork handlers. The parent-death signal comes from ``setpriv
--pdeathsig TERM`` (util-linux, setpriv(1)), which sets it and executes ExifTool in place; it is
needed because ExifTool does not exit when its standard input closes. The pool starts its
processes before the process opens any gRPC channel
(:func:`src.services.curation.capture_read.start_capture_pool`); a process that is replaced later
is started the same way.

Every read is bounded by ``timeout_s``. PyExifTool has no timeout of its own and keeps reading a
dead pipe forever, so a watchdog kills the process on expiry and replaces the pipe it reads with a
descriptor that cannot be read, which ends the read with an ``OSError``.

Failures are typed: :class:`ExifToolError` means the reader itself failed (binary missing, process
died, timed out, unparseable answer), an outage the caller records as "capture unknown". A file
ExifTool cannot read is not an outage: its dictionary carries ``ExifTool:Error``. Messages and
log records name the exception type only: PyExifTool's own exception text embeds ExifTool's
output, which can hold a coordinate.
"""

from __future__ import annotations

import contextlib
import functools
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
import warnings
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from exiftool import ExifToolHelper
from exiftool.exceptions import ExifToolException, ExifToolVersionError

from src.core.logging import get_logger
from src.domain.capture_metadata import EXIFTOOL_TAGS, FILE_ERROR_KEY


if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

DEFAULT_TIMEOUT_S: Final = 10.0
#: ``-fast`` stops reading a JPEG after the metadata segments (exiftool.org, option ``-fast``).
_ARGS: Final = ('-fast',)
_PIPE_CHUNK: Final = 1 << 20
#: How often the writer looks again for a reader that has not opened the pipe yet, and for the
#: end of the read it serves.
_PIPE_POLL_S: Final = 0.002
_FEED_JOIN_S: Final = 5.0


#: The process ExifTool runs under: it sets the parent-death signal, then executes ExifTool in
#: place (``setpriv --pdeathsig <signal> <program>``, setpriv(1): "set or clear parent death
#: signal"), so the signal arrives when the thread that started the process ends.
_PDEATHSIG: Final = ('setpriv', '--pdeathsig', 'TERM', '--')


class _ExifToolHelper(ExifToolHelper):
    """PyExifTool's helper with its own process start: no ``preexec_fn`` (see the module doc).

    PyExifTool 0.5.6 (the newest release) has no switch for the ``preexec_fn`` its ``run`` passes;
    this ``run`` is that method's body with the same arguments, the same start-up check and the
    same exceptions, minus the hook. ``tests/curation/test_exiftool_spawn.py`` runs it against
    the real ExifTool, so a PyExifTool change that breaks the pieces it reads fails there."""

    def run(self) -> None:
        if self.running:
            warnings.warn('ExifTool already running; doing nothing.', UserWarning, stacklevel=2)
            return
        launcher = shutil.which(_PDEATHSIG[0])
        if launcher is None:
            msg = f'{_PDEATHSIG[0]} (util-linux) is not installed'
            raise FileNotFoundError(msg)
        argv = [launcher, *_PDEATHSIG[1:], str(self._executable), '-stay_open', 'True', '-@', '-']
        if self._common_args:
            argv += ['-common_args', *self._common_args]
        process = subprocess.Popen(  # nosec B603  fixed argv, no shell
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._process = process
        if process.poll() is not None:
            self._process = None  # type: ignore[assignment]  # PyExifTool types it as always set
            msg = 'exiftool did not execute successfully'
            raise RuntimeError(msg)
        self._running = True
        try:
            self._parse_ver()  # answers or raises: proves the process is ExifTool 12.15+
        except ValueError as exc:
            self.terminate()
            msg = 'exiftool did not answer its version: is it ExifTool 12.15 or newer?'
            raise ExifToolVersionError(msg) from exc


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


def _feed_pipe(path: Path, data: bytes, done: threading.Event, deadline: float) -> None:
    """Write ``data`` into the named pipe ``path`` until ExifTool has read what it wants.

    The pipe is opened non-blocking and retried until a reader is there: a blocking open would
    wait forever for a process that never opens it. A reader that goes away (``-fast`` stops
    after the metadata) is the normal end: ``BrokenPipeError`` is not an error. The thread also
    ends when ``done`` is set (the read ended some other way) or at ``deadline``."""
    fd = -1
    try:
        while fd < 0:
            if done.is_set() or time.monotonic() > deadline:
                return
            try:
                fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:  # ENXIO: no reader yet
                done.wait(_PIPE_POLL_S)
        view = memoryview(data)
        while view:
            if done.is_set() or time.monotonic() > deadline:
                return
            try:
                view = view[os.write(fd, view[:_PIPE_CHUNK]) :]
            except BlockingIOError:
                select.select([], [fd], [], _PIPE_POLL_S * 10)
            except OSError:  # the reader closed its end (EPIPE), or the pipe is gone
                return
    finally:
        if fd >= 0:
            os.close(fd)


def _remove_tree(directory: Path, owner_pid: int) -> None:
    """Remove ``directory`` unless this is a forked child of the process that made it."""
    if os.getpid() == owner_pid:
        shutil.rmtree(directory, ignore_errors=True)


class ExifToolClient:
    """One shared ``-stay_open`` ExifTool process; ``read`` returns the capture tags of a file."""

    def __init__(self, *, executable: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S):
        self._executable = executable
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._helper: ExifToolHelper | None = None
        self._aborted = threading.Event()
        self._pipe_path: Path | None = None
        self._pipe_cleanup: weakref.finalize | None = None

    def _process(self) -> subprocess.Popen[bytes] | None:
        # PyExifTool exposes no handle on its subprocess; the watchdog needs it to kill a hung one.
        return getattr(self._helper, '_process', None)

    def pid(self) -> int | None:
        """The running process's id, or ``None`` when none is running."""
        proc = self._process()
        return proc.pid if proc is not None and proc.poll() is None else None

    def _start(self) -> ExifToolHelper:
        # -n: numeric values, GPS as signed decimals; -G: group-prefixed keys. The same two
        # arguments PyExifTool's default common_args carry, spelled out because the parser reads
        # those key names (https://exiftool.org/exiftool_pod.html).
        helper = _ExifToolHelper(
            executable=self._executable, common_args=['-G', '-n'], check_execute=False
        )
        self._helper = helper
        helper.run()
        return helper

    def start(self) -> None:
        """Start the process now (a no-op when it runs). Call it from a thread that lives as long
        as the process, before any gRPC channel is opened (module doc). Raises
        :class:`ExifToolError` when it cannot start."""
        with self._lock:
            if self._helper is not None and self._helper.running:
                return
            try:
                self._start()
            except (ExifToolException, OSError, ValueError, RuntimeError) as exc:
                self._reset()
                msg = f'exiftool failed to start ({type(exc).__name__})'
                raise ExifToolError(msg) from exc

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

    def _attempts(self, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """``run`` under the lock with one fresh process tried when the first attempt failed; a
        timeout is not retried. Raises :class:`ExifToolError` on an outage."""
        for attempt in (1, 2):
            try:
                return run()
            except (ExifToolException, OSError, ValueError, RuntimeError) as exc:
                timed_out = self._aborted.is_set()
                self._reset()
                logger.warning(
                    'exiftool_read_failed',
                    attempt=attempt,
                    timed_out=timed_out,
                    error=type(exc).__name__,
                )
                if timed_out or attempt == 2 or isinstance(exc, FileNotFoundError):
                    msg = f'exiftool failed ({type(exc).__name__})'
                    raise ExifToolError(msg) from exc
        raise AssertionError  # unreachable: the loop returns or raises

    def read(self, path: str | Path) -> dict[str, Any]:
        """The capture tags of the file at ``path``. Raises :class:`ExifToolError` on an outage;
        one fresh process is tried when the first attempt failed, a timeout is not retried."""
        target = str(path)
        with self._lock:
            return self._attempts(lambda: self._run_once(target))

    def _pipe(self) -> Path:
        """This process's named pipe, in a private directory made on first use."""
        if self._pipe_path is None or not self._pipe_path.exists():
            self._drop_pipe()
            directory = Path(tempfile.mkdtemp(prefix='op-exiftool-'))  # mode 0700
            # Removed at close, at garbage collection and at interpreter exit (weakref.finalize
            # runs its callback at exit by default); not in a forked child, which does not own it.
            self._pipe_cleanup = weakref.finalize(self, _remove_tree, directory, os.getpid())
            self._pipe_path = directory / 'photo'
            os.mkfifo(self._pipe_path, 0o600)
        return self._pipe_path

    def _run_fed(self, data: bytes) -> dict[str, Any]:
        pipe = self._pipe()
        done = threading.Event()
        feeder = threading.Thread(
            target=_feed_pipe,
            args=(pipe, data, done, time.monotonic() + self._timeout_s),
            name='exiftool-feed',
            daemon=True,
        )
        feeder.start()
        try:
            return self._run_once(str(pipe))
        finally:
            done.set()
            feeder.join(_FEED_JOIN_S)
            if feeder.is_alive():  # unreachable by the writer's own bounds; never leave one behind
                logger.error('exiftool_feeder_stuck')
                self._reset()
                self._drop_pipe()

    def read_bytes(self, data: bytes) -> dict[str, Any]:
        """The capture tags of in-memory file ``data``, handed to the process through its named
        pipe (no temporary copy of the file). Raises :class:`ExifToolError` on an outage."""
        with self._lock:
            return self._attempts(lambda: self._run_fed(data))

    def _drop_pipe(self) -> None:
        if self._pipe_cleanup is not None:
            self._pipe_cleanup()
            self._pipe_cleanup = None
        self._pipe_path = None

    def strip_gps(self, src: str | Path, dest: str | Path) -> None:
        """Write ``dest`` as ``src`` without any GPS tag, losslessly (no re-encode; every other
        tag is kept): ``exiftool -GPS*= -o dest src``. ``-GPS*=`` also removes the XMP copy of
        the position that ``-gps:all=`` leaves (https://exiftool.org/exiftool_pod.html, "-TAG=").
        ``dest`` must not exist. Raises :class:`ExifToolError` when the strip did not succeed."""
        with self._lock:
            try:
                helper = self._helper or self._start()
                helper.execute('-GPS*=', '-o', str(dest), str(src))
            except (ExifToolException, OSError, ValueError, RuntimeError) as exc:
                self._reset()
                msg = f'exiftool GPS strip failed ({type(exc).__name__})'
                raise ExifToolError(msg) from exc
            if helper.last_status:
                msg = f'exiftool GPS strip failed (exit {helper.last_status})'
                raise ExifToolError(msg)

    def close(self) -> None:
        with self._lock:
            helper, proc = self._helper, self._process()
            self._helper = None
            if helper is not None and helper.running:
                with contextlib.suppress(ExifToolException, OSError):
                    helper.terminate(timeout=5)
            _discard(proc)
            self._drop_pipe()


@functools.cache
def get_exiftool_client() -> ExifToolClient:
    """The process-wide client: one ExifTool process per worker process, started on first read."""
    return ExifToolClient()


def _forget_inherited_process() -> None:
    """In a forked child: the parent's ExifTool pipes are not ours. Mark the inherited helper as
    not running, so its destructor never tells the parent's process to exit, and start fresh."""
    if get_exiftool_client.cache_info().currsize:
        helper = get_exiftool_client()._helper
        if helper is not None:
            helper._flag_running_false()
        get_exiftool_client.cache_clear()


os.register_at_fork(after_in_child=_forget_inherited_process)


__all__ = [
    'DEFAULT_TIMEOUT_S',
    'ExifToolClient',
    'ExifToolError',
    'get_exiftool_client',
]
