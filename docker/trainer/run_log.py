"""The run log: stdout and stderr teed to a scratch file, published through the storage seam.

This is what makes ``GET {api_prefix}/train/log/tail/{job_id}`` show Ultralytics' own progress
output. The log lives under the run's ``artifact_prefix`` (``run.log``); the scratch copy is
rewritten into the store atomically, so a reader never sees a half-written log.
"""

from __future__ import annotations

import contextlib
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from logutil import get_logger


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.clients.storage.base import BlobStore

logger = get_logger('trainer.run_log')


class RunLog:
    """A scratch log file and the storage key it is published to."""

    def __init__(self, scratch: Path, store: BlobStore, key: str) -> None:
        self.scratch = scratch
        self._store = store
        self._key = key
        self._published = -1

    def publish(self) -> None:
        """Copy the scratch file to the store when it grew since the last publish."""
        try:
            size = self.scratch.stat().st_size
        except OSError:
            return
        if size == self._published:
            return
        try:
            self._store.write_atomic(self._key, self.scratch.read_bytes())
        except OSError:
            logger.warning('run log publish failed', key=self._key, exc_info=True)
            return
        self._published = size

    def append(self, text: str) -> None:
        with contextlib.suppress(OSError), self.scratch.open('a', encoding='utf-8') as fh:
            fh.write(text)


@contextmanager
def tee_to_run_log(run_log_path: Path) -> Iterator[None]:
    """Tee stdout + stderr into ``run_log_path``."""
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = run_log_path.open('a', encoding='utf-8', buffering=1)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    class _Tee:
        def __init__(self, *streams: Any) -> None:
            self._streams = streams

        def write(self, data: str) -> int:
            for s in self._streams:
                with contextlib.suppress(Exception):
                    s.write(data)
                    s.flush()
            return len(data)

        def flush(self) -> None:
            for s in self._streams:
                with contextlib.suppress(Exception):
                    s.flush()

        def isatty(self) -> bool:
            return False

    try:
        sys.stdout = _Tee(orig_stdout, fp)
        sys.stderr = _Tee(orig_stderr, fp)
        yield
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        with contextlib.suppress(Exception):
            fp.close()
