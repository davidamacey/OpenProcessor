"""Atomic file replacement with a temp file unique to each writer.

A fixed temp name (``state.tmp``) is shared by every process that writes the same file, so
the second writer's ``replace()`` finds the temp file already renamed away and raises
``FileNotFoundError``. Each call here gets its own temp file in the target directory (same
filesystem, so ``os.replace`` is atomic); concurrent writers then only race on which whole
document wins.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so a reader sees the old or the new content, never a mix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        # mkstemp is 0600; other containers read these files, so match a normal write.
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_name).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp_name).unlink()
        raise


def atomic_write_json(
    path: Path, obj: Any, *, default: Callable[[Any], Any] | None = str, **dumps_kwargs: Any
) -> None:
    """Serialise ``obj`` as JSON and write it with :func:`atomic_write_bytes`.

    The serialisation happens before any file is created, so a value that cannot be encoded
    leaves nothing behind.
    """
    text = json.dumps(obj, default=default, **dumps_kwargs)
    atomic_write_bytes(path, text.encode('utf-8'))
