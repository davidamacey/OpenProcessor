"""Size-capped file reads for the small text files a dataset is described by."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


def read_bytes_capped(path: Path, cap: int) -> bytes | None:
    """The file's bytes, or ``None`` when it is over ``cap`` bytes or
    unreadable. Never reads more than ``cap + 1`` bytes."""
    try:
        with path.open('rb') as fh:
            raw = fh.read(cap + 1)
    except OSError:
        return None
    if len(raw) > cap:
        return None
    return raw


def read_text_capped(path: Path, cap: int) -> str | None:
    """The file's text, or ``None`` when it is over ``cap`` bytes or
    unreadable."""
    raw = read_bytes_capped(path, cap)
    return None if raw is None else raw.decode('utf-8', errors='replace')


__all__ = ['read_bytes_capped', 'read_text_capped']
