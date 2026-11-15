"""The control hook of the harness's frame engine (``docker/test/fake_frame_engine.py``).

The engine reads its mode from ``verify-data/fake_engine_mode`` at every probe and request, so a
test makes it refuse and recover without a restart or a port: ``not_ready`` (down or restarting),
``unavailable`` (ready for the probe, every call UNAVAILABLE) or ``overloaded`` (every call
RESOURCE_EXHAUSTED). No file is normal operation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from .conftest import VERIFY_DATA_DIR


if TYPE_CHECKING:
    from collections.abc import Iterator

MODE_FILE = VERIFY_DATA_DIR / 'fake_engine_mode'
Refusal = Literal['not_ready', 'unavailable', 'overloaded']


def engine_normal() -> None:
    MODE_FILE.unlink(missing_ok=True)


@contextmanager
def engine_refuses(mode: Refusal = 'not_ready') -> Iterator[None]:
    """The engine refuses inside the block and is normal after it, whatever happens in it."""
    MODE_FILE.write_text(mode)
    try:
        yield
    finally:
        engine_normal()
